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

### C1 (F3) — Unbounded allocation / panic in safetensors header parse (CRASH/DoS) — HIGH confidence, reachability TBD
- `x/safetensors/extractor.go:201-208` `OpenForExtraction`: reads 8-byte LE `headerSize` from attacker file, then `headerBytes := make([]byte, headerSize)` with NO bound → `makeslice: len out of range` panic (headerSize > maxInt) or OOM (e.g. 1<<48). Also uses `f.Read` not `io.ReadFull` (short read). Contrast: `fs/ggml/gguf.go` bounds everything (MaxArraySize etc.); `x/safetensors` has zero limits.
- Secondary: `GetTensor` (l.235) negative/OOB DataOffsets unvalidated → negative size (clamped by SectionReader, low sev).
- Reachability per F3: `x/create` import path via `xcreate` CLI. NEED TO CONFIRM whether server `/api/create` reaches this (remote) vs CLI-only.

## Blocked / Dead Ends
(none yet)
