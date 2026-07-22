# Performance Review

A performance-focused review of the Ollama codebase. Findings are ranked so
that **high-value + low-effort** items come first. File:line references are
included so each item is actionable.

**Scope note:** Ollama's *production* token sampling runs inside the
`llama.cpp` subprocess (`llm/llama_server.go` is a JSON-over-socket client), so
the Go-side per-token *sampler* findings apply to the **experimental MLX
runner** (`x/mlxrunner`). The convert / GGUF / tokenizer / server findings are
all mainline.

---

## Tier 1 — High value, low effort (do these first)

1. **GGUF numeric arrays parsed one element at a time.**
   `fs/ggml/gguf.go:542-555` calls `binary.Read` per element (fresh alloc +
   `io.ReadFull` each). Vocab arrays (`scores`, `token_type`) are 100k+
   elements → hundreds of thousands of tiny reads **on every model load and
   metadata/estimation scan**. Fix: single bulk `binary.Read(r, order,
   a.values)`. Biggest single parsing win.

2. **Oversized (discarded) numeric arrays are still read element-by-element.**
   `fs/ggml/gguf.go:543-552`: when the array exceeds `maxArraySize` (the common
   metadata-only case) the loop still decodes every element and throws it away.
   Strings have a discard fast-path; numbers don't. Fix:
   `Seek`/`io.CopyN(io.Discard, …)` past the block.

3. **`Template.Vars()` recomputed on every `Execute`.**
   `template/template.go:259` walks the whole parse tree, builds a map, and
   sorts keys **per request**, though the tree is immutable after `Parse`. Fix:
   compute once in `Parse`, cache on the struct.

4. **`GraphSize` rebuilds the layer map twice.**
   `fs/ggml/ggml.go:760` calls `f.Tensors().GroupLayers()["rope_freqs"]`
   re-splitting every tensor name, when `layers` was already built at `:662`.
   One-line fix: reuse `layers`.

5. **Rotating KV cache evaluates trace args per token, per layer.**
   `x/mlxrunner/cache/rotating.go:60,106`: `logutil.Trace(..., keys.Dims(),
   values.Dims(), ...)` — `Dims()` does cgo calls + slice allocs and
   `logutil.go:37` builds a `context.WithValue` **before** the level gate. Paid
   on every decode step even with tracing off. Fix: guard with
   `if slog.Default().Enabled(ctx, LevelTrace)` or drop the lines.

6. **Download progress takes a mutex on every ~32 KB chunk.**
   `server/download.go:121-127` locks `lastUpdatedMu` per `TeeReader` write
   across 16 parallel parts. Fix: make `lastUpdated` an `atomic.Int64`
   (UnixNano), drop the mutex.

7. **SentencePiece merge check allocates 4 strings per heap pop.**
   `tokenizer/sentencepiece.go:138` does `string(left.runes)`/
   `string(right.runes)` up to 4× just for empty/length checks, in the
   innermost BPE loop (runs on every prompt). Fix: `len(left.runes)==0` and
   compare byte lengths. Zero allocations.

8. **Byte-fallback uses `fmt.Sprintf("<0x%02X>", b)` per byte.**
   `tokenizer/bytepairencoding.go:271`, `tokenizer/sentencepiece.go:170` —
   reflection + alloc per unknown byte (fires for every non-ASCII/emoji token
   in some vocabs). Fix: precompute a package-level `[256]string` table and
   index it.

9. **`WriteGGUF` sort calls `fmt.Sscanf` O(n log n) times.**
   `fs/ggml/gguf.go:648` sorts via `Tensor.block()` (`ggml.go:395`), which runs
   `fmt.Sscanf(name, "blk.%d.", &n)` ~2× per comparison → tens of thousands of
   reflective parses. Fix: decorate-sort-undecorate (parse block number once
   per tensor).

10. **Safetensors copy capped at a 32 KB buffer.**
    `convert/reader_safetensors.go:210,217` wraps the same-dtype fast path in
    `bufio.NewReaderSize(r, min(32<<10, …))`, so multi-GB tensors copy 32 KB at
    a time. Fix: `io.CopyBuffer` with a ~1 MB buffer — ~32× fewer iterations
    per tensor.

11. **`sched.processPending` snapshots all loaded runners every retry
    iteration.** `server/sched.go:255-258` copies `s.loaded` before the
    already-loaded fast-path check at `:261`. Fix: build the snapshot only in
    the `else` branch that needs GPU discovery.

12. **MLX runner `Responses` channel is unbuffered.**
    `x/mlxrunner/server.go:121` — each token blocks decode until the HTTP
    handler's per-token `json.Encode` + `Flush` completes; flush jitter
    back-pressures generation. Fix: small buffer (16-32).

13. **`writeGGUFString` allocates a `strings.Reader` per string.**
    `fs/ggml/gguf.go:434` `io.Copy(w, strings.NewReader(s))` for every KV
    string / token. Fix: `io.WriteString(w, s)`.

14. **`readGGUFString` zeroes a buffer it's about to fully overwrite.**
    `fs/ggml/gguf.go:416-421`: `clear(buf)` right before `io.ReadFull(r, buf)`
    — dead work per KV key / string token. Fix: delete the `clear`.

15. **`skipLayer` compiles a regexp per tensor.**
    `convert/convert_deepseek2.go:144` and `convert/convert_glm4moelite.go:213`
    — the closure runs `regexp.MustCompile(` + "`^blk\\.(\\d+)`" + `)` inside a
    `for _, t := range s` loop over all tensors. Fix: hoist to a package-level
    `var`.

16. **`time.Tick` leaks a ticker in `GetDevicesFromRunner`.**
    `ml/device.go:816` — `time.Tick` can never be stopped/GC'd; this function
    returns, leaking a ticker per call (device discovery / model load). Fix:
    `time.NewTicker` + `defer t.Stop()`.

17. **`TrimSpace` computed twice per token.**
    `llm/llama_server.go:1652,1655` calls `strings.TrimSpace(lsResp.Content)`
    twice in the streaming loop. Fix: compute once.

18. **Single-int32 MLX arrays built via the reflection path in the speculative
    loop.** `x/mlxrunner/speculate.go:487,506,526` — one
    `mlx.FromValues([]int32{id}, 1)` per accepted/next draft token, each going
    through `binary.Encode`. A fast `NewArrayInt32` already exists
    (`mlx/ops_extra.go:570`). Fix: call it here.

---

## Tier 2 — High/medium value, moderate effort

19. **`Model.Capabilities()` re-opens & re-parses the GGUF (and every
    projector) ~4-6× per request.** `server/images.go:146,392`; invoked at
    `routes.go:2620`, via `scheduleRunner`→`CheckCapabilities` (`:217`),
    `getRunner` (`sched.go:180`), and `routes.go:2770`. Not memoized. Fix:
    compute once behind a `sync.Once`/cached field on `Model`.

20. **`GetModel` runs twice per generate/chat request.**
    `routes.go:2486` then again in `scheduleRunner` at `routes.go:208`
    (`GenerateHandler`: `:294`). Each re-parses the manifest + reads
    config/template/params blobs + opens the GGUF. Fix: thread the already-built
    `*Model` into `scheduleRunner`.

21. **`chatPrompt` truncation loop is O(n²).**
    `server/prompt.go:37-73`: for each start index it rebuilds the system slice
    (inner `for j := range i`), re-renders the full template, and re-tokenizes
    the whole conversation. On long contexts this is N full renders + N tokenize
    (CGO/IPC) calls. Fix: prompt length is monotonic in `i` → binary-search the
    fit (O(log N) tokenize calls); hoist system-message collection out of the
    loop.

22. **`mlx.FromValues` uses reflection + `binary.Encode` for every
    slice→array.** `x/mlxrunner/mlx/array.go:89,118-124` on the prefill path
    (up to ~2048 int32/chunk). Fix: add int32/float32 type-switch fast-paths
    passing the pointer directly, like `NewArrayInt32`.

23. **BPE/SP merge nodes reconvert `runes`→`string` on every touch.**
    `tokenizer/bytepairencoding.go:210,239,266,270`, `sentencepiece.go:115,159`
    — the `merge` struct stores only `[]rune`, so `pairwise`/validity rebuild
    the string form repeatedly per node. The newer `x/tokenizer` already caches
    a `token string` field. Fix: add a cached `token` field, set only on
    init/merge.

24. **`Vocabulary.Merge` builds a `left+" "+right` key per pair lookup.**
    `tokenizer/vocabulary.go:107` allocates a string for every candidate pair
    during encode. Fix: key the map on a small `struct{a,b string}` built from
    the cached node tokens.

25. **`GroupLayers` splits every tensor name into a fresh slice.**
    `fs/ggml/ggml.go:354` `strings.Split(t.Name, ".")` + re-`Join` per tensor
    (thousands of tensors). Fix: locate dot boundaries with `IndexByte` and
    slice directly; memoize `GroupLayers` on the `Tensors` value.

26. **`keyValue` rebuilds the arch-prefixed key and re-looks-up architecture per
    access.** `fs/ggml/ggml.go:318-329` allocates `arch + "." + key` and
    re-derives arch each call; `:187-205` do `append(defaultValue, "")`
    allocating a 1-element slice per `String/Uint/Float/Bool`. Called in loops
    from `GraphSize`/`HeadCount*`. Fix: cache arch once per `GraphSize`; pass
    scalar defaults.

27. **Per-step `Batch` + two length-1 `[]int32` slices allocate every
    decode/draft step.** `x/mlxrunner/pipeline.go:306-310`,
    `speculate.go:414-416`, `mtp.go:229-234` —
    `&batch.Batch{SeqOffsets:[]int32{…}, SeqQueryLens:[]int32{…}}` escapes to
    heap (3 allocs/token). Fix: reusable `Batch` value + reusable length-1
    slices overwritten each step.

28. **Serial sampler rebuilds a slot→row map every decode step.**
    `x/mlxrunner/sample/sample.go:730-733` `make(map[*slotState]int, …)` per
    `Sample` call; row indices only change on Add/Remove. Fix: maintain the
    index on `slotState`, reuse a pooled scratch slice.

29. **`Result.Arrays()` allocates a fresh 4-element slice on every lifecycle
    call.** `x/mlxrunner/sample/sample.go:43-45`, called several times per token
    from `Pin`/`Unpin`/`AsyncEval`. Fix: pass fields directly to the variadic
    `mlx.Pin`/`Unpin`, or cache the slice.

30. **`/api/tags` re-reads & re-parses every manifest on each call despite the
    cache.** `server/model_list_cache.go:145,165` always calls `syncManifests`→
    `manifest.Manifests(true)` (full dir read + per-file parse). Fix: throttle
    by manifests-dir mtime / rate-limit; serve from `entries` in between.

31. **Embedding path reloads full model metadata per request.**
    `server/routes.go:860` `getModelData(...)`→`llm.LoadModel` re-reads
    BOS/EOS/ctx-length KV on every embed call (high-rate, small-payload
    endpoint). Fix: cache the handful of tokenizer KV values on the loaded
    runner.

32. **`addSpecials` prepends BOS by copying the whole id slice.**
    `tokenizer/vocabulary.go:53` `append([]int32{BOS}, ids...)` on the
    per-prompt path. Fix: reserve the BOS slot up front / build back-to-front.

33. **`GenerateHandler` re-tokenizes prompt+response on Done.**
    `server/routes.go:717` `r.Tokenize(ctx, prompt+sb.String())` concatenates
    the full prompt (already tokenized) with the whole output into a new string
    just to fill `res.Context`. Fix: reuse the runner's final token sequence, or
    avoid the big concat.

34. **Special-token splitting is O(numSpecials × fragments) with per-split
    realloc.** `tokenizer/bytepairencoding.go:129-156`,
    `sentencepiece.go:57-81` — `append(fragments[:i], append(middle,
    fragments[i+1:]...)...)` per hit. Vocabs with hundreds of special tokens pay
    it quadratically per Encode. Fix: single-pass scanner (as in
    `x/tokenizer/tokenizer_encode.go`), or a trie/Aho-Corasick.

35. **`session.outputs` grows token-by-token with no preallocation.**
    `x/mlxrunner/pipeline.go:197,235` (declared `prefix_cache.go:58`). Fix:
    preallocate `cap = len(inputs)+NumPredict`.

---

## Tier 3 — Larger / structural

36. **MLX KV cache grows by a fixed 256-slot step via `Concatenate` → O(N²)
    copying.** `x/mlxrunner/cache/kvcache.go:42,75-91` reallocates and copies
    the entire live K/V every 256 tokens. Fix: geometric growth
    (`max(prev+L, prev*2)`) → O(log N) reallocs, O(N) total copy. (Preserve the
    lazy-snapshot/`rewound` invariants.)

37. **Global mutex + unbounded array registry on every MLX op, full scan per
    Sweep.** `x/mlxrunner/mlx/array.go:33-46` locks `arraysMu` and appends to a
    global slice for every array created; `:162-176` `Sweep` scans the whole
    list per token. Fix: generation/arena scheme (bulk-drop intermediates) or
    per-thread unlocked list. High effort.

38. **Top-p does a full-vocab argsort when `TopK` is unset.**
    `x/mlxrunner/sample/sample.go:861` `probs.Negative().ArgsortAxis(-1)` sorts
    the entire distribution every token when `TopK==0 && 0<TopP<1`. Fix:
    argpartition/top-k prefilter to bound the sort width.

39. **Logprobs allocate a `[]int` per token (and per top-logprob).**
    `server/logprob.go:33-44` stores one `int` per byte (8× the bytes) for every
    streamed token when logprobs are on. Fix: pool/reuse buffers.

40. **Snapshot store copies the whole token history.**
    `x/mlxrunner/prefix_cache.go:393,523` `c.key(append(s.inputs,
    s.outputs...))` — O(context) copy per snapshot (and risks aliasing
    `s.inputs`' backing array). Fix: reusable combined buffer / key off existing
    slices.

41. **Chat setup reallocates the full message slice twice per request.**
    `server/routes.go:2661` `append(m.Messages, req.Messages...)` (can alias
    `m.Messages`) plus the system-prepend at `:2663`, plus `slices.Clone` in
    `imageTaggedMessages` (`prompt.go:95`). Fix: one preallocated slice sized
    `len(m.Messages)+len(req.Messages)+1`.

42. **`PsHandler` reads `s.sched.loaded` without holding `loadedMu`.**
    `server/routes.go:2265` ranges the shared map while the scheduler mutates it
    under lock — primarily a data race, flagged alongside the shared-state
    locking items. Fix: take the short read lock (or a locked accessor) before
    snapshotting.

---

## Highest-ROI shortlist

If only a handful get done: **#1, #2** (GGUF parse — every model load),
**#3, #4** (per-request/per-schedule template & estimation waste), **#5**
(per-token trace overhead), **#6** (download lock), **#7 + #8** (per-prompt
tokenizer allocations), and **#19–#21** (repeated GGUF opens + O(n²) prompt
truncation on the request-setup path). Items #1–#18 are all small, contained
diffs.

---

### Verification notes

Items **#15** (regexp compiled per tensor in `convert_deepseek2.go` /
`convert_glm4moelite.go`), **#16** (`time.Tick` leak in `ml/device.go`), and
**#33** (re-tokenize on Done in `server/routes.go`) were verified directly
against the source. The remaining items were identified via a subsystem sweep
with cited file:line references; treat them as reviewed candidates and confirm
the surrounding context before implementing.
