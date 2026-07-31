# Prompt KV Cache

The prompt KV cache stores the key/value tensors produced while prefilling a
prompt so that a later request sharing a **prefix** of those tokens can skip
re-prefilling the shared part. It is a **language-model (`lm`) feature only**
and is keyed on the exact token sequence.

This page covers three things:

1. [Configuration](#configuration) — the knobs.
2. [Persistence](#persistence) — disk-backed caches that survive restarts (the
   ownership rules referenced from [on-demand-models.md](./on-demand-models.md)).
3. [Auto-segmentation](#auto-segmentation) — reusing prefixes mid-history.

---

## Configuration

| CLI flag                      | YAML key                    | Default | Meaning                                                                                                                                                                   |
|-------------------------------|-----------------------------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--prompt-cache-size`         | `prompt_cache_size`         | `10`    | Max number of cached prefix entries (LRU-evicted).                                                                                                                        |
| `--max-bytes`                 | `prompt_cache_max_bytes`    | `2^63`  | Max total bytes retained before byte-based eviction.                                                                                                                      |
| `--prompt-cache-dir`          | `prompt_cache_dir`          | none    | Directory for disk-backed payloads. **When set, the cache persists across restarts** (see below). When omitted, a process-local temp dir is used and removed on shutdown. |
| `--prompt-cache-auto-segment` | `prompt_cache_auto_segment` | `false` | Checkpoint the cache at role boundaries so prefixes can be reused mid-history (see below).                                                                                |

In multi-handler (`--config`) mode these are per-entry keys under `models:`; in
single-model mode they are the CLI flags above. See
[configuration.md](./configuration.md) for the full surface.

### In-memory behavior

- Entries are indexed in a **prefix trie**, so a new prompt reuses the longest
  cached prefix of its token sequence (`fetch_nearest_cache`).
- Eviction is **LRU**, bounded by both `prompt_cache_size` (entry count) and
  `prompt_cache_max_bytes` (total bytes).

---

## Persistence

By default the cache is purely in-process: payloads live in a temp directory
and the tokens→payload index lives in memory, so nothing survives a restart.
Setting `prompt_cache_dir` to a **caller-supplied directory** makes the cache
genuinely persistent.

### On-disk layout

Each entry is two files in the cache directory:

- `{uuid}.pkl` — the serialized KV payload.
- `{uuid}.meta.json` — a JSON **sidecar** carrying the trie key (`tokens`),
  `nbytes`, `cache_type`, `trimmable`, `source`, a schema version, and a
  **fingerprint** of the model + KV-cache configuration.

The sidecar is written **after** the payload via temp-then-`replace`, so a
crash can only ever orphan a payload (harmless — discarded on rehydrate), never
leave a sidecar pointing at a missing payload.

### Fingerprint gating

`build_cache_fingerprint(model_path, kv_bits, kv_group_size, quantized_kv_start,
mlx_version, format_version)` folds every input that affects payload layout into
one string (`app/utils/prompt_cache.py:27`). On startup only entries whose
sidecar fingerprint matches the **running** configuration are adopted, so a
cache built for one model or quantization is never loaded into another.

### Rehydration on startup

For a caller-supplied directory (not an owned temp dir), the constructor calls
`_rehydrate_from_disk` (`app/utils/prompt_cache.py:483`):

- The trie / LRU / byte accounting are rebuilt **from sidecars only**; payloads
  stay on disk and load lazily on first hit.
- Recency is approximated by sidecar mtime; `prompt_cache_size` and
  `prompt_cache_max_bytes` are re-applied (over-limit entries trimmed).
- Mismatched, stale (wrong version), corrupt, or orphaned entries are discarded
  and their files cleaned up.
- The directory is then **swept** of unreferenced `.pkl` files (e.g. from a
  pre-sidecar server version) and any interrupted-write `.tmp` / `.meta.json.tmp`
  files (`_sweep_orphan_files`). The result is logged as
  `N adopted, N skipped, N swept`.

### Directory ownership and `close()`

This is the rule the on-demand idle-unload path depends on
(`close()`, `app/utils/prompt_cache.py:762`):

| Directory                                                     | `close()` behavior                                                                                                           |
|---------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| **Owned** (auto-created temp dir, `prompt_cache_dir` omitted) | Payloads cleared and the directory `rmtree`'d.                                                                               |
| **Caller-supplied** (`prompt_cache_dir` set)                  | Only the **in-memory index** is dropped (`_reset_in_memory_state`); payloads and sidecars are left on disk for the next run. |

Because graceful shutdown — and **on-demand / idle unload**, which routes
through `handler.cleanup() → prompt_cache.close()` — uses this path, a
configured `prompt_cache_dir` is **preserved** across restarts and idle
unloads, while an owned temp dir is still cleaned up.

---

## Auto-segmentation

By default a cache entry is created for the full prompt of a request, so two
requests only share cache if one's prompt is a prefix of the other's. With
`prompt_cache_auto_segment` enabled, the cache is **checkpointed at every role
boundary**, so requests that share only an earlier span — e.g. just the system
prompt, or a system prompt plus the first few turns — can reuse that span even
when their later content differs.

- **Trimmable caches**: each prefix entry is derived via `deepcopy` +
  `trim_prompt_cache`.
- **Non-trimmable caches** (hybrid SSM, e.g. Qwen3.5): the state is
  **checkpointed forward during prefill** at each non-terminal segment
  boundary, since these caches cannot be trimmed after the fact.

Role boundaries come from `_compute_role_boundaries` (sentinel substitution,
collapsing consecutive same-role messages) and are expressed as **token offsets
into the real stream**, so every reused prefix is KV-correct.

The flag is **opt-in and default off**; all existing cache paths are unchanged
when it is disabled. The non-trimmable checkpoint-forward path is covered by
`tests/test_non_trimmable_checkpoint.py`.

---

## Interaction with on-demand models

An on-demand model that is unloaded by its idle timer and later reloaded will
**rehydrate its prompt cache from disk** if it was configured with a persistent
`prompt_cache_dir`, rather than starting cold — this is a direct consequence of
the `close()` ownership rule above. See
[on-demand-models.md](./on-demand-models.md#interaction-with-the-persistent-prompt-cache).

---

## Source map

| Behavior                          | Location                                                   |
|-----------------------------------|------------------------------------------------------------|
| Fingerprint construction          | `app/utils/prompt_cache.py:27` (`build_cache_fingerprint`) |
| Sidecar write (temp-then-replace) | `app/utils/prompt_cache.py:379` (`_write_sidecar`)         |
| Rehydrate from sidecars           | `app/utils/prompt_cache.py:483` (`_rehydrate_from_disk`)   |
| Orphan/`.tmp` sweep               | `app/utils/prompt_cache.py:556` (`_sweep_orphan_files`)    |
| Prefix lookup                     | `app/utils/prompt_cache.py:586` (`fetch_nearest_cache`)    |
| Ownership-aware close             | `app/utils/prompt_cache.py:762` (`close`)                  |
