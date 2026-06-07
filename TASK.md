# Auto-Segmented Prompt Cache Implementation Plan (Revised)

## Overview
Generalize the prompt cache's existing **single-boundary checkpoint** mechanism into
**multi-boundary auto-segmentation**, so multi-role conversations (especially
tool-heavy ones) can reuse cached prefixes at every role boundary instead of only
at the last-user-message boundary.

> **Why this is a rewrite of the previous plan:** the original plan inserted *one
> full-conversation KV cache object* under *multiple role-sliced keys*. That is
> incorrect. A prompt-cache entry maps a token prefix → the KV state produced by
> processing *exactly* those tokens. Reusing one full KV object under shorter keys
> returns KV state that does not correspond to the key, silently corrupting context.
> The original plan also re-tokenized each role's text independently, so the keys
> were not prefixes of the real token stream and would never match. This revision
> reuses the codebase's proven checkpoint/trim primitives instead.

---

## Ground truth (verified against current source)

| Item | Location | Notes |
| --- | --- | --- |
| `_InferenceContext` dataclass | `app/handler/mlx_lm.py:90` | Segment fields at lines 106-107 **already have defaults**; any new field must also have a default or precede them. |
| Non-batch cache insertion | `app/handler/mlx_lm.py:339` | `insert_cache(cache_key, cache)` — cache_key is the FULL prompt+generated token stream. |
| Serialized insert helper | `app/handler/mlx_lm.py:343` | `_insert_prompt_cache(key, cache, cache_type=...)` runs under `_generation_lock`. |
| Single-boundary checkpoint (non-trimmable) | `app/handler/mlx_lm.py:520-551` | Deep-copies KV state at one boundary via `checkpoint_callback`. **This is the correct segmentation pattern to generalize.** |
| Boundary computation | `app/handler/mlx_lm.py:354` `_compute_checkpoint_boundary` | Sentinel-substitution to find the last-user-message token boundary. |
| Batched-path segmentation | `app/handler/mlx_lm.py:461-471` | Builds `["system","assistant"]` two-segment split for non-trimmable caches. |
| Trim primitives | `app/handler/mlx_lm.py:622` | `from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache`. |
| `insert_cache` signature | `app/utils/prompt_cache.py:669` | `(tokens_ids, prompt_cache, *, cache_type="assistant", source="nonbatch")`. ✓ matches usage below. |
| `CACHE_FORMAT_VERSION` | `app/utils/prompt_cache.py:24` | Currently `1`; version mismatch skips old sidecars on rehydrate (`prompt_cache.py:517`). |
| Config flags | `app/config.py:57-59`, `294-296` | `prompt_cache_*` live on both `ServerConfig` and `ModelConfig`; mapped at `config.py:197-199`. |

---

## Design: two cache regimes

The correct segmentation strategy depends on whether the model's KV cache is trimmable.

### A. Trimmable caches (standard transformers)
The KV state can be truncated to any prefix length with `trim_prompt_cache`.
At insertion time we can derive each segment's state by deep-copying the final
cache and trimming it back to each boundary.

### B. Non-trimmable caches (hybrid / SSM / recurrent — Qwen3.5, Nemotron-H, Jamba)
The state **cannot** be trimmed. The only correct way to obtain prefix state is to
**checkpoint during prefill** (deep-copy the in-flight state when prefill reaches
the boundary). This is exactly what `checkpoint_callback` already does for a single
boundary — auto-segmentation registers *multiple* checkpoint boundaries instead.

> Boundaries in both regimes are expressed as **token offsets into the real token
> stream** (`input_ids` for prefill prefixes, `cache_key` for the full stream),
> never as independently re-tokenized role text.

---

## Phase 0: Boundary computation (shared)

**File:** `app/handler/mlx_lm.py`

Add a helper that returns the token offsets at each role boundary, generalizing
`_compute_checkpoint_boundary` from one boundary to all of them.

```python
def _compute_role_boundaries(
    self,
    messages: list[dict[str, Any]],
    input_ids: list[int],
    chat_template_kwargs: dict[str, Any],
) -> list[tuple[int, str]]:
    """Return ascending (token_offset, role) boundaries for the prompt.

    Each boundary marks the token position at which a new role block begins,
    computed by tokenizing cumulative message prefixes through the chat
    template and measuring the common-prefix length against ``input_ids``.

    Returns
    -------
    list[tuple[int, str]]
        Boundaries with 0 < offset < len(input_ids), ascending, each tagged
        with the role of the segment that ends at that offset.
    """
    ...
```

**Notes / open questions (resolve during implementation):**
- Reuse the sentinel-substitution approach already proven in
  `_compute_checkpoint_boundary` rather than naive `encode_prompt(messages[:k])`,
  because chat templates inject role headers / generation prompts that make naive
  prefixes diverge.
- Collapse consecutive same-role messages into one boundary (so a run of tool
  messages becomes one segment, matching the original intent).
- Must be cheap: cache template renders where possible; this runs per request.

---

## Phase 1: Configuration

**File:** `app/config.py`

Add `prompt_cache_auto_segment` to **both** config classes and the mapping at
`config.py:197-199`, mirroring the other `prompt_cache_*` fields:

```python
# ServerConfig (around line 57) and ModelConfig (around line 294)
prompt_cache_auto_segment: bool = False
```

```python
# mapping around config.py:197
prompt_cache_auto_segment=self.prompt_cache_auto_segment,
```

- `False` (default): existing single-boundary behavior — **no change**.
- `True`: register all role boundaries.

---

## Phase 2: `_InferenceContext` changes

**File:** `app/handler/mlx_lm.py:90`

Add fields **with defaults** (they follow the existing defaulted segment fields):

```python
    # Auto-segmentation: ascending (token_offset, role) boundaries for the
    # full prompt, populated only when prompt_cache_auto_segment is enabled.
    segment_boundaries: list[tuple[int, str]] | None = None
```

`refined_messages` is **not** added to the context — boundaries are precomputed in
`_build_inference_context()` where `refined_messages` and `input_ids` are already in
scope (`mlx_lm.py:430-437`), so no extra plumbing is needed.

Populate it in both return sites (`mlx_lm.py:473` batched, `mlx_lm.py:579` non-batch)
guarded by `if self.config.prompt_cache_auto_segment:`.

---

## Phase 3: Insertion logic

### 3A. Trimmable caches — non-batch path (`mlx_lm.py:336-341`)

```python
finally:
    if cache is not None:
        try:
            self._insert_segmented_cache(cache_key, cache, ctx.segment_boundaries)
        except Exception as cache_error:  # noqa: BLE001 - cache persistence is best-effort
            logger.warning(f"Failed to persist prompt cache: {cache_error}")
```

New helper:

```python
def _insert_segmented_cache(
    self,
    cache_key: list[int],
    cache: list[Any],
    boundaries: list[tuple[int, str]] | None,
) -> None:
    """Insert the full cache plus a trimmed entry per role boundary.

    Falls back to a single full-key insert when auto-segmentation is disabled,
    no boundaries exist, or the cache is non-trimmable (handled via checkpoints
    during prefill instead).
    """
    # Always insert the full entry (existing behavior).
    self.prompt_cache.insert_cache(cache_key, cache)

    if not boundaries or not self.model.cache_is_trimmable:
        return

    from mlx_lm.models.cache import trim_prompt_cache

    for offset, role in boundaries:
        prefix = cache_key[:offset]
        trimmed = copy.deepcopy(cache)
        # Trim the tail tokens so state corresponds to exactly `prefix`.
        trim_prompt_cache(trimmed, len(cache_key) - offset)
        self.prompt_cache.insert_cache(prefix, trimmed, cache_type=role)
```

> Each inserted entry's KV state now genuinely corresponds to its key — the
> correctness defect in the original plan is gone. Deep-copy + trim is the same
> primitive already used at `mlx_lm.py:622`.

### 3B. Non-trimmable caches — multi-checkpoint during prefill

Generalize the single `checkpoint_callback` (`mlx_lm.py:520-551`) to fire at each
boundary. The model's prefill loop currently supports one `checkpoint_position`;
extend it to accept a sorted list of `(position, prefix_ids, role)` and emit a
deep-copied checkpoint as prefill crosses each one. (Scope check: confirm the
generate loop can invoke the callback at multiple positions; if not, this part is a
follow-up and 3B ships as "single boundary only" initially.)

---

## Phase 4: Schema version bump

**File:** `app/utils/prompt_cache.py:24`

```python
CACHE_FORMAT_VERSION = 2
```

Old (v1) sidecars are skipped on rehydrate by the existing check at
`prompt_cache.py:517`; no manual cleanup needed.

---

## Phase 5: Testing

**Unit (`tests/handler/` + `tests/utils/`):**
- `_compute_role_boundaries`: single role, alternating roles, consecutive
  same-role collapse, tool blocks, empty/missing content, single message, empty list.
- `_insert_segmented_cache` trimmable: N boundaries → N+1 entries; each trimmed
  entry's token length matches its key length.
- `_insert_segmented_cache` non-trimmable: falls back to full insert only (no
  bogus trimmed entries).
- Disabled flag: byte-for-byte identical to current single-insert behavior.

**Integration:**
- End-to-end multi-turn reuse: second request hits a mid-conversation prefix.
- Persistence across restart with `CACHE_FORMAT_VERSION = 2`.
- Eviction accounting still balances (`_n_bytes` / `_n_bytes_by_type`) with many
  small segment entries.

**Edge cases:**
- Boundary at 0 or `len(cache_key)` is filtered out (no empty/full-dup entries).
- `trim_prompt_cache` unavailable (ImportError) → graceful fallback to full insert.

---

## Phase 6: Risks & notes

1. **Metadata overhead:** N boundaries create N extra trie + sidecar entries.
   Tool-heavy turns could add 10-20% metadata; eviction (`max_size`, `max_bytes`)
   must still hold — covered by Phase 5 accounting test.
2. **Deep-copy cost:** 3A deep-copies the cache once per boundary. For long
   contexts this is non-trivial; measure and consider trimming a single shared copy
   progressively (trim from longest prefix down) to avoid repeated full copies.
3. **Non-trimmable multi-checkpoint (3B)** depends on the generate loop supporting
   multiple checkpoint positions — verify before committing; otherwise ship 3A first.
4. **Boundary accuracy** is the correctness keystone: a wrong offset stores
   mismatched state. The sentinel technique must be validated against real chat
   templates in tests, not assumed.
5. **Backward compatibility:** with `prompt_cache_auto_segment=False`, every code
   path is unchanged.
```