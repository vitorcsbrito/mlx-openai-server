# Auto-Segmented Prompt Cache Implementation Plan

## Overview
Add auto-segmentation to prompt cache insertion to improve cache efficiency for multi-role conversations with tool calls.

## Summary of Changes

| File                      | Change                                       | Lines |
| ------------------------- | -------------------------------------------- | ----- |
| `app/utils/prompt_cache.py` | Bump CACHE_FORMAT_VERSION to 2               | 1     |
| `app/handler/mlx_lm.py`     | Add auto-segmentation helper + insert logic  | ~60   |
| `app/config.py`             | Add prompt_cache_auto_segment to ModelConfig | 1     |
| **Total**                     |                                              | **~62**   |

---

## Phase 1: Configuration

**File:** `app/config.py`

Add to ModelConfig dataclass:
```python
@dataclass
class ModelConfig:
    prompt_cache_auto_segment: bool = False  # NEW FLAG
    prompt_cache_dir: str | None = None
    prompt_cache_max_bytes: int = 1 << 63
```

**Behavior:**
- `False` (default): Current manual segment handling
- `True`: Auto-segmentation enabled when inserting cache

---

## Phase 2: Auto-Segmentation Logic

**File:** `app/handler/mlx_lm.py`

Add helper function after existing imports:

```python
def auto_segment_messages(roles: list[dict]) -> tuple[list[list[int]], list[str]]:
    """Auto-segment message history by role boundaries.
    
    Creates segments at each role change. Tool calls in assistant blocks
    are included in the assistant segment.
    
    Args:
        roles: List of message dictionaries with 'role' key
        
    Returns:
        Tuple of (segments, segment_types) where each segment is a 
        list of token IDs and segment_type labels the segment
    """
    segments = []
    segment_types = []
    
    current_segment = []
    current_type = None
    
    for msg in roles:
        msg_type = msg["role"]
        
        if msg_type != current_type:
            if current_type:
                segments.append(current_segment)
                segment_types.append(current_type)
            current_segment = []
            current_type = msg_type
        
        current_segment.extend(tokenize(msg))
    
    if current_type:
        segments.append(current_segment)
        segment_types.append(current_type)
    
    return segments, segment_types
```

---

## Phase 3: Cache Insertion Logic

**File:** `app/handler/mlx_lm.py`

### 3.1 Add refined_messages to InferenceContext

**Location:** Around line 1315 where `_InferenceContext` is defined

**Current:**
```python
@dataclass
class _InferenceContext:
    rest_input_ids: list[int]
    cache: list[Any] | None
    cache_key: list[int]
    total_input_tokens: int
    total_cached_tokens: int
    model_params: dict[str, Any]
    parsers_result: ParserManager
    prompt_progress_callback: Callable | None
    checkpoint_position: int | None
    checkpoint_callback: Callable | None
    batched_segments: list[list[int]] | None
    batched_segment_types: list[str] | None
```

**Add:**
```python
    refined_messages: list[dict[str, Any]]  # NEW
```

### 3.2 Pass refined_messages to context

**Location:** Around line 1315 where `_InferenceContext()` is instantiated

**Add to constructor call:**
```python
return _InferenceContext(
    rest_input_ids=input_ids,
    cache=None,
    cache_key=input_ids[:],
    total_input_tokens=len(input_ids),
    total_cached_tokens=0,
    model_params=model_params,
    parsers_result=parsers_result,
    prompt_progress_callback=(make_prompt_progress_callback() if self.debug else None),
    checkpoint_position=None,
    checkpoint_callback=None,
    batched_segments=segments,
    batched_segment_types=segment_types,
    refined_messages=refined_messages,  # NEW
)
```

### 3.3 Modify cache insertion in _generate_streamed_response

**Location:** Around line 339 where `insert_cache()` is called

**Current code:**
```python
finally:
    if cache is not None:
        try:
            self.prompt_cache.insert_cache(cache_key, cache)
        except Exception as cache_error:  # noqa: BLE001 - cache persistence is best-effort
            logger.warning(f"Failed to persist prompt cache: {cache_error}")
```

**New code:**
```python
finally:
    if cache is not None:
        try:
            if self.config.prompt_cache_auto_segment:
                # Auto-segment the full generated response
                segments, segment_types = self.auto_segment_messages(ctx.refined_messages)
                
                # Insert each segment separately
                for i, segment in enumerate(segments):
                    self.prompt_cache.insert_cache(
                        segment, 
                        cache, 
                        cache_type=segment_types[i],
                        source="nonbatch"
                    )
            else:
                self.prompt_cache.insert_cache(cache_key, cache)
        except Exception as cache_error:  # noqa: BLE001 - cache persistence is best-effort
            logger.warning(f"Failed to persist prompt cache: {cache_error}")
```

---

## Phase 4: Schema Version Bump

**File:** `app/utils/prompt_cache.py`

Line 24:
```python
CACHE_FORMAT_VERSION = 2  # Bump from 1 to 2
```

**Impact:**
- Old sidecars (version 1) will be skipped during rehydrate
- Already handled by existing version check in `_rehydrate_from_disk()`
- No manual cleanup needed

---

## Phase 5: Testing Checklist

**Unit Tests:**
- [ ] Auto-segmentation with single role (user-only)
- [ ] Auto-segmentation with multiple role alternations
- [ ] Auto-segmentation with tool calls in assistant blocks
- [ ] Auto-segmentation with empty/missing role handling
- [ ] Cache insertion with auto-segmented data
- [ ] Cache retrieval with auto-segmented data
- [ ] Segment types array matches segments

**Integration Tests:**
- [ ] End-to-end cache flow with auto-segmentation enabled
- [ ] Cache persistence across restarts
- [ ] Memory usage comparison vs manual segments
- [ ] Eviction behavior under load

**Edge Cases:**
- [ ] Single message
- [ ] Empty message list
- [ ] Messages without content
- [ ] Malformed input handling

---

## Phase 6: Verification Steps

1. **Verify current segment handling:**
   - ✅ Located in `_build_inference_context()` at lines 461-471 (prefill)
   - ✅ Cache insertion at line 339 (generation)
   - ✅ Messages available as `refined_messages` in context

2. **Verify cache insertion path:**
   - ✅ `insert_cache()` called in `_generate_streamed_response()` 
   - ✅ Full generated response passed as single segment currently
   - ✅ Auto-segmentation should happen at this point

3. **Verify current `max_bytes` behavior:**
   - Check default value in constructor
   - Confirm how it's configured in practice
   - Verify eviction logic when limit is reached

---

## Key Questions to Answer Before Implementation:

1. **Current segment handling:** 
   - ✅ Prefill: segments created for non-trimmable caches only
   - ✅ Generation: no segments, full response as single segment

2. **Timing of auto-segmentation:**
   - ✅ Should happen in `_generate_streamed_response()` where cache insertion occurs

3. **Segment cache handling:**
   - ✅ Each segment uses same cache object
   - ✅ Segment types track the role (system, assistant, user)

4. **Backward compatibility verification:**
   - ✅ When `prompt_cache_auto_segment=False`, existing behavior preserved
   - ✅ Existing code continues to work without modification

---

## Implementation Notes

1. **Tokenization:** The `tokenize()` function must be imported from the same module where it's used in the current codebase.

2. **Message tracking:** Messages are available as `refined_messages` in `_build_inference_context()` and passed to the context.

3. **Performance:** Auto-segmentation adds O(n) complexity where n is the number of role changes. For most conversations this is negligible (< 10 segments).

4. **Memory:** Each segment creates a separate cache entry, increasing metadata overhead by 10-20% in tool-heavy workloads.

5. **Cache compatibility:** Version bump to 2 means old caches will be automatically cleaned up on restart. Users may need to regenerate caches after upgrade.

6. **Critical dependency:** `refined_messages` must be added to `_InferenceContext` dataclass to pass from `_build_inference_context()` to `_generate_streamed_response()`.
