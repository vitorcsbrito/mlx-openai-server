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

Modify `_generate_streamed_response()` around line 339 where `insert_cache()` is called:

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
                segments, segment_types = self.auto_segment_messages(self.messages)
                
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

**Note:** Need to pass `self.messages` to the function where roles are available.

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
   - Locate where segments are created in current code
   - Trace segment flow from creation to cache insertion
   - Confirm where the full generated response is available

2. **Verify cache insertion path:**
   - Find where `insert_cache()` is called
   - Check if segments are already present or need to be computed
   - Confirm the timing of when segments should be auto-calculated

3. **Verify current `max_bytes` behavior:**
   - Check default value in constructor
   - Confirm how it's configured in practice
   - Verify eviction logic when limit is reached

---

## Key Questions to Answer Before Implementation:

1. **Current segment handling:** 
   - Where exactly in the code is the full generated response available when inserting cache?
   - Is the segment creation already happening somewhere, or is it missing?

2. **Timing of auto-segmentation:**
   - Should it happen in the same function where cache insertion occurs?
   - Or in a separate preprocessing step?

3. **Segment cache handling:**
   - Will each segment use the same cache object, or separate cache objects?
   - How do we track which segment produced which cache?

4. **Backward compatibility verification:**
   - What happens when `prompt_cache_auto_segment=False`?
   - Does existing code continue to work without modification?

---

## Implementation Notes

1. **Tokenization:** The `tokenize()` function must be imported from the same module where it's used in the current codebase.

2. **Message tracking:** The `messages` parameter must be available in the scope where `insert_cache()` is called. May need to pass it as an additional parameter to the generation function.

3. **Performance:** Auto-segmentation adds O(n) complexity where n is the number of role changes. For most conversations this is negligible (< 10 segments).

4. **Memory:** Each segment creates a separate cache entry, increasing metadata overhead by 10-20% in tool-heavy workloads.

5. **Cache compatibility:** Version bump to 2 means old caches will be automatically cleaned up on restart. Users may need to regenerate caches after upgrade.
