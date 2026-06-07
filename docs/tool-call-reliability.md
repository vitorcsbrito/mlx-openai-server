# Tool-Call Reliability

This document describes two related reliability features that make multi-turn,
tool-using agents far more dependable on local MLX models:

1. **Tool-call self-healing** — recovers well-formed tool calls that a model
   emits *inside* its reasoning span instead of after it.
2. **OOM recovery on checkpoint insert** — keeps a transient out-of-memory
   error during prompt-cache checkpointing from cascading into a crash.

They are causally linked: self-healing increases the volume of successful tool
calls, which increases multi-turn traffic and prompt-cache pressure, which is
what surfaced the latent OOM. Documenting them together reflects how they were
discovered and why both matter.

If you change the code referenced below, update this doc.

---

## 1. Tool-call self-healing

### The problem

Non fine-tuned (or weakly tool-tuned) models sometimes "think out loud" and emit
a complete tool-call block *inside* their reasoning span:

```text
<think>
I should read the file to answer this.
<tool_call>{"name":"read_file","arguments":{"path":"app/handler/mlx_lm.py"}}</tool_call>
</think>
```

In the non-streaming path, parsing happens in a strict order
(`app/handler/mlx_lm.py`, the separate-parsers branch):

1. The reasoning parser
   ([`HermesReasoningParser.extract_reasoning`](../app/parsers/hermes.py),
   `hermes.py:35`) pulls **everything** between `<think>` and `</think>` into
   `reasoning_content`, and hands the tool parser only the text *after*
   `</think>`.
2. The tool parser
   ([`HermesToolParser.extract_tool_calls`](../app/parsers/hermes.py),
   `hermes.py:149`) therefore never sees a `<tool_call>` that lived inside the
   reasoning span — the call is silently dropped.

For an agent, a dropped tool call is a dead end: no tool executes, the
conversation stalls, and the user has to nudge the model to continue every turn.

### The fix

A fallback recovery pass runs after the normal parse:
[`_recover_misplaced_tool_calls`](../app/handler/mlx_lm.py) (`mlx_lm.py:89`),
invoked at `mlx_lm.py:1575`.

It only acts when **all** of the following hold, so correctly-behaving models are
never affected:

- a tool parser is configured, **and**
- no tool calls were recovered from the post-`</think>` content, **and**
- `reasoning_content` is a string that contains the tool-open marker.

When it fires, it:

1. Re-runs `extract_tool_calls` against `reasoning_content`.
2. Promotes any recovered, **well-formed** calls to `parsed_response["tool_calls"]`.
3. Strips the recovered blocks out of the reasoning text via
   [`_strip_complete_tool_blocks`](../app/handler/mlx_lm.py) (`mlx_lm.py:49`), so
   the JSON does not leak into the visible reasoning.

### Scope and boundaries

| Aspect | Behavior |
|--------|----------|
| Path | Non-streaming only. Streaming already handles this via its tool-before-reasoning parse + `requeue_reasoning_tail` logic. |
| Parsers | Separate reasoning + tool parsers. Unified parsers own their own split and are out of scope. |
| Recovery | **Well-formed** `<tool_call>…</tool_call>` blocks only. Malformed JSON and bare fenced blocks are intentionally *not* recovered (higher false-positive risk). |
| Safety | Pure post-generation string work — allocates no MLX/Metal memory and runs only as a fallback. |

### Observed impact

On `darwin-9b` — a model that previously needed a nudge nearly every turn to keep
going — self-healing enabled **8 consecutive tool calls at ~31k context** with no
manual intervention.

---

## 2. OOM recovery on checkpoint insert

### The problem

With auto-segmented prompt caching, the batched scheduler saves a KV-cache
checkpoint at each role boundary. For a tool-using request that includes a
`tool` segment, after generation it calls `extract_cache` and then
`insert_cache(..., cache_type="tool")`
([`BatchScheduler._handle_prompt_responses`](../app/core/batch_scheduler.py),
`batch_scheduler.py:739`).

`insert_cache` materializes the full KV cache (to compute `nbytes` and serialize
to disk), which forces a GPU→host evaluation and spikes Metal memory. Under load
this can raise an OOM **inside** the insert.

Previously, the best-effort `except` logged the failure and continued **without
reclaiming anything**:

```text
prompt_cache.insert_cache (tool) failed for uid=…: [METAL] Insufficient memory
```

The extracted caches stayed referenced and the allocator stayed exhausted, so
the next request — and any retry — OOMed immediately. The result was a hard
crash cascade.

> Self-healing made this far more visible: by reviving tool-calling loops that
> used to dead-end, it greatly increased the number of `tool`-segment checkpoint
> inserts, and therefore the frequency of hitting the memory ceiling at exactly
> that point.

### The fix

In the failure handler (`batch_scheduler.py:792`), the scheduler now drops the
extracted-cache references and trims MLX buffers before bailing out
(`batch_scheduler.py:801-803`):

```python
cache = None
caches.clear()
try:
    mx.clear_cache()
except Exception as clear_exc:  # best-effort reclaim
    logger.warning(f"mx.clear_cache after failed insert failed: {clear_exc!s}")
return
```

This lets the process recover from a transient OOM instead of cascading into a
crash with failing retries. The save is still best-effort — a missed checkpoint
only costs prefix reuse on a later request, never correctness.

---

## Defense in depth

These two code changes work together with one configuration lever to address the
OOM from both ends:

| Layer | Mechanism | Role |
|-------|-----------|------|
| Prevention | `batch_prefill_step_size = 2048` (lower than 4096) | Smaller transient prefill activations → lower peak memory at the moment checkpointing runs. Tune upward only with verified headroom. |
| Containment | `batch_scheduler.py` reclaim-on-failure | A transient checkpoint OOM no longer crashes the server. |
| Capability | Tool-call self-healing | Keeps agents progressing across turns without manual nudging. |

`batch_prefill_step_size` controls how many prompt tokens are processed per
prefill pass; at 4096 the prefill peak roughly doubles versus 2048 and coincides
with the checkpoint materialization peak. See
[`docs/configuration.md`](./configuration.md) for the full configuration
reference.

---

## Tests

- `tests/test_mixed_think_tool_handoff_stream_handler_integration.py`
  - `test_nonstream_recovers_tool_call_emitted_inside_reasoning_block` — a tool
    call stranded inside `<think>` is recovered and stripped from reasoning.
  - `test_nonstream_does_not_recover_when_post_reasoning_tool_call_exists` — the
    negative case: self-healing stays out of the way when the normal path
    succeeds.
- `tests/test_batch_scheduler.py`
  - `test_failed_checkpoint_insert_reclaims_mlx_memory` — a failing
    `insert_cache` triggers `mx.clear_cache()` and does not propagate.
