# On-Demand Models (Lazy Load + Idle Unload)

On-demand models are **loaded the first time they are requested** and
**unloaded after a period of inactivity**, so a multi-model deployment can
advertise more models than fit in memory at once and only pay the RAM cost for
the ones actively in use.

On-demand is **multi-handler (`--config`) only** — there is no equivalent in
single-model mode. See [configuration.md](./configuration.md) for where the
keys live.

```yaml
models:
  - model_path: black-forest-labs/FLUX.2-klein-4B
    model_type: image-generation
    served_model_name: flux2-klein-4b
    on_demand: true                # lazy load + idle unload
    on_demand_idle_timeout: 120    # seconds idle before unloading
    queue_timeout: 600             # per-request wall-clock timeout
```

| Key | Default | Meaning |
|-----|---------|---------|
| `on_demand` | `false` | Enable lazy load + idle unload for this entry. |
| `on_demand_idle_timeout` | `60` | Seconds the model may sit **idle** before it is unloaded. |
| `queue_timeout` | `300` | Unrelated knob: the wall-clock timeout for a single request. |

> **`on_demand_idle_timeout` and `queue_timeout` are independent.** One bounds
> how long an *idle* model stays resident; the other bounds how long a *running*
> request may take. The idle timer does not run while a request is in flight
> (see below), so a short idle timeout no longer unloads a model out from under
> a long request.

---

## When does the idle timer start?

**The idle timer starts when a request finishes — never while one is running.**

Internally each on-demand model carries a **reference count**. A request
*acquires* a reference when it starts and *releases* it when it completes. The
idle timer (`on_demand_idle_timeout`) is scheduled **only at the moment the
reference count drops back to 0**, i.e. when the *last* in-flight request for
that model finishes. While any request holds a reference the timer is not
running at all.

- **Acquire** — on request entry, `_resolve_handler` calls
  `ensure_on_demand_loaded`, which bumps the ref count and **cancels any pending
  idle-unload timer** (`app/core/model_registry.py:360`). If the model is not
  resident, it is loaded here first.
- **Release** — when the request completes, `release_on_demand` decrements the
  ref count; reaching **0** schedules `_idle_unload(timeout)`
  (`app/core/model_registry.py:432`).
- **Unload** — `_idle_unload` sleeps for `on_demand_idle_timeout`, then
  re-checks under a lock that the ref count is still 0 before calling
  `handler.cleanup()` and dropping the handler
  (`app/core/model_registry.py:447`).

### When "release" happens depends on the response type

| Response type | Release point | Idle timer starts… |
|---------------|---------------|--------------------|
| Non-streaming | endpoint `finally`, after the generation `await` returns (`app/api/endpoints.py:644`) | when generation is fully complete |
| Streaming | deferred: the wrapped response body releases when the stream is exhausted, closed, or errors (`_attach_on_demand_release`, `app/api/endpoints.py`) | when the last chunk has been sent |

A `StreamingResponse` is returned from the endpoint *before* any token is
generated. The release is therefore deferred until the body iterator is fully
consumed; otherwise the timer would start mid-generation and could unload the
model while it is still streaming. See
[the streaming-release section below](#streaming-vs-non-streaming) for detail.

---

## Single-request timeline

```
request in ──▶ acquire (ref 0→1, load if needed, cancel any timer)
                  │
                  ▼
            generation runs        ← idle timer NOT running (ref == 1)
                  │
                  ▼
request done ─▶ release (ref 1→0) ─▶ start idle timer (on_demand_idle_timeout)
                                          │
                       ┌──────────────────┴───────────────────┐
              new request arrives                       timer expires
              before timeout                            (still idle)
                  │                                           │
            cancel timer,                               unload model
            reuse model                                 (next request reloads)
```

The clock starts at the **end** of the request — the last token for a
non-streaming response, or the last streamed chunk for a streaming response.
There is no heartbeat that resets the timer mid-request; it doesn't need one,
because the held reference keeps the timer from running in the first place.

---

## Concurrent / batched scenario

The reference count lives at the **HTTP request boundary**, not inside the
continuous-batching scheduler. "Batched" here means **N overlapping requests**
against the same on-demand model. Each request independently acquires on entry
and releases on its own completion.

```
A in ─▶ acquire (ref 0→1, load, cancel timer)
B in ─▶ acquire (ref 1→2)          ← model already resident, just bump
A done ─▶ release (ref 2→1)        ← no timer: only ref==0 schedules one
B done ─▶ release (ref 1→0) ─▶ start idle timer
```

**The idle timer starts only when the last in-flight request releases.** As
long as requests keep overlapping, the ref count never reaches 0 and the timer
never starts. Sequential (non-overlapping) requests each start a timer on
completion; if the next request lands within `on_demand_idle_timeout` it
cancels the pending unload, otherwise the model unloads and the next request
pays a reload.

---

## Streaming vs non-streaming

For a streaming request the model must stay resident for the whole stream, even
though the endpoint coroutine returns the `StreamingResponse` immediately:

- `_attach_on_demand_release` wraps the response's body iterator so
  `release_on_demand` runs in the iterator's `finally` — after the last chunk,
  on client disconnect, or on error.
- `_release_on_demand` (called from the endpoint's own `finally`) becomes a
  no-op once the release has been deferred, so the reference is not dropped
  twice or dropped early.

This is wired into the streaming endpoints: `/v1/chat/completions`,
`/v1/audio/transcriptions`, and `/v1/responses`. Non-streaming responses
release immediately in the endpoint `finally`, which is correct because the
`await` has already produced the full result.

---

## Only one on-demand model resident at a time

Loading an on-demand model evicts **idle** on-demand models to free memory
(`ensure_on_demand_loaded`, `app/core/model_registry.py:373`). An on-demand
model that still has in-flight requests (ref count > 0) is **not** evicted — it
is kept loaded alongside the newly requested one until its own requests drain.
Always-on (non-`on_demand`) models are never evicted by this path.

Loads are serialized by a lock, so concurrent first-hits for the same model
load it once and then share the single resident handler.

---

## Interaction with the persistent prompt cache

Idle unload routes through `handler.cleanup()`. When a model is configured with
a persistent `prompt_cache_dir`, that cleanup **preserves** the on-disk
payloads (only the in-memory index is dropped), so a model that is unloaded by
the idle timer and later reloaded rehydrates its prompt KV cache from disk
rather than starting cold. See the prompt-cache persistence notes for the
ownership rules. An auto-created temp cache directory is still removed on
unload.

---

## Source map

| Behavior | Location |
|----------|----------|
| Acquire + cancel timer + load | `app/core/model_registry.py:333` (`ensure_on_demand_loaded`) |
| Evict idle peers / keep in-flight | `app/core/model_registry.py:373` |
| Release + schedule idle timer | `app/core/model_registry.py:418` (`release_on_demand`) |
| Idle sleep + re-check + unload | `app/core/model_registry.py:447` (`_idle_unload`) |
| Request acquire | `app/api/endpoints.py:146` (`_resolve_handler`) |
| Deferred streaming release | `app/api/endpoints.py:185` (`_release_on_demand`, `_attach_on_demand_release`) |
