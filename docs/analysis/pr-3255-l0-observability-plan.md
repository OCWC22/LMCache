# PR #3255 L0 Observability Plan

## Context / Scope

PR #3255 (`pr-3255`) is titled **[MP][Observability] Export L0 block allocation metrics**. The immediate review question from Kuntai is:

> Does this capture the right L0 lifecycle signal for CB use case?

This plan is scoped to making the PR answer that question and the closely related office-hours requirements without turning this PR into a broader observability expansion.

Known status:

- Focused local suite previously passed: `python -m pytest -q tests/v1/mp_observability` → `344 passed, 1 skipped`.
- Visible CI checks pass.
- Modal runtime verification is blocked locally by missing auth: `modal token info` → token not found.
- External docs search found current LMCache MP pages: `https://docs.lmcache.ai/mp/architecture.html` and `https://docs.lmcache.ai/mp/observability.html`.
- Local user-facing docs state L0/L1 throughput timestamps should reflect true GPU-stream copy time, not Python/lock overhead.

## Short Answer to Kuntai

**Yes, if we distinguish two different L0 signals and fix one CacheBlend timing semantic before responding.**

- `lmcache_mp.l0_block_*` metrics from `L0LifecycleSubscriber` measure **vLLM-owned physical GPU block lifecycle**: allocation, cache-hit reuse gaps, eviction, and end-session release. These are sourced from `MP_VLLM_BLOCK_ALLOCATION` and `MP_VLLM_END_SESSION`.
- CacheBlend’s relevant L0 signal is **not** that vLLM block lifecycle metric. It is the CacheBlend GPU-copy lifecycle metric family in `BlendMetricsSubscriber`: `lmcache_blend.l0_gpu_operation_duration_seconds`, `lmcache_blend.l0_gpu_transfer_chunks`, and `lmcache_blend.l0_gpu_transfer_tokens`, sourced from CB GPU `START`/`END` events.
- The caveat: `CB_STORE_PRE_COMPUTED_START` currently has inconsistent timing versus `CB_STORE_FINAL_START`; it can include vLLM IPC wait differently. That should be fixed or explicitly documented before claiming the CB L0 signal matches the “true GPU-stream copy time” contract.

Suggested PR comment response after the must-fix lands:

> We now separate the two L0 meanings. The `lmcache_mp.l0_block_*` series is the vLLM physical block lifecycle signal. For CacheBlend, the lifecycle signal is the `lmcache_blend.l0_gpu_*` series from stream-bound CB store/retrieve/final START→END events. I also aligned `CB_STORE_PRE_COMPUTED_START` with `CB_STORE_FINAL_START` so the CB duration excludes pre-copy vLLM IPC wait and represents GPU-stream copy progress consistently.

## Priority Matrix

| Priority | Item | Files / Functions | Why |
|---|---|---|---|
| **Must-fix before responding to Kuntai** | Align `CB_STORE_PRE_COMPUTED_START` timing with `CB_STORE_FINAL_START` | `lmcache/v1/multiprocess/blend_server_v2.py`: `cb_store_pre_computed()`, `_cb_store_gpu_copy()`, `cb_store_final()` | Directly affects whether CB L0 GPU duration is the right lifecycle signal. |
| **Must-fix before responding to Kuntai** | Add/adjust focused test coverage for the timing contract | `tests/v1/mp_observability/subscribers/metrics/test_cb_server.py` | Protects the metric semantics from regression. |
| **Must-fix before responding to Kuntai** | Document the two L0 meanings and CB metric family | `docs/design/v1/mp_observability/METRICS.md`, `EVENTS.md`, `README.md`; user-facing `docs/source/mp/observability.rst` | Makes the PR self-explanatory and answerable. |
| **Should-fix in this PR** | Tighten CB metadata wording (`num_tokens`, `stored_chunks`, operation labels) | `EVENTS.md`, `METRICS.md`, maybe `test_cb_server.py` | Improves operator interpretation without widening subsystem scope. |
| **Follow-up PR** | API1 serde encode/decode metrics | `lmcache/v1/distributed/serde/async_processor.py`, `lmcache/v1/distributed/l2_adapters/serde_wrapper.py` | Larger distributed/serde observability expansion. |
| **Follow-up PR** | API2 token-range/model-addressed KV API observability | `lmcache/v1/internal_api_server/vllm/lookup_api.py`, `lmcache/v1/lookup_client/`, `lmcache/v1/multiprocess/blend_server_v2.py`, `lmcache/v1/mp_observability/subscribers/metrics/lookup.py` | API-level lifecycle and validation scope, not necessary for CB L0 timing. |
| **Follow-up PR** | Queue wait / thread-pool saturation metrics | `lmcache/v1/multiprocess/mq.py`, `lmcache/v1/multiprocess/affinity_pool.py`, `lmcache/v1/multiprocess/config.py` | Requires MQ/pool instrumentation beyond this PR. |
| **Follow-up validation** | Direct GPU runtime proof artifact | Modal/GPU run using `publish_on_stream()` paths and boundary evidence | Blocked locally by missing Modal token. |

## Must-Fix Details

### 1. Align `cb_store_pre_computed` timing semantics

Verified local behavior:

- `lmcache/v1/multiprocess/blend_server_v2.py` `_cb_store_gpu_copy(...)` waits on the vLLM IPC event before publishing the optional `start_event`.
- `cb_store_pre_computed(...)` currently publishes `CB_STORE_PRE_COMPUTED_START` before calling `_cb_store_gpu_copy(...)`, so its measured duration can include time queued before the vLLM event wait completes.
- `cb_store_final(...)` passes `CB_STORE_FINAL_START` into `_cb_store_gpu_copy(..., start_event=...)`, so START is published after the vLLM IPC wait is enqueued on the stream.

Minimal code recommendation:

- Remove the direct `publish_on_stream(CB_STORE_PRE_COMPUTED_START, ...)` block from `cb_store_pre_computed()`.
- Pass the same event as `start_event=` into `_cb_store_gpu_copy(...)`, matching `cb_store_final()`.
- Keep `CB_STORE_PRE_COMPUTED_SUBMITTED` as the CPU-synchronous span/request guard. Do not use submitted events for GPU duration metrics.

Expected effect:

- `lmcache_blend.l0_gpu_operation_duration_seconds{operation="store_pre_computed"}` shifts to the same semantic boundary as `store_final`: GPU-stream copy progress, not upstream vLLM wait.
- This is a metric semantics correction, not a protocol or persistence change.

### 2. Tests to update

Primary file:

- `tests/v1/mp_observability/subscribers/metrics/test_cb_server.py`

Recommended assertions:

- Existing duration tests should continue to prove `BlendMetricsSubscriber` computes duration from CB START→END timestamps.
- Add/adjust a small test around event ordering if feasible: construct events where `SUBMITTED` precedes START and ensure duration uses START→END only.
- If tests inspect emitted events from `cb_store_pre_computed`, assert the START event is supplied through `_cb_store_gpu_copy(start_event=...)` or use a lightweight monkeypatch/fake event bus to verify ordering relative to the helper.

Do not expand into full runtime integration unless Modal/GPU credentials are available.

### 3. Docs to update

#### `docs/design/v1/mp_observability/METRICS.md`

Add near **L0 (GPU) Block Lifecycle Histograms**:

- `lmcache_mp.l0_block_*` = vLLM physical GPU block lifecycle.
- These metrics do not measure CacheBlend GPU-copy duration.
- CacheBlend GPU-copy lifecycle lives under `lmcache_blend.l0_gpu_*`.

Add to the CacheBlend metrics section:

| Metric | Prometheus name | Type | Source events | Meaning |
|---|---|---|---|---|
| `lmcache_blend.l0_gpu_operation_duration_seconds` | `lmcache_blend_l0_gpu_operation_duration_seconds` | Histogram | `CB_*_START` → `CB_*_END` | Stream-bound CB GPU operation duration. |
| `lmcache_blend.l0_gpu_transfer_chunks` | `lmcache_blend_l0_gpu_transfer_chunks` | Counter/Histogram as implemented | `CB_*_END` metadata | CB GPU chunks transferred by operation. |
| `lmcache_blend.l0_gpu_transfer_tokens` | `lmcache_blend_l0_gpu_transfer_tokens` | Counter/Histogram as implemented | `CB_*_END` metadata | CB GPU tokens transferred by operation. |

Use the exact instrument types from `lmcache/v1/mp_observability/subscribers/metrics/cb_server.py` when implementing.

#### `docs/design/v1/mp_observability/EVENTS.md`

Under Blend Server events:

- State `CB_*_SUBMITTED` events are CPU-synchronous request/span sentinels.
- State `CB_*_START/END` events are stream-bound GPU lifecycle events used for CB L0 GPU metrics.
- Define `CB_STORE_PRE_COMPUTED_START` as emitted after the vLLM IPC wait is enqueued, matching `CB_STORE_FINAL_START`, if the code fix lands.

#### `docs/design/v1/mp_observability/README.md`

Add a short architecture note:

- Standard MP/vLLM L0 block lifecycle uses `lmcache_mp.l0_block_*`.
- CacheBlend L0 GPU-copy lifecycle uses `lmcache_blend.l0_gpu_*`.

#### `docs/source/mp/observability.rst`

User-facing docs gaps:

- The intro currently implies all metrics use `lmcache_mp.`. Update it to mention CacheBlend uses `lmcache_blend.`.
- Add a **CacheBlend Metrics** subsection near the metrics section, ideally after L0/L1 throughput or before L1/L2 throughput.
- Include the CB L0 GPU metric table and the same distinction from vLLM block lifecycle.

## Should-Fix In This PR

These are worthwhile if quick, but should not delay the narrow Kuntai response once the must-fix items are done:

1. Tighten wording around CB metadata fields: `num_tokens`, `stored_chunks`, `num_chunks`, `success`, and operation labels.
2. Ensure design docs and user docs use the same metric names and Prometheus names.
3. Mention opt-in redacted boundary evidence only as supporting proof, not the primary metric.

## Follow-Up PRs / Office-Hours Backlog

### API1 serde encode/decode metrics

Likely files:

- `lmcache/v1/distributed/serde/async_processor.py`
- `lmcache/v1/distributed/l2_adapters/serde_wrapper.py`
- `docs/design/v1/distributed/serde/README.md`
- `docs/design/v1/distributed/l2_adapters/serde_wrapper.md`

Suggested future metrics:

- serialize/deserialize latency
- failures by direction/backend/model if available
- queue depth / in-flight tasks
- bytes or object counts if cheaply available

Reason to defer: crosses distributed L2 adapter + serde async processor and needs a new event/metric contract.

### API2 token-range/model-addressed KV API observability

Likely files:

- `lmcache/v1/internal_api_server/vllm/lookup_api.py`
- `lmcache/v1/lookup_client/`
- `lmcache/v1/multiprocess/blend_server_v2.py`
- `lmcache/v1/mp_observability/subscribers/metrics/lookup.py`
- `lmcache/v1/distributed/api.py`
- `lmcache/v1/multiprocess/custom_types.py`

Suggested future metrics:

- request lifecycle counters/latencies by model and token-range outcome
- validation failures
- hit/miss/stale decisions for token-range requests

Reason to defer: broader API observability, not needed for CB L0 GPU lifecycle correctness.

### Queue wait / thread-pool saturation

Likely files:

- `lmcache/v1/multiprocess/mq.py`
- `lmcache/v1/multiprocess/affinity_pool.py`
- `lmcache/v1/multiprocess/config.py`
- maybe `lmcache/v1/mp_observability/subscribers/metrics/event_bus.py` or a new MQ metrics subscriber

Suggested future metrics:

- request queue wait seconds
- executor queue length
- active CPU/GPU workers
- affinity-worker queued tasks
- saturation ratio vs `max_gpu_workers` / `max_cpu_workers`

Reason to defer: requires MQ/pool instrumentation and probably public accessors on `AffinityThreadPool`.

### Direct GPU runtime proof

Code-level proof exists through:

- `lmcache/v1/mp_observability/event_bus.py`: `publish_on_stream()`
- `lmcache/v1/mp_observability/l0_boundary_evidence.py`
- CB GPU event emission in `lmcache/v1/multiprocess/blend_server_v2.py`
- standard MP GPU event emission in `lmcache/v1/multiprocess/server.py`

Runtime artifact remains blocked until Modal auth is available.

## Verification Commands

Focused after code/doc changes:

```bash
python -m pytest -q tests/v1/mp_observability/subscribers/metrics/test_cb_server.py
python -m pytest -q tests/v1/mp_observability/subscribers/metrics/test_l0_l1_throughput.py
python -m pytest -q tests/v1/mp_observability/subscribers/metrics/test_l0_lifecycle.py
python -m pytest -q tests/v1/mp_observability
```

Full standard suite if desired before final PR update:

```bash
python -m pytest -xvs --ignore=tests/disagg \
  --ignore=tests/v1/test_nixl_storage.py \
  --ignore=tests/v1/multiprocess/ \
  --ignore=tests/v1/distributed/ \
  --ignore=tests/skipped \
  --ignore=tests/v1/storage_backend/test_eic.py
```

Docs verification after user-facing docs changes:

```bash
cd docs
make clean
make html
```

Runtime verification currently blocked:

```bash
modal token info
# expected currently: Token not found
```

Once Modal auth is fixed, run the project’s Modal/GPU verification flow for the CB metrics path and capture:

- Prometheus samples for `lmcache_blend_l0_gpu_operation_duration_seconds`
- operation labels for `store_pre_computed`, `retrieve`, and `store_final`
- optional redacted boundary evidence JSONL showing START/END stream-bound events

## Implementation Work Items For Next Agent

1. **CB timing fix and tests**
   - Modify `blend_server_v2.py` only for `cb_store_pre_computed` START event placement.
   - Update `test_cb_server.py` or a targeted fake-event test to guard the semantics.

2. **Docs update**
   - Update design docs: `METRICS.md`, `EVENTS.md`, `README.md`.
   - Update user docs: `docs/source/mp/observability.rst`.

3. **Verification and PR comment**
   - Run focused pytest commands.
   - Run Sphinx docs build if docs are changed.
   - Post response to Kuntai using the short answer above, adjusted to match final code/tests.
