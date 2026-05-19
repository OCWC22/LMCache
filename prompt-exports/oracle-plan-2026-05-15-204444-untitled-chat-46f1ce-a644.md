# Oracle Plan

## 1) Summary

This plan splits work into: **(a) PR #3255 pre-merge fixes (P0/P1)** to ensure correctness and merge readiness, **(b) a focused **vLLM v0.21.0 compatibility validation stream** (mostly P1, some P0 if regressions found), and **(c) medium-term LMCache roadmap items (P2)** informed by v0.21.0 changes. The approach is a **targeted change set** for PR #3255 (no refactor), then a **separate compatibility PR/test matrix** to isolate risk and keep merge velocity.

---

## 2) Current-state analysis

Assumptions to validate during implementation (since code snapshot is not included here):
- `server.py` currently imports `os` but does not use it.
- `EVENTS.md` documents a `CB submitted` event but does not include `num_chunks` and `num_tokens` in metadata schema.
- `cb_server.py` has a pending-ops map keyed in a way that may collide under concurrent/multi-request conditions (P1 review finding).
- Existing event emission and pending-op lifecycle are already wired into request/session lifecycle and tests.

Likely relevant flow (to validate):
1. Request/session enters callback server (`cb_server.py`).
2. Pending op state is stored in a dict/map keyed by an identifier.
3. Event emission path publishes `CB submitted` metadata.
4. Session teardown / scheduler pause transitions trigger `END_SESSION`.
5. Worker/engine connectors (Ray, NIXL, offloading connectors) handle IPC + KV/block metadata.

Reusable assets expected:
- Existing event schema docs + event emission helper.
- Existing callback pending-op registry logic.
- Existing integration tests for session lifecycle and worker execution.

Blocking risks today:
- Pending-op key collision can corrupt state or misroute completions.
- Event docs drift from runtime payloads if not aligned now.

---

## 3) Design

## A. PR #3255 remaining fixes (pre-merge)

### Priority
- **P0**: `server.py` unused import removal; `EVENTS.md` metadata update.
- **P1**: pending-ops key collision hardening in `cb_server.py`.

### A1) Remove unused `import os` in `server.py` (P0)
- **Change type**: targeted hygiene fix.
- **Design**: remove only the unused import; no behavior change.
- **Validation**:
  - Lint/type checks pass.
  - No runtime path references removed symbol.

### A2) Update `EVENTS.md` for `CB submitted` metadata (P0)
- **Change type**: documentation contract alignment.
- **Required metadata additions**:
  - `num_chunks` (integer; default behavior when unknown must be documented explicitly, e.g. `0` or omitted).
  - `num_tokens` (integer; same default/optional semantics documented).
- **Design decision**: mirror actual emitted payload, not aspirational schema.
- **Validation**:
  - Confirm emitter currently sends these fields; if not, either add emitter in same PR (atomic) or document as conditional with version note.
  - Ensure event examples in doc include both fields.

### A3) Pending-ops key collision mitigation in `cb_server.py` (P1)
- **Problem**: non-unique key (e.g., request id/session id alone) may collide across retries, multi-phase ops, or concurrent chunks.
- **Recommended fix**: switch to **composite deterministic key** with collision-safe components, e.g.:
  - `pending_key = (session_id, request_id, op_type, sequence_no)`  
  or serialized equivalent if dict keys must be strings.
- **State/lifecycle rules**:
  - Key creation occurs at op enqueue.
  - Key removal occurs exactly once on terminal callback.
  - Duplicate completion must be idempotent (no crash; log + ignore).
- **Concurrency expectations**:
  - If map is shared across async tasks/threads, enforce existing lock/actor discipline around insert/remove/lookup.
- **Failure behavior**:
  - On missing key at completion: warn + drop.
  - On duplicate insert: reject with error metric/log and regenerate unique sequence if applicable.
- **Validation**:
  - Unit test: two ops with same prior key material no longer collide.
  - Integration test: concurrent submits/completions do not overwrite each other.

---

## B. vLLM v0.21.0 compatibility plan (post-merge or separate PR)

### Priority
- **P1** validation stream overall.
- Escalate any production-breaking finding to **P0** hotfix.

### B1) HMA compatibility: block allocation delta reporting
- **Goal**: verify delta reporting semantics still correct under HMA block placement.
- **Test design**:
  - Capture per-step allocation/free deltas from current tracker.
  - Compare against ground truth from vLLM 0.21.0 APIs/telemetry under HMA enabled.
- **Pass criteria**:
  - No negative/phantom deltas.
  - Session totals reconcile with final allocator state.
- **Unknown to validate**:
  - Whether allocator callback field names or timing changed in 0.21.0.

### B2) Two-phase scheduler pause and `END_SESSION`
- **Goal**: ensure `END_SESSION` still fires exactly once with new pause semantics.
- **Test matrix**:
  - pause before submit, mid-generation, and after final token.
  - resume/no-resume paths.
- **Pass criteria**:
  - Exactly-once `END_SESSION`.
  - No premature cleanup while pending callbacks exist.

### B3) Speculative decoding per-step allocation elimination
- **Goal**: ensure block tracking still works when per-step alloc events are reduced/removed.
- **Design**:
  - Fallback to coarser lifecycle signals (request start/end + periodic snapshots) if step-level hooks disappear.
- **Pass criteria**:
  - Tracking remains monotonic and reconciles at request end.

### B4) RayExecutorV2 worker IPC
- **Goal**: verify worker IPC contracts unchanged or adaptors updated.
- **Tests**:
  - Multi-worker request fan-out/fan-in.
  - Callback ordering and timeout handling.
- **Pass criteria**:
  - No deadlocks; stable completion ordering guarantees documented.

### B5) NIXL 1.x connector abstraction
- **Goal**: detect connector interface changes impacting current integration.
- **Checks**:
  - Constructor signature changes.
  - Send/receive API and lifecycle hooks.
- **Outcome**:
  - If compatible: add version gate note.
  - If incompatible: implement thin adapter layer in separate PR.

---

## C. LMCache roadmap items surfaced by v0.21.0

### Priority
- **P2** (design/prototyping unless urgent dependency emerges).

### C1) MooncakeStoreConnector pattern for distributed KV offloading
- Extract reusable connector pattern:
  - capability discovery,
  - async write-back/read-through,
  - failure fallback to local cache.
- Deliverable: short architecture RFC + spike implementation plan.

### C2) DCP/PCP OffloadingConnector support
- Define interface contract first (minimal stable abstraction):
  - `put/get/evict`, batch semantics, timeout/cancellation behavior.
- Build compatibility matrix against existing connectors and failure modes.

### C3) Transformers v5 migration readiness
- Inventory dependency touchpoints:
  - tokenizer/model APIs,
  - generation outputs,
  - config loading.
- Add CI lane (allow-fail initially) for v5 to quantify breakage before migration PR.

---

## 4) File-by-file impact

(Use actual repository paths/names during implementation; names below reflect user-referenced files.)

- **`server.py`**
  - Remove unused `import os`.
  - Why: lint cleanliness and pre-merge gate.
  - Dependency: none.

- **`EVENTS.md`**
  - Update `CB submitted` event metadata schema and examples with `num_chunks`, `num_tokens`.
  - Why: doc/runtime contract sync.
  - Dependency: validate emitter payload shape.

- **`cb_server.py`**
  - Modify pending-op key generation and map access lifecycle to avoid collisions.
  - Add idempotent completion handling/logging for duplicates/missing keys.
  - Why: P1 correctness risk under concurrency/retries.
  - Dependency: tests for concurrent ops should be updated with new key semantics.

- **Test files (unit/integration; exact files to locate)**
  - Add/extend tests for:
    - pending-op collision scenarios,
    - `END_SESSION` under two-phase pause,
    - block tracking under HMA/spec decoding,
    - RayExecutorV2 IPC behavior,
    - NIXL connector compatibility checks.
  - Why: lock in compatibility behavior for 0.21.0 stream.

---

## 5) Risks and migration

- **Backward compatibility risk**: pending-op key format changes may affect logs/metrics dashboards if they parse raw keys.
  - Mitigation: keep external log fields stable, or add explicit migration note.
- **Behavioral risk**: event schema doc may diverge from emitter if not validated atomically.
  - Mitigation: pair doc update with emitter verification in same merge window.
- **Upgrade risk (vLLM 0.21.0)**: allocator/scheduler API timing changes can silently break accounting.
  - Mitigation: reconciliation-based tests (not just hook-presence tests).

---

## 6) Implementation order

1. **P0 pre-merge hygiene**: remove `import os` in `server.py`; update `EVENTS.md`.
2. **P1 pre-merge safety**: implement pending-op key collision mitigation in `cb_server.py` + unit tests.
3. Run full CI/lint; merge PR #3255 once green.
4. Open **separate v0.21.0 compatibility PR** with test matrix scaffolding first.
5. Implement and run B1–B5 validations; patch compatibility issues found.
6. Land compatibility PR with explicit release notes.
7. Start P2 LMCache RFC/spikes (Mooncake pattern, DCP/PCP connector abstraction, Transformers v5 CI lane).