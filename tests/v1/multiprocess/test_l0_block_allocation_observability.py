# SPDX-License-Identifier: Apache-2.0
"""Production SDLC tests for vLLM L0 block allocation observability wiring.

These tests intentionally exercise the wire boundaries around
REPORT_BLOCK_ALLOCATION rather than only subscriber metric math:

* scheduler adapter -> message queue request + redacted boundary evidence
* MP server handler -> EventBus event + redacted boundary evidence
"""

# Standard
from __future__ import annotations

from unittest.mock import MagicMock
import json
import sys
import threading
import types

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
)
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.l0_boundary_evidence import (
    L0_BLOCK_BOUNDARY_EVIDENCE_ENV,
    reset_l0_block_boundary_evidence_path_cache,
)
from lmcache.v1.multiprocess.custom_types import BlockAllocationRecord
from lmcache.v1.multiprocess.protocol import RequestType


def _install_native_storage_ops_stub() -> None:
    if "lmcache.native_storage_ops" in sys.modules:
        return
    native_storage_ops = types.ModuleType("lmcache.native_storage_ops")

    class TTLLock:
        def __init__(self) -> None:
            self.count = 0

        def lock(self) -> None:
            self.count += 1

        def unlock(self) -> None:
            self.count = max(0, self.count - 1)

        def is_locked(self) -> bool:
            return self.count > 0

    class Bitmap:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    native_storage_ops.TTLLock = TTLLock  # type: ignore[attr-defined]
    native_storage_ops.Bitmap = Bitmap  # type: ignore[attr-defined]
    sys.modules["lmcache.native_storage_ops"] = native_storage_ops

    cupy = types.ModuleType("cupy")
    cupy_cuda = types.SimpleNamespace(
        ExternalStream=lambda *args, **kwargs: None,
        Stream=object,
    )
    cupy.cuda = cupy_cuda  # type: ignore[attr-defined]
    sys.modules["cupy"] = cupy


class _RecordingEventBus:
    def __init__(self) -> None:
        self.events = []

    def publish(self, event) -> None:
        self.events.append(event)


def _records() -> list[BlockAllocationRecord]:
    return [
        BlockAllocationRecord(
            req_id="req-prod-1",
            new_block_ids=[7, 8, 9],
            new_token_ids=[101, 102, 103],
        )
    ]


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_scheduler_adapter_reports_block_allocation_with_redacted_boundary_evidence(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "l0-boundary.jsonl"
    monkeypatch.setenv(L0_BLOCK_BOUNDARY_EVIDENCE_ENV, str(evidence_path))
    reset_l0_block_boundary_evidence_path_cache()

    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.model_name = "production-model"
    adapter._health_event = threading.Event()
    adapter._health_event.set()
    adapter.mq_client = MagicMock()
    records = _records()

    try:
        adapter.report_block_allocations(records)
    finally:
        reset_l0_block_boundary_evidence_path_cache()

    adapter.mq_client.submit_request.assert_called_once()
    request_type, payloads, response_class = (
        adapter.mq_client.submit_request.call_args[0]
    )
    assert request_type == RequestType.REPORT_BLOCK_ALLOCATION
    assert payloads[1] == "production-model"
    assert payloads[2] == records
    assert response_class is None

    [event] = _read_jsonl(evidence_path)
    assert event["schema_version"] == "inferguard-l0-block-boundary-event/v1"
    assert event["source"] == "lmcache_vllm_multi_process_adapter"
    assert event["stage"] == "report_block_allocation_submitted"
    assert event["records"] == [{"request_id": "req-prod-1", "block_count": 3}]
    serialized = json.dumps(event)
    assert "new_token_ids" not in serialized
    assert "new_block_ids" not in serialized
    assert "block_ids" not in serialized


def test_mp_server_publishes_report_block_allocation_event_with_redacted_evidence(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "l0-boundary.jsonl"
    monkeypatch.setenv(L0_BLOCK_BOUNDARY_EVIDENCE_ENV, str(evidence_path))
    reset_l0_block_boundary_evidence_path_cache()

    _install_native_storage_ops_stub()
    # First Party
    from lmcache.v1.multiprocess.server import MPCacheEngine

    engine = MPCacheEngine.__new__(MPCacheEngine)
    recording_bus = _RecordingEventBus()
    engine._event_bus = recording_bus  # type: ignore[assignment]
    records = _records()

    try:
        engine.report_block_allocations(
            instance_id=42,
            model_name="production-model",
            records=records,
        )
    finally:
        reset_l0_block_boundary_evidence_path_cache()

    [published] = recording_bus.events
    assert published.event_type == EventType.MP_VLLM_BLOCK_ALLOCATION
    assert published.metadata == {
        "instance_id": 42,
        "model_name": "production-model",
        "records": records,
    }

    [event] = _read_jsonl(evidence_path)
    assert event["schema_version"] == "inferguard-l0-block-boundary-event/v1"
    assert event["source"] == "lmcache_mp_server"
    assert event["stage"] == "report_block_allocation_received"
    assert event["records"] == [{"request_id": "req-prod-1", "block_count": 3}]
    serialized = json.dumps(event)
    assert "new_token_ids" not in serialized
    assert "new_block_ids" not in serialized
    assert "block_ids" not in serialized
