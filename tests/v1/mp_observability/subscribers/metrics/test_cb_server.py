# SPDX-License-Identifier: Apache-2.0

"""Tests for BlendMetricsSubscriber."""

# Standard
import json
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.l0_boundary_evidence import (
    L0_BLOCK_BOUNDARY_EVIDENCE_ENV,
    reset_l0_block_boundary_evidence_path_cache,
)
from lmcache.v1.mp_observability.subscribers.metrics.cb_server import (
    BlendMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import (
    counter_delta,
    read_counters,
    reader as _reader,
)

_DRAIN_WAIT = 0.15


def _read_histograms() -> dict[str, list]:
    data = _reader.get_metrics_data()
    result: dict[str, list] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if not metric.name.startswith("lmcache_blend.l0_gpu"):
                    continue
                result[metric.name] = list(metric.data.data_points)
    return result


def _histogram_count(name: str) -> int:
    return sum(dp.count for dp in _read_histograms().get(name, []))


def _histogram_attrs(name: str) -> list[dict]:
    return [
        dict(dp.attributes)
        for dp in _read_histograms().get(name, [])
        if getattr(dp, "count", 0) > 0
    ]


def _read_counter_points() -> dict[str, list]:
    data = _reader.get_metrics_data()
    result: dict[str, list] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if not metric.name.startswith("lmcache_blend.l0_gpu"):
                    continue
                points = [dp for dp in metric.data.data_points if hasattr(dp, "value")]
                if points:
                    result[metric.name] = points
    return result


def _counter_sum(name: str, **attrs: object) -> int:
    total = 0
    for dp in _read_counter_points().get(name, []):
        dp_attrs = dict(dp.attributes)
        if all(dp_attrs.get(k) == v for k, v in attrs.items()):
            total += int(dp.value)
    return total


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = BlendMetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    """Capture counters before the test; yield a callable that returns deltas."""
    before = read_counters()

    def get_delta() -> dict[str, int]:
        return counter_delta(before, read_counters())

    return get_delta


class TestBlendMetricsSubscriber:
    def test_subscriptions_cover_all_cb_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_LOOKUP_START in subs
        assert EventType.CB_LOOKUP_END in subs
        assert EventType.CB_RETRIEVE_START in subs
        assert EventType.CB_RETRIEVE_END in subs
        assert EventType.CB_STORE_PRE_COMPUTED_START in subs
        assert EventType.CB_STORE_PRE_COMPUTED_END in subs
        assert EventType.CB_STORE_FINAL_START in subs
        assert EventType.CB_STORE_FINAL_END in subs
        assert EventType.CB_FINGERPRINTS_REGISTERED in subs
        assert EventType.CB_CHUNKS_EVICTED in subs

    def test_subscribes_to_gpu_lifecycle_sentinels_for_l0_evidence(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_REQUEST_START not in subs
        assert EventType.CB_REQUEST_END not in subs
        assert EventType.CB_STORE_PRE_COMPUTED_SUBMITTED in subs
        assert EventType.CB_RETRIEVE_SUBMITTED in subs
        assert EventType.CB_STORE_FINAL_SUBMITTED in subs

    def test_lookup_start_increments_counter(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id="req-1",
                metadata={"num_tokens": 128},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_lookup_end_normal(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-1",
                metadata={
                    "requested_tokens": 1024,
                    "hit_tokens": 768,
                    "fingerprint_hits": 4,
                    "storage_hits": 3,
                    "stale_chunks": 1,
                    "no_gpu_context": False,
                },
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_lookup_end_no_gpu_context_flag(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-1",
                metadata={
                    "requested_tokens": 0,
                    "hit_tokens": 0,
                    "fingerprint_hits": 0,
                    "storage_hits": 0,
                    "stale_chunks": 0,
                    "no_gpu_context": True,
                },
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_retrieve_success(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_START,
                session_id="req-2",
                metadata={"instance_id": 0, "num_chunks": 3},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id="req-2",
                metadata={"instance_id": 0, "num_chunks": 3, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_retrieve_failure_counted(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_START,
                session_id="req-2",
                metadata={"instance_id": 0, "num_chunks": 2},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id="req-2",
                metadata={"instance_id": 0, "num_chunks": 2, "success": False},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_store_pre_computed_failure_counted(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                session_id="req-3",
                metadata={"instance_id": 0, "num_tokens": 64},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                session_id="req-3",
                metadata={"instance_id": 0, "stored_chunks": 0, "success": False},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_store_final_failure_counted(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_START,
                session_id="req-4",
                metadata={"instance_id": 1, "num_tokens": 256},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_END,
                session_id="req-4",
                metadata={"instance_id": 1, "stored_chunks": 0, "success": False},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_fingerprints_registered(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_FINGERPRINTS_REGISTERED,
                metadata={"num_chunks": 8},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_chunks_evicted(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_CHUNKS_EVICTED,
                metadata={"num_chunks": 3},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_multiple_events_accumulate(self, bus, subscriber):
        bus.start()
        for _ in range(5):
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_START,
                    session_id="req-bulk",
                    metadata={"num_tokens": 100},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id="req-bulk",
                    metadata={
                        "requested_tokens": 96,
                        "hit_tokens": 32,
                        "fingerprint_hits": 2,
                        "storage_hits": 1,
                        "stale_chunks": 1,
                        "no_gpu_context": False,
                    },
                )
            )
        time.sleep(0.15)
        bus.stop()


# ---------------------------------------------------------------------------
# Blend token-level hit-rate counters
#
# These counters expose the numerator/denominator that let dashboards compute
# the blend hit rate identically to the L1+L2 lookup hit rate:
#
#     rate(lmcache_blend_lookup_hit_tokens_total[5m])
#     / rate(lmcache_blend_lookup_requested_tokens_total[5m])
#
# Asserts on actual counter deltas via the InMemoryMetricReader fixture.
# ---------------------------------------------------------------------------


class TestBlendLookupHitTokenCounters:
    def test_full_hit(self, bus, subscriber, snapshot):
        """All requested tokens are served by blend."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-1",
                metadata={
                    "requested_tokens": 1024,
                    "hit_tokens": 1024,
                    "fingerprint_hits": 4,
                    "storage_hits": 4,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_blend.lookup_requested_tokens"] == 1024
        assert delta["lmcache_blend.lookup_hit_tokens"] == 1024

    def test_partial_hit(self, bus, subscriber, snapshot):
        """A subset of the requested tokens is served by blend."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-2",
                metadata={
                    "requested_tokens": 1024,
                    "hit_tokens": 256,
                    "fingerprint_hits": 4,
                    "storage_hits": 1,
                    "stale_chunks": 3,
                    "no_gpu_context": False,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_blend.lookup_requested_tokens"] == 1024
        assert delta["lmcache_blend.lookup_hit_tokens"] == 256

    def test_full_miss_still_records_denominator(self, bus, subscriber, snapshot):
        """Cold lookup: the request must still increment the denominator so
        the running hit rate properly reflects the miss."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-3",
                metadata={
                    "requested_tokens": 512,
                    "hit_tokens": 0,
                    "fingerprint_hits": 0,
                    "storage_hits": 0,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_blend.lookup_requested_tokens"] == 512
        assert delta.get("lmcache_blend.lookup_hit_tokens", 0) == 0

    def test_no_gpu_context_records_zero_tokens(self, bus, subscriber, snapshot):
        """``no_gpu_context`` lookups emit ``hit_tokens=0`` and
        ``requested_tokens=0`` — neither counter should move so the ratio
        stays meaningful."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-4",
                metadata={
                    "requested_tokens": 0,
                    "hit_tokens": 0,
                    "fingerprint_hits": 5,
                    "storage_hits": 0,
                    "stale_chunks": 0,
                    "no_gpu_context": True,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta.get("lmcache_blend.lookup_requested_tokens", 0) == 0
        assert delta.get("lmcache_blend.lookup_hit_tokens", 0) == 0

    def test_multiple_lookups_accumulate(self, bus, subscriber, snapshot):
        """Counters accumulate across multiple completed lookups."""
        bus.start()
        # 3 full-hit lookups @ 256 tokens each
        for i in range(3):
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id=f"hit-{i}",
                    metadata={
                        "requested_tokens": 256,
                        "hit_tokens": 256,
                        "fingerprint_hits": 1,
                        "storage_hits": 1,
                        "stale_chunks": 0,
                        "no_gpu_context": False,
                    },
                )
            )
        # 2 partial-hit lookups: 1024 requested, 128 hit
        for i in range(2):
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id=f"partial-{i}",
                    metadata={
                        "requested_tokens": 1024,
                        "hit_tokens": 128,
                        "fingerprint_hits": 4,
                        "storage_hits": 1,
                        "stale_chunks": 3,
                        "no_gpu_context": False,
                    },
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        # 3*256 + 2*1024 = 768 + 2048 = 2816
        assert delta["lmcache_blend.lookup_requested_tokens"] == 2816
        # 3*256 + 2*128 = 768 + 256 = 1024
        assert delta["lmcache_blend.lookup_hit_tokens"] == 1024


class TestBlendL0GpuObservability:
    def test_store_pre_computed_records_l0_duration_and_transfer_counters(
        self, bus, subscriber
    ):
        duration_before = _histogram_count(
            "lmcache_blend.l0_gpu_operation_duration_seconds"
        )
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                session_id="req-store-pre",
                timestamp=100.0,
                metadata={"instance_id": 7, "num_tokens": 512},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                session_id="req-store-pre",
                timestamp=100.125,
                metadata={
                    "instance_id": 7,
                    "num_tokens": 512,
                    "stored_chunks": 2,
                    "success": True,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert (
            _histogram_count("lmcache_blend.l0_gpu_operation_duration_seconds")
            - duration_before
        ) == 1
        assert {
            "operation": "store_pre_computed",
            "instance_id": 7,
            "success": True,
        } in _histogram_attrs("lmcache_blend.l0_gpu_operation_duration_seconds")
        assert (
            _counter_sum(
                "lmcache_blend.l0_gpu_transfer_chunks",
                operation="store_pre_computed",
                instance_id=7,
                direction="gpu_to_l1",
            )
            >= 2
        )
        assert (
            _counter_sum(
                "lmcache_blend.l0_gpu_transfer_tokens",
                operation="store_pre_computed",
                instance_id=7,
                direction="gpu_to_l1",
            )
            >= 512
        )

    def test_retrieve_records_l0_transfer_direction_and_failure_duration(
        self, bus, subscriber
    ):
        duration_before = _histogram_count(
            "lmcache_blend.l0_gpu_operation_duration_seconds"
        )
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_START,
                session_id="req-retrieve",
                timestamp=200.0,
                metadata={"instance_id": 3, "num_chunks": 4, "num_tokens": 1024},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id="req-retrieve",
                timestamp=200.250,
                metadata={
                    "instance_id": 3,
                    "num_chunks": 4,
                    "num_tokens": 1024,
                    "success": False,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert (
            _histogram_count("lmcache_blend.l0_gpu_operation_duration_seconds")
            - duration_before
        ) == 1
        assert {
            "operation": "retrieve_pre_computed",
            "instance_id": 3,
            "success": False,
        } in _histogram_attrs("lmcache_blend.l0_gpu_operation_duration_seconds")
        assert (
            _counter_sum(
                "lmcache_blend.l0_gpu_transfer_chunks",
                operation="retrieve_pre_computed",
                instance_id=3,
                direction="l1_to_gpu",
            )
            == 0
        )

    def test_cb_l0_boundary_evidence_is_redacted(
        self, bus, subscriber, tmp_path, monkeypatch
    ):
        evidence_path = tmp_path / "cb-l0-boundary.jsonl"
        monkeypatch.setenv(L0_BLOCK_BOUNDARY_EVIDENCE_ENV, str(evidence_path))
        reset_l0_block_boundary_evidence_path_cache()
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_FINAL_SUBMITTED,
                    session_id="req-secret",
                    metadata={
                        "instance_id": 9,
                        "num_chunks": 3,
                        "num_tokens": 768,
                        "token_ids": [1, 2, 3],
                        "hashes": ["secret-hash"],
                        "block_ids": [10, 11],
                        "object_keys": ["secret-key"],
                    },
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_FINAL_START,
                    session_id="req-secret",
                    metadata={"instance_id": 9, "num_tokens": 768},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_STORE_FINAL_END,
                    session_id="req-secret",
                    metadata={
                        "instance_id": 9,
                        "num_tokens": 768,
                        "stored_chunks": 3,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()
            reset_l0_block_boundary_evidence_path_cache()

        events = [
            json.loads(line)
            for line in evidence_path.read_text(encoding="utf-8").splitlines()
        ]
        assert [event["stage"] for event in events] == [
            "cb_store_final_submitted",
            "cb_store_final_gpu_start",
            "cb_store_final_gpu_end",
        ]
        assert events[0]["schema_version"] == "inferguard-cb-l0-boundary-event/v1"
        assert events[0]["request_id"] == "req-secret"
        assert events[0]["operation"] == "store_final"
        assert events[0]["instance_id"] == 9
        assert events[0]["num_chunks"] == 3
        assert events[0]["num_tokens"] == 768
        assert events[-1]["success"] is True
        redacted = json.dumps(events)
        assert "token_ids" not in redacted
        assert "secret-hash" not in redacted
        assert "block_ids" not in redacted
        assert "secret-key" not in redacted
