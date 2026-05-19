# SPDX-License-Identifier: Apache-2.0

"""Tests for L0LifecycleSubscriber.

Uses ``InMemoryMetricReader`` to read back actual OTel histogram values
and verifies shadow-map eviction detection logic with END_SESSION tracking.

OTel only allows one MeterProvider per process, so we use a module-scoped
provider and assert on histogram observations.
"""

# Standard
from dataclasses import dataclass
import json
import subprocess
import sys
import textwrap
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
from lmcache.v1.mp_observability.subscribers.metrics.l0_lifecycle import (
    L0LifecycleSubscriber,
    _BlockStatus,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

# Time for the drain thread to process queued events.
_DRAIN_WAIT = 0.15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeBlockAllocationRecord:
    """Mimics BlockAllocationRecord for testing without importing vLLM types."""

    req_id: str
    new_block_ids: list[int]
    new_token_ids: list[int]


def _make_allocation_event(
    records: list[FakeBlockAllocationRecord],
    instance_id: int = 0,
    model_name: str = "test-model",
) -> Event:
    return Event(
        event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
        metadata={
            "instance_id": instance_id,
            "model_name": model_name,
            "records": records,
        },
    )


def _make_end_session_event(request_id: str) -> Event:
    return Event(
        event_type=EventType.MP_VLLM_END_SESSION,
        metadata={"request_id": request_id},
    )


def _read_histograms() -> dict[str, list]:
    """Snapshot all histogram data points."""
    data = _reader.get_metrics_data()
    result: dict[str, list] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                result[metric.name] = list(metric.data.data_points)
    return result


def _get_histogram_count(name: str) -> int:
    histograms = _read_histograms()
    dps = histograms.get(name, [])
    return sum(dp.count for dp in dps)


def _get_histogram_attrs(name: str) -> list[dict]:
    """Return a list of attribute dicts from all data points of a histogram."""
    histograms = _read_histograms()
    dps = histograms.get(name, [])
    return [dict(dp.attributes) for dp in dps if dp.count > 0]


def _read_counter_values() -> dict[str, int]:
    """Snapshot summed OTel counter values by metric name."""
    data = _reader.get_metrics_data()
    result: dict[str, int] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                total = 0
                has_value = False
                for dp in metric.data.data_points:
                    if not hasattr(dp, "value"):
                        continue
                    total += int(dp.value)
                    has_value = True
                if has_value:
                    result[metric.name] = result.get(metric.name, 0) + total
    return result


def _read_counter_values_by_attrs() -> dict[str, dict[tuple, int]]:
    """Snapshot counter values keyed by (metric_name, attr_tuple)."""
    data = _reader.get_metrics_data()
    result: dict[str, dict[tuple, int]] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                for dp in metric.data.data_points:
                    if not hasattr(dp, "value"):
                        continue
                    key = tuple(sorted(dict(dp.attributes).items()))
                    result.setdefault(metric.name, {})[key] = int(dp.value)
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = L0LifecycleSubscriber(sample_rate=1.0)
    bus.register_subscriber(sub)
    return sub


# ---------------------------------------------------------------------------
# Tests: New allocation
# ---------------------------------------------------------------------------


class TestL0NewAllocation:
    def test_boundary_evidence_records_processed_events(
        self,
        bus,
        subscriber,
        tmp_path,
        monkeypatch,
    ):
        evidence_path = tmp_path / "l0-boundary.jsonl"
        monkeypatch.setenv(L0_BLOCK_BOUNDARY_EVIDENCE_ENV, str(evidence_path))
        reset_l0_block_boundary_evidence_path_cache()
        bus.start()
        try:
            bus.publish(
                _make_allocation_event(
                    [FakeBlockAllocationRecord("req-1", [0, 1], [10, 20])]
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()
            reset_l0_block_boundary_evidence_path_cache()

        [event] = [
            json.loads(line)
            for line in evidence_path.read_text(encoding="utf-8").splitlines()
        ]
        assert event["source"] == "lmcache_l0_lifecycle_subscriber"
        assert event["stage"] == "l0_lifecycle_subscriber_processed"
        assert event["records"] == [{"request_id": "req-1", "block_count": 2}]
        assert event["metrics_updated_count"] == 2
        assert "new_token_ids" not in json.dumps(event)

    def test_block_allocation_updates_l0_block_counters(self, bus, subscriber):
        before = _read_counter_values()
        bus.start()
        try:
            bus.publish(
                _make_allocation_event(
                    [
                        FakeBlockAllocationRecord("req-1", [0, 1], [10, 20]),
                        FakeBlockAllocationRecord("req-2", [2], [30]),
                    ],
                    instance_id=42,
                    model_name="llama-7b",
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        after = _read_counter_values()
        assert (
            after.get("lmcache_mp.l0_block_allocation_records", 0)
            - before.get("lmcache_mp.l0_block_allocation_records", 0)
        ) == 2
        assert (
            after.get("lmcache_mp.l0_block_allocated_blocks", 0)
            - before.get("lmcache_mp.l0_block_allocated_blocks", 0)
        ) == 3

    def test_block_allocation_counters_export_to_prometheus(self):
        code = r"""
from dataclasses import dataclass
import os
import sys
import time

from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import REGISTRY, generate_latest

from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l0_lifecycle import (
    L0LifecycleSubscriber,
)

@dataclass
class Rec:
    req_id: str
    new_block_ids: list[int]
    new_token_ids: list[int]

reader = PrometheusMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
bus.register_subscriber(L0LifecycleSubscriber(sample_rate=1.0))
bus.start()
bus.publish(Event(event_type=EventType.MP_VLLM_BLOCK_ALLOCATION, metadata={
    "instance_id": 42,
    "model_name": "llama-7b",
    "records": [Rec("req-1", [0, 1], [10, 20])],
}))
time.sleep(0.2)
bus.stop()
for line in generate_latest(REGISTRY).decode().splitlines():
    if "lmcache_mp_l0_block" in line:
        print(line)
sys.stdout.flush()
os._exit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert "lmcache_mp_l0_block_allocation_records_total" in result.stdout
        assert "lmcache_mp_l0_block_allocated_blocks_total" in result.stdout
        assert 'instance_id="42"' in result.stdout
        assert 'model_name="llama-7b"' in result.stdout

    def test_new_block_no_lifecycle_histogram_emitted(self, bus, subscriber):
        count_before = _get_histogram_count("lmcache_mp.l0_block_lifetime_seconds")
        bus.start()
        try:
            bus.publish(
                _make_allocation_event(
                    [FakeBlockAllocationRecord("req-1", [0, 1, 2], [10, 20, 30])]
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        count_after = _get_histogram_count("lmcache_mp.l0_block_lifetime_seconds")
        assert count_after == count_before

    def test_shadow_map_populated(self, bus, subscriber):
        bus.start()
        try:
            bus.publish(
                _make_allocation_event(
                    [FakeBlockAllocationRecord("req-1", [10, 11], [100, 200])]
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        assert (0, 10) in subscriber._shadow
        assert (0, 11) in subscriber._shadow
        assert subscriber._shadow[(0, 10)].status == _BlockStatus.ACTIVE


# ---------------------------------------------------------------------------
# Tests: Prefix sharing (no reuse gap)
# ---------------------------------------------------------------------------


class TestL0PrefixSharing:
    def test_prefix_sharing_no_reuse_gap(self, bus, subscriber):
        """Two requests sharing the same block while both active = no reuse."""
        bus.start()

        # Request A allocates block 5.
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-A", [5], [42])])
        )
        time.sleep(_DRAIN_WAIT)

        reuse_before = _get_histogram_count("lmcache_mp.l0_block_reuse_gap_seconds")

        # Request B also uses block 5 with same tokens (prefix sharing).
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-B", [5], [42])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        reuse_after = _get_histogram_count("lmcache_mp.l0_block_reuse_gap_seconds")
        # No reuse gap should be recorded — this is prefix sharing.
        assert reuse_after == reuse_before

    def test_prefix_sharing_adds_owner(self, bus, subscriber):
        """Prefix sharing should add the new request as co-owner."""
        bus.start()
        try:
            bus.publish(
                _make_allocation_event([FakeBlockAllocationRecord("req-A", [6], [99])])
            )
            time.sleep(_DRAIN_WAIT)
            bus.publish(
                _make_allocation_event([FakeBlockAllocationRecord("req-B", [6], [99])])
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        state = subscriber._shadow[(0, 6)]
        assert "req-A" in state.owners
        assert "req-B" in state.owners
        assert state.status == _BlockStatus.ACTIVE


# ---------------------------------------------------------------------------
# Tests: END_SESSION and true reuse
# ---------------------------------------------------------------------------


class TestL0EndSessionAndReuse:
    def test_end_session_releases_block(self, bus, subscriber):
        """After END_SESSION, block with no remaining owners is RELEASED."""
        bus.start()
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-1", [7], [10])])
        )
        time.sleep(_DRAIN_WAIT)

        bus.publish(_make_end_session_event("req-1"))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert subscriber._shadow[(0, 7)].status == _BlockStatus.RELEASED

    def test_end_session_with_coowner_stays_active(self, bus, subscriber):
        """If another request still owns the block, it stays ACTIVE."""
        bus.start()
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-A", [8], [10])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-B", [8], [10])])
        )
        time.sleep(_DRAIN_WAIT)

        bus.publish(_make_end_session_event("req-A"))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        state = subscriber._shadow[(0, 8)]
        assert state.status == _BlockStatus.ACTIVE
        assert "req-A" not in state.owners
        assert "req-B" in state.owners

    def test_true_reuse_after_release(self, bus, subscriber):
        """Block released then reused with same tokens = true cache hit."""
        bus.start()

        # Allocate.
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-1", [9], [42])])
        )
        time.sleep(_DRAIN_WAIT)

        # Release.
        bus.publish(_make_end_session_event("req-1"))
        time.sleep(_DRAIN_WAIT)

        # Reuse with same tokens — true cache hit.
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-2", [9], [42])])
        )
        time.sleep(_DRAIN_WAIT)

        # Release again.
        bus.publish(_make_end_session_event("req-2"))
        time.sleep(_DRAIN_WAIT)

        # Another reuse.
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-3", [9], [42])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        state = subscriber._shadow[(0, 9)]
        # Two true reuses → 2 access history entries → 1 reuse gap.
        assert len(state.access_history) == 2


# ---------------------------------------------------------------------------
# Tests: Eviction detection
# ---------------------------------------------------------------------------


class TestL0EvictionDetection:
    def test_different_tokens_triggers_eviction(self, bus, subscriber):
        count_before = _get_histogram_count("lmcache_mp.l0_block_lifetime_seconds")
        bus.start()

        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-1", [3], [10])])
        )
        time.sleep(_DRAIN_WAIT)

        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-2", [3], [99])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        count_after = _get_histogram_count("lmcache_mp.l0_block_lifetime_seconds")
        assert count_after == count_before + 1

    def test_eviction_records_idle_time(self, bus, subscriber):
        idle_before = _get_histogram_count(
            "lmcache_mp.l0_block_idle_before_evict_seconds"
        )
        bus.start()

        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-1", [20], [1])])
        )
        time.sleep(_DRAIN_WAIT)

        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-2", [20], [2])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        idle_after = _get_histogram_count(
            "lmcache_mp.l0_block_idle_before_evict_seconds"
        )
        assert idle_after == idle_before + 1

    def test_eviction_clears_old_owners(self, bus, subscriber):
        """Eviction should clear old owner references."""
        bus.start()
        try:
            bus.publish(
                _make_allocation_event([FakeBlockAllocationRecord("req-1", [40], [1])])
            )
            time.sleep(_DRAIN_WAIT)
            bus.publish(
                _make_allocation_event([FakeBlockAllocationRecord("req-2", [40], [2])])
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        state = subscriber._shadow[(0, 40)]
        assert state.owners == {"req-2"}
        assert state.token_ids == [2]


# ---------------------------------------------------------------------------
# Tests: Reuse gap with proper release cycle
# ---------------------------------------------------------------------------


class TestL0ReuseGaps:
    def test_reuse_gap_after_release_and_reuse(self, bus, subscriber):
        """Proper release→reuse cycle should record reuse gaps on eviction."""
        bus.start()

        # Allocate block 30.
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-1", [30], [1])])
        )
        time.sleep(_DRAIN_WAIT)

        # Release and reuse twice (true cache hits).
        bus.publish(_make_end_session_event("req-1"))
        time.sleep(_DRAIN_WAIT)
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-2", [30], [1])])
        )
        time.sleep(_DRAIN_WAIT)

        bus.publish(_make_end_session_event("req-2"))
        time.sleep(_DRAIN_WAIT)
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-3", [30], [1])])
        )
        time.sleep(_DRAIN_WAIT)

        gap_before = _get_histogram_count("lmcache_mp.l0_block_reuse_gap_seconds")

        # Now evict to flush reuse gaps.
        bus.publish(
            _make_allocation_event(
                [FakeBlockAllocationRecord("req-evict", [30], [999])]
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        gap_after = _get_histogram_count("lmcache_mp.l0_block_reuse_gap_seconds")
        # 2 true reuses → 1 reuse gap.
        assert gap_after == gap_before + 1


# ---------------------------------------------------------------------------
# Tests: Sampling
# ---------------------------------------------------------------------------


class TestL0Sampling:
    def test_full_sample_rate_tracks_all(self, bus):
        sub = L0LifecycleSubscriber(sample_rate=1.0)
        bus.register_subscriber(sub)
        bus.start()

        for i in range(10):
            bus.publish(
                _make_allocation_event(
                    [FakeBlockAllocationRecord(f"req-{i}", [2000 + i], [i])]
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert len(sub._shadow) == 10


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestL0EdgeCases:
    def test_empty_block_ids(self, bus, subscriber):
        bus.start()
        try:
            bus.publish(
                _make_allocation_event([FakeBlockAllocationRecord("req-empty", [], [])])
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        assert len(subscriber._shadow) == 0

    def test_same_req_same_block_ignored(self, bus, subscriber):
        """Same request reporting same block again should be ignored."""
        bus.start()
        try:
            bus.publish(
                _make_allocation_event([FakeBlockAllocationRecord("req-1", [50], [1])])
            )
            time.sleep(_DRAIN_WAIT)

            # Same request, same block, same tokens — decode continuation.
            bus.publish(
                _make_allocation_event([FakeBlockAllocationRecord("req-1", [50], [1])])
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        state = subscriber._shadow[(0, 50)]
        # No access recorded — it was the same request.
        assert len(state.access_history) == 0

    def test_end_session_unknown_req(self, bus, subscriber):
        """END_SESSION for unknown req_id should not crash."""
        bus.start()
        try:
            bus.publish(_make_end_session_event("unknown-req"))
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()
        # No crash = pass.


# ---------------------------------------------------------------------------
# Tests: OTel attributes (instance_id, model_name)
# ---------------------------------------------------------------------------


class TestL0MetricAttributes:
    def test_eviction_emits_instance_id_and_model_name(self, bus, subscriber):
        """Histogram data points should carry instance_id and model_name."""
        bus.start()
        try:
            bus.publish(
                _make_allocation_event(
                    [FakeBlockAllocationRecord("req-1", [60], [10])],
                    instance_id=42,
                    model_name="llama-7b",
                )
            )
            time.sleep(_DRAIN_WAIT)
            bus.publish(
                _make_allocation_event(
                    [FakeBlockAllocationRecord("req-2", [60], [99])],
                    instance_id=42,
                    model_name="llama-7b",
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        attrs_list = _get_histogram_attrs("lmcache_mp.l0_block_lifetime_seconds")
        matching = [
            a
            for a in attrs_list
            if a.get("instance_id") == "42" and a.get("model_name") == "llama-7b"
        ]
        assert len(matching) > 0


# ---------------------------------------------------------------------------
# Tests: Subscription surface contract
# ---------------------------------------------------------------------------


class TestL0SubscriptionContract:
    """Verify get_subscriptions() returns exactly the expected event set."""

    def test_subscriptions_returns_exact_event_set(self):
        sub = L0LifecycleSubscriber(sample_rate=1.0)
        subs = sub.get_subscriptions()

        expected_keys = {
            EventType.MP_VLLM_BLOCK_ALLOCATION,
            EventType.MP_VLLM_END_SESSION,
        }
        assert set(subs.keys()) == expected_keys

    def test_subscriptions_all_values_callable(self):
        sub = L0LifecycleSubscriber(sample_rate=1.0)
        subs = sub.get_subscriptions()

        assert all(callable(v) for v in subs.values())


# ---------------------------------------------------------------------------
# Tests: Counter attribute strictness
# ---------------------------------------------------------------------------


class TestL0CounterAttributes:
    """Verify counter data points carry exact attribute dimensions."""

    def test_allocation_counter_attributes_include_instance_id_and_model_name(
        self, bus, subscriber
    ):
        """Counter data points must carry exactly instance_id and model_name.

        We snapshot before/after and look for the new data point created by
        our event. OTel accumulates across tests in-process, so we must
        match on the specific (instance_id, model_name) pair rather than
        assuming only one data point exists.
        """
        instance_id = 7
        model_name = "test-attr-model"

        before = _read_counter_values_by_attrs()

        bus.start()
        try:
            bus.publish(
                _make_allocation_event(
                    [FakeBlockAllocationRecord("req-attr", [70, 71], [100, 200])],
                    instance_id=instance_id,
                    model_name=model_name,
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        after = _read_counter_values_by_attrs()

        # Find the data point for our specific (instance_id, model_name).
        metric_name = "lmcache_mp.l0_block_allocation_records"
        attr_key = (("instance_id", str(instance_id)), ("model_name", model_name))
        before_val = before.get(metric_name, {}).get(attr_key, 0)
        after_val = after.get(metric_name, {}).get(attr_key, 0)
        assert after_val > before_val, (
            f"Expected new data point for {attr_key} in {metric_name}"
        )

        # Verify the data point carries exactly the expected attributes.
        data = _reader.get_metrics_data()
        assert data is not None
        for resource_metrics in data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name != metric_name:
                        continue
                    for dp in metric.data.data_points:
                        if not hasattr(dp, "value") or int(dp.value) == 0:
                            continue
                        attrs = dict(dp.attributes)
                        if attrs.get("instance_id") != str(instance_id):
                            continue
                        if attrs.get("model_name") != model_name:
                            continue
                        # Found our data point — verify exact attribute keys.
                        assert set(attrs.keys()) == {
                            "instance_id",
                            "model_name",
                        }, (
                            f"Expected exactly {{instance_id, model_name}}, "
                            f"got {set(attrs.keys())}"
                        )
                        return
        pytest.fail("No matching data point found for the published event")


class TestL0SkippedSetCap:
    """Skipped block tracking remains bounded under sampling misses."""

    def test_skipped_set_cap_evicts_entry_when_sampling_skips(self, monkeypatch):
        from lmcache.v1.mp_observability.subscribers.metrics import l0_lifecycle
        from lmcache.v1.mp_observability.subscribers.metrics.l0_lifecycle import (
            L0LifecycleSubscriber,
        )

        monkeypatch.setattr(l0_lifecycle, "_MAX_SKIPPED", 2)
        monkeypatch.setattr(L0LifecycleSubscriber, "_should_sample", lambda self: False)
        subscriber = L0LifecycleSubscriber(sample_rate=1.0)
        callback = subscriber.get_subscriptions()[EventType.MP_VLLM_BLOCK_ALLOCATION]

        for block_id in (1, 2, 3):
            callback(
                Event(
                    event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
                    metadata={
                        "instance_id": 0,
                        "model_name": "model",
                        "records": [
                            FakeBlockAllocationRecord(
                                req_id=f"req-{block_id}",
                                new_block_ids=[block_id],
                                new_token_ids=[block_id],
                            )
                        ],
                    },
                )
            )

        assert len(subscriber._skipped) == 2
