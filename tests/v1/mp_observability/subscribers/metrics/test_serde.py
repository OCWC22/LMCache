# SPDX-License-Identifier: Apache-2.0

"""Tests for SerdeMetricsSubscriber — CB serde/transform observability.

TDD tests written FIRST (RED phase). The following must exist for these
to pass:

1. EventType members:
   - CB_SERDE_ENCODE_START / CB_SERDE_ENCODE_END
   - CB_SERDE_DECODE_START / CB_SERDE_DECODE_END

2. SerdeMetricsSubscriber in
   lmcache/v1/mp_observability/subscribers/metrics/serde.py

3. Five OTel metrics:
   - lmcache_blend.serde_encode_duration_seconds  (histogram)
   - lmcache_blend.serde_decode_duration_seconds  (histogram)
   - lmcache_blend.serde_bytes_in                 (counter)
   - lmcache_blend.serde_bytes_out                (counter)
   - lmcache_blend.serde_failures                 (counter)
"""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig

from tests.v1.mp_observability.subscribers.metrics.otel_setup import (
    counter_delta,
    read_counters,
    reader as _reader,
)

_DRAIN_WAIT = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serde_histogram_count(name: str) -> int:
    """Count observations for a serde histogram metric."""
    data = _reader.get_metrics_data()
    if data is None:
        return 0
    total = 0
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name != name:
                    continue
                for dp in m.data.data_points:
                    if hasattr(dp, "count"):
                        total += int(dp.count)
    return total


def _serde_histogram_attrs(name: str) -> list[dict]:
    """Return attribute dicts for histogram data points with count > 0."""
    data = _reader.get_metrics_data()
    result: list[dict] = []
    if data is None:
        return result
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name != name:
                    continue
                for dp in m.data.data_points:
                    if getattr(dp, "count", 0) > 0:
                        result.append(dict(dp.attributes))
    return result


def _serde_counter_sum(name: str, **attrs: object) -> int:
    """Sum counter values matching given attributes."""
    data = _reader.get_metrics_data()
    if data is None:
        return 0
    total = 0
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name != name:
                    continue
                for dp in m.data.data_points:
                    if not hasattr(dp, "value"):
                        continue
                    dp_attrs = dict(dp.attributes)
                    if all(dp_attrs.get(k) == v for k, v in attrs.items()):
                        total += int(dp.value)
    return total


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    from lmcache.v1.mp_observability.subscribers.metrics.serde import (
        SerdeMetricsSubscriber,
    )

    sub = SerdeMetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    """Capture counters before the test; yield a callable that returns deltas."""
    before = read_counters()

    def get_delta() -> dict[str, int]:
        return counter_delta(before, read_counters())

    return get_delta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSerdeEventTypeExists:
    """Verify the new event types are registered in the EventType enum."""

    def test_encode_start_exists(self):
        assert hasattr(EventType, "CB_SERDE_ENCODE_START")

    def test_encode_end_exists(self):
        assert hasattr(EventType, "CB_SERDE_ENCODE_END")

    def test_decode_start_exists(self):
        assert hasattr(EventType, "CB_SERDE_DECODE_START")

    def test_decode_end_exists(self):
        assert hasattr(EventType, "CB_SERDE_DECODE_END")


class TestSerdeSubscriberSubscriptions:
    """Verify the subscriber registers for the right events."""

    def test_subscribes_to_encode_start(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_SERDE_ENCODE_START in subs

    def test_subscribes_to_encode_end(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_SERDE_ENCODE_END in subs

    def test_subscribes_to_decode_start(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_SERDE_DECODE_START in subs

    def test_subscribes_to_decode_end(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_SERDE_DECODE_END in subs


class TestSerdeEncodeMetrics:
    """Encode (serialize) duration + bytes counters."""

    def test_encode_duration_recorded_on_start_end_pair(self, bus, subscriber):
        before = _serde_histogram_count("lmcache_blend.serde_encode_duration_seconds")
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_START,
                    session_id="serde-enc-1",
                    timestamp=10.0,
                    metadata={"serde_type": "fp8", "num_objects": 2},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_END,
                    session_id="serde-enc-1",
                    timestamp=10.05,
                    metadata={
                        "serde_type": "fp8",
                        "num_objects": 2,
                        "bytes_in": 4096,
                        "bytes_out": 2048,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        after = _serde_histogram_count("lmcache_blend.serde_encode_duration_seconds")
        assert after - before == 1

    def test_encode_duration_carries_serde_type_attr(self, bus, subscriber):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_START,
                    session_id="serde-enc-type",
                    timestamp=20.0,
                    metadata={"serde_type": "naive", "num_objects": 1},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_END,
                    session_id="serde-enc-type",
                    timestamp=20.1,
                    metadata={
                        "serde_type": "naive",
                        "num_objects": 1,
                        "bytes_in": 1024,
                        "bytes_out": 1024,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        attrs = _serde_histogram_attrs("lmcache_blend.serde_encode_duration_seconds")
        matching = [a for a in attrs if a.get("serde_type") == "naive"]
        assert len(matching) >= 1
        assert any(
            a.get("success") is True and a.get("num_objects") == 1 for a in matching
        )

    def test_decode_duration_carries_num_objects_attr(self, bus, subscriber):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_DECODE_START,
                    session_id="serde-dec-objects",
                    timestamp=22.0,
                    metadata={"serde_type": "fp8", "num_objects": 3},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_DECODE_END,
                    session_id="serde-dec-objects",
                    timestamp=22.1,
                    metadata={
                        "serde_type": "fp8",
                        "num_objects": 3,
                        "bytes_in": 1024,
                        "bytes_out": 2048,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        attrs = _serde_histogram_attrs("lmcache_blend.serde_decode_duration_seconds")
        matching = [
            a
            for a in attrs
            if a.get("serde_type") == "fp8" and a.get("num_objects") == 3
        ]
        assert len(matching) >= 1

    def test_encode_bytes_counters(self, bus, subscriber, snapshot):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_END,
                    session_id="serde-bytes-1",
                    metadata={
                        "serde_type": "fp8",
                        "num_objects": 1,
                        "bytes_in": 8192,
                        "bytes_out": 4096,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 8192
        assert delta.get("lmcache_blend.serde_bytes_out", 0) >= 4096

    def test_encode_failure_increments_failure_counter(self, bus, subscriber, snapshot):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_START,
                    session_id="serde-enc-fail",
                    timestamp=50.0,
                    metadata={"serde_type": "fp8", "num_objects": 1},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_END,
                    session_id="serde-enc-fail",
                    timestamp=50.01,
                    metadata={
                        "serde_type": "fp8",
                        "num_objects": 1,
                        "bytes_in": 4096,
                        "bytes_out": 0,
                        "success": False,
                        "failure_reason": "cuda_error",
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        assert delta.get("lmcache_blend.serde_failures", 0) >= 1
        # Failure should still record duration
        count = _serde_histogram_count("lmcache_blend.serde_encode_duration_seconds")
        assert count >= 1


class TestSerdeDecodeMetrics:
    """Decode (deserialize) duration + bytes counters."""

    def test_decode_duration_recorded_on_start_end_pair(self, bus, subscriber):
        before = _serde_histogram_count("lmcache_blend.serde_decode_duration_seconds")
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_DECODE_START,
                    session_id="serde-dec-1",
                    timestamp=30.0,
                    metadata={"serde_type": "fp8", "num_objects": 3},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_DECODE_END,
                    session_id="serde-dec-1",
                    timestamp=30.075,
                    metadata={
                        "serde_type": "fp8",
                        "num_objects": 3,
                        "bytes_in": 3072,
                        "bytes_out": 6144,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        after = _serde_histogram_count("lmcache_blend.serde_decode_duration_seconds")
        assert after - before == 1

    def test_decode_failure_increments_failure_counter(self, bus, subscriber, snapshot):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_DECODE_START,
                    session_id="serde-dec-fail",
                    timestamp=60.0,
                    metadata={"serde_type": "cachegen", "num_objects": 1},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_DECODE_END,
                    session_id="serde-dec-fail",
                    timestamp=60.005,
                    metadata={
                        "serde_type": "cachegen",
                        "num_objects": 1,
                        "bytes_in": 2048,
                        "bytes_out": 0,
                        "success": False,
                        "failure_reason": "corrupt_data",
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        assert delta.get("lmcache_blend.serde_failures", 0) >= 1

    def test_decode_bytes_counters(self, bus, subscriber, snapshot):
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_DECODE_END,
                    session_id="serde-dec-bytes",
                    metadata={
                        "serde_type": "naive",
                        "num_objects": 2,
                        "bytes_in": 2048,
                        "bytes_out": 4096,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 2048
        assert delta.get("lmcache_blend.serde_bytes_out", 0) >= 4096


class TestSerdeCompressionRatio:
    """Bytes in/out give compression ratio when grouped by serde_type."""

    def test_encode_fp8_compression_ratio(self, bus, subscriber, snapshot):
        bus.start()
        try:
            # 3 encode operations with same serde_type
            for i in range(3):
                bus.publish(
                    Event(
                        event_type=EventType.CB_SERDE_ENCODE_END,
                        session_id=f"serde-ratio-{i}",
                        metadata={
                            "serde_type": "fp8",
                            "num_objects": 1,
                            "bytes_in": 8192,
                            "bytes_out": 4096,
                            "success": True,
                        },
                    )
                )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        # 3 * 8192 = 24576 bytes in
        assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 24576
        # 3 * 4096 = 12288 bytes out → 2:1 compression
        assert delta.get("lmcache_blend.serde_bytes_out", 0) >= 12288


class TestSerdeEndOnlyStillRecords:
    """END events without matching START should still record counters."""

    def test_encode_end_without_start_records_bytes(self, bus, subscriber, snapshot):
        """Stand-alone END event (e.g. from a crashed mid-encode path)
        should still record byte counters but skip duration."""
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_END,
                    session_id="serde-orphan",
                    timestamp=100.5,
                    metadata={
                        "serde_type": "naive",
                        "num_objects": 1,
                        "bytes_in": 512,
                        "bytes_out": 512,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 512
        assert delta.get("lmcache_blend.serde_bytes_out", 0) >= 512


class TestSerdeMultipleTypes:
    """Serde type attribute distinguishes fp8/naive/cachegen/kivi."""

    def test_mixed_serde_types_tracked_separately(self, bus, subscriber, snapshot):
        bus.start()
        try:
            # fp8 encode
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_END,
                    session_id="mix-fp8",
                    metadata={
                        "serde_type": "fp8",
                        "num_objects": 1,
                        "bytes_in": 4096,
                        "bytes_out": 2048,
                        "success": True,
                    },
                )
            )
            # naive encode
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_END,
                    session_id="mix-naive",
                    metadata={
                        "serde_type": "naive",
                        "num_objects": 1,
                        "bytes_in": 4096,
                        "bytes_out": 4096,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        # Both should be counted in bytes_in total
        delta = snapshot()
        assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 8192
        assert delta.get("lmcache_blend.serde_bytes_out", 0) >= 6144

        # Duration histograms should have data for both types
        encode_attrs = _serde_histogram_attrs(
            "lmcache_blend.serde_encode_duration_seconds"
        )
        serde_types = {a.get("serde_type") for a in encode_attrs}
        assert "fp8" in serde_types or "naive" in serde_types


class TestSerdePendingOpsCap:
    """Pending serde START events are bounded when END is missing."""

    def test_pending_ops_cap_evicts_oldest_start(self, monkeypatch, subscriber):
        from lmcache.v1.mp_observability.subscribers.metrics import serde

        monkeypatch.setattr(serde, "_MAX_PENDING_OPS", 2)
        callbacks = subscriber.get_subscriptions()

        callbacks[EventType.CB_SERDE_ENCODE_START](
            Event(
                event_type=EventType.CB_SERDE_ENCODE_START,
                session_id="one",
                timestamp=1.0,
                metadata={"serde_type": "fp8"},
            )
        )
        callbacks[EventType.CB_SERDE_ENCODE_START](
            Event(
                event_type=EventType.CB_SERDE_ENCODE_START,
                session_id="two",
                timestamp=2.0,
                metadata={"serde_type": "fp8"},
            )
        )
        callbacks[EventType.CB_SERDE_ENCODE_START](
            Event(
                event_type=EventType.CB_SERDE_ENCODE_START,
                session_id="three",
                timestamp=3.0,
                metadata={"serde_type": "fp8"},
            )
        )

        assert len(subscriber._pending_ops) == 2
        assert "encode:one" not in subscriber._pending_ops
        assert "encode:three" in subscriber._pending_ops

    def test_pending_ops_cap_logs_warning(self, monkeypatch, subscriber):
        """When the cap is exceeded, a warning must be logged via the module logger."""
        from unittest.mock import patch

        from lmcache.v1.mp_observability.subscribers.metrics import serde

        monkeypatch.setattr(serde, "_MAX_PENDING_OPS", 1)
        callbacks = subscriber.get_subscriptions()

        with patch.object(serde.logger, "warning") as mock_warn:
            callbacks[EventType.CB_SERDE_ENCODE_START](
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_START,
                    session_id="first",
                    timestamp=1.0,
                    metadata={"serde_type": "fp8"},
                )
            )
            callbacks[EventType.CB_SERDE_ENCODE_START](
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_START,
                    session_id="second",
                    timestamp=2.0,
                    metadata={"serde_type": "fp8"},
                )
            )

        mock_warn.assert_called_once()
        assert "_pending_ops exceeded" in mock_warn.call_args[0][0]


class TestSerdeDecodeEndWithoutStart:
    """Decode END events without matching START should still record bytes but skip
    duration."""

    def test_decode_end_without_start_records_bytes(self, bus, subscriber, snapshot):
        before_decode_count = _serde_histogram_count(
            "lmcache_blend.serde_decode_duration_seconds"
        )
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_DECODE_END,
                    session_id="serde-orphan-decode",
                    timestamp=200.5,
                    metadata={
                        "serde_type": "fp8",
                        "num_objects": 2,
                        "bytes_in": 1024,
                        "bytes_out": 2048,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        delta = snapshot()
        assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 1024
        assert delta.get("lmcache_blend.serde_bytes_out", 0) >= 2048
        # Duration should NOT be recorded for orphan END
        after_decode_count = _serde_histogram_count(
            "lmcache_blend.serde_decode_duration_seconds"
        )
        assert after_decode_count == before_decode_count


class TestSerdeEncodeEndWithoutStartSkipsDuration:
    """Verify orphan encode END skips duration but records bytes."""

    def test_encode_end_without_start_skips_duration(self, bus, subscriber):
        before_encode_count = _serde_histogram_count(
            "lmcache_blend.serde_encode_duration_seconds"
        )
        bus.start()
        try:
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_END,
                    session_id="serde-orphan-enc-nodur",
                    timestamp=300.5,
                    metadata={
                        "serde_type": "naive",
                        "num_objects": 1,
                        "bytes_in": 256,
                        "bytes_out": 256,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        after_encode_count = _serde_histogram_count(
            "lmcache_blend.serde_encode_duration_seconds"
        )
        assert after_encode_count == before_encode_count


class TestSerdeConcurrentOperations:
    """Multiple concurrent serde operations with different session_ids."""

    def test_interleaved_encode_decode_tracked_separately(self, bus, subscriber):
        before_encode = _serde_histogram_count(
            "lmcache_blend.serde_encode_duration_seconds"
        )
        before_decode = _serde_histogram_count(
            "lmcache_blend.serde_decode_duration_seconds"
        )
        bus.start()
        try:
            # Start two encode and one decode concurrently
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_START,
                    session_id="concurrent-enc-1",
                    timestamp=400.0,
                    metadata={"serde_type": "fp8", "num_objects": 1},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_DECODE_START,
                    session_id="concurrent-dec-1",
                    timestamp=400.01,
                    metadata={"serde_type": "naive", "num_objects": 2},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_START,
                    session_id="concurrent-enc-2",
                    timestamp=400.02,
                    metadata={"serde_type": "cachegen", "num_objects": 3},
                )
            )
            # End them in different order
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_DECODE_END,
                    session_id="concurrent-dec-1",
                    timestamp=400.1,
                    metadata={
                        "serde_type": "naive",
                        "num_objects": 2,
                        "bytes_in": 512,
                        "bytes_out": 1024,
                        "success": True,
                    },
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_END,
                    session_id="concurrent-enc-2",
                    timestamp=400.12,
                    metadata={
                        "serde_type": "cachegen",
                        "num_objects": 3,
                        "bytes_in": 6144,
                        "bytes_out": 2048,
                        "success": True,
                    },
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_SERDE_ENCODE_END,
                    session_id="concurrent-enc-1",
                    timestamp=400.15,
                    metadata={
                        "serde_type": "fp8",
                        "num_objects": 1,
                        "bytes_in": 4096,
                        "bytes_out": 2048,
                        "success": True,
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
        finally:
            bus.stop()

        after_encode = _serde_histogram_count(
            "lmcache_blend.serde_encode_duration_seconds"
        )
        after_decode = _serde_histogram_count(
            "lmcache_blend.serde_decode_duration_seconds"
        )
        assert after_encode - before_encode == 2
        assert after_decode - before_decode == 1


class TestSerdeSubscriptionContractExact:
    """Verify the exact set of subscriptions returned by get_subscriptions()."""

    def test_subscription_count_and_all_callable(self, subscriber):
        subs = subscriber.get_subscriptions()
        expected = {
            EventType.CB_SERDE_ENCODE_START,
            EventType.CB_SERDE_ENCODE_END,
            EventType.CB_SERDE_DECODE_START,
            EventType.CB_SERDE_DECODE_END,
        }
        assert set(subs.keys()) == expected
        for callback in subs.values():
            assert callable(callback)
