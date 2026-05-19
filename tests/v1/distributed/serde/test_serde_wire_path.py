# SPDX-License-Identifier: Apache-2.0

"""Wire-path tests: verify that AsyncSerdeProcessor actually publishes
CB_SERDE_ENCODE/DECODE events through the EventBus to the
SerdeMetricsSubscriber, producing correct OTel metric values.

These tests exercise the *real* production code path:
  AsyncSerdeProcessor._run_task
    → _publish_serde_event
      → get_event_bus().publish(Event)
        → EventBus drain thread
          → SerdeMetricsSubscriber callbacks
            → OTel instrument recordings

They do NOT mock the event bus, the subscriber, or the OTel instruments.
Only the sync Serializer/Deserializer are faked to avoid GPU dependencies.
"""

# Standard
import select
import time
from typing import Optional, Callable

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde import (
    AsyncSerdeProcessor,
    Deserializer,
    Serializer,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus
from lmcache.v1.mp_observability.subscribers.metrics.serde import (
    SerdeMetricsSubscriber,
)
from lmcache.v1.platform import consume_fd
from tests.v1.mp_observability.subscribers.metrics.otel_setup import (
    counter_delta,
    histogram_count,
    read_counters,
    reader as _reader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSerializer(Serializer):
    """Sync serializer that copies src size into dst or raises on cue."""

    def __init__(
        self,
        transform: Optional[Callable[[int], None]] = None,
        output_size_fraction: float = 0.5,
    ) -> None:
        self._transform = transform
        self._output_size_fraction = output_size_fraction
        self.calls = 0

    def serialize(self, src, dst) -> int:  # type: ignore[no-untyped-def]
        if self._transform is not None:
            self._transform(self.calls)
        self.calls += 1
        out_size = int(src.get_size() * self._output_size_fraction)
        return out_size

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        return 1


class _FakeDeserializer(Deserializer):
    """Sync deserializer that succeeds or raises on cue."""

    def __init__(
        self,
        transform: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._transform = transform
        self.calls = 0

    def deserialize(self, src, dst) -> None:  # type: ignore[no-untyped-def]
        if self._transform is not None:
            self._transform(self.calls)
        self.calls += 1


class _SizedObject:
    """Minimal MemoryObj-like object with a known byte size."""

    def __init__(self, size: int) -> None:
        self._size = size

    def get_size(self) -> int:
        return self._size


def _wait_for_fd(fd: int, timeout_s: float = 2.0) -> bool:
    """Wait until ``fd`` is readable or timeout. Drains the signal."""
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining_ms = int(max(0, (deadline - time.monotonic()) * 1000))
        if poller.poll(remaining_ms):
            try:
                consume_fd(fd)
            except OSError:
                pass
            return True
    return False


def _histogram_attrs_for(name: str) -> list[dict]:
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


def _counter_sum_for(name: str, **attrs: object) -> int:
    """Sum counter values matching given attributes."""
    data = _reader.get_metrics_data()
    total = 0
    if data is None:
        return total
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


_DRAIN_WAIT = 0.15


# ---------------------------------------------------------------------------
# Test: Encode success full wire path
# ---------------------------------------------------------------------------


class TestEncodeSuccessWirePath:
    """Verify encode START → END produces correct OTel metrics."""

    def test_encode_records_duration_histogram(self) -> None:
        """Encode success must produce exactly 1 observation in the encode
        duration histogram."""
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = histogram_count("lmcache_blend.serde_encode_duration_seconds")
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="fp8"
        )
        try:
            bus.start()
            task_id = processor.submit_serialize(
                [_SizedObject(4096)], [_SizedObject(2048)]
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        after = histogram_count("lmcache_blend.serde_encode_duration_seconds")
        assert after - before == 1

    def test_encode_duration_carries_serde_type_attribute(self) -> None:
        """The histogram observation must be tagged with serde_type='fp8'."""
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="fp8"
        )
        try:
            bus.start()
            task_id = processor.submit_serialize(
                [_SizedObject(4096)], [_SizedObject(2048)]
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        attrs = _histogram_attrs_for("lmcache_blend.serde_encode_duration_seconds")
        fp8_attrs = [a for a in attrs if a.get("serde_type") == "fp8"]
        assert len(fp8_attrs) >= 1, f"No fp8 attrs found in {attrs}"

    def test_encode_records_bytes_in_counter(self) -> None:
        """bytes_in must equal the source object size (4096)."""
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = read_counters()
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="fp8"
        )
        try:
            bus.start()
            task_id = processor.submit_serialize(
                [_SizedObject(4096)], [_SizedObject(2048)]
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        delta = counter_delta(before, read_counters())
        assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 4096

    def test_encode_records_bytes_out_counter(self) -> None:
        """bytes_out must reflect the serializer's return value."""
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = read_counters()
        # output_size_fraction=0.5 → serializer returns 4096*0.5=2048
        processor = AsyncSerdeProcessor(
            _FakeSerializer(output_size_fraction=0.5),
            _FakeDeserializer(),
            serde_type="fp8",
        )
        try:
            bus.start()
            task_id = processor.submit_serialize(
                [_SizedObject(4096)], [_SizedObject(2048)]
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        delta = counter_delta(before, read_counters())
        assert delta.get("lmcache_blend.serde_bytes_out", 0) >= 2048

    def test_encode_success_no_failure_counter(self) -> None:
        """Successful encode must NOT increment the failures counter."""
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = read_counters()
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="fp8"
        )
        try:
            bus.start()
            task_id = processor.submit_serialize(
                [_SizedObject(4096)], [_SizedObject(2048)]
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        delta = counter_delta(before, read_counters())
        assert delta.get("lmcache_blend.serde_failures", 0) == 0


# ---------------------------------------------------------------------------
# Test: Encode failure wire path
# ---------------------------------------------------------------------------


class TestEncodeFailureWirePath:
    """Verify encode failures produce failure metrics."""

    def test_encode_failure_increments_failure_counter(self) -> None:
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = read_counters()

        def _boom(_i: int) -> None:
            raise RuntimeError("encode OOM")

        processor = AsyncSerdeProcessor(
            _FakeSerializer(transform=_boom),
            _FakeDeserializer(),
            serde_type="naive",
        )
        try:
            bus.start()
            task_id = processor.submit_serialize(
                [_SizedObject(4096)], [_SizedObject(2048)]
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is False
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        delta = counter_delta(before, read_counters())
        assert delta.get("lmcache_blend.serde_failures", 0) >= 1

    def test_encode_failure_carries_correct_attributes(self) -> None:
        """Failure counter must be tagged with serde_type='naive',
        direction='encode', and the exception class name."""
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())

        def _boom(_i: int) -> None:
            raise RuntimeError("encode OOM")

        processor = AsyncSerdeProcessor(
            _FakeSerializer(transform=_boom),
            _FakeDeserializer(),
            serde_type="naive",
        )
        try:
            bus.start()
            task_id = processor.submit_serialize(
                [_SizedObject(4096)], [_SizedObject(2048)]
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is False
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        fail_count = _counter_sum_for(
            "lmcache_blend.serde_failures",
            serde_type="naive",
            direction="encode",
            failure_reason="RuntimeError",
        )
        assert fail_count >= 1

    def test_encode_failure_still_records_duration(self) -> None:
        """Even on failure, the duration histogram must have an observation."""
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = histogram_count("lmcache_blend.serde_encode_duration_seconds")

        def _boom(_i: int) -> None:
            raise RuntimeError("encode OOM")

        processor = AsyncSerdeProcessor(
            _FakeSerializer(transform=_boom),
            _FakeDeserializer(),
            serde_type="naive",
        )
        try:
            bus.start()
            task_id = processor.submit_serialize(
                [_SizedObject(4096)], [_SizedObject(2048)]
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is False
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        after = histogram_count("lmcache_blend.serde_encode_duration_seconds")
        assert after - before >= 1

    def test_encode_failure_records_bytes_in_from_src(self) -> None:
        """Failed encode should still record bytes_in from the source size."""
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = read_counters()

        def _boom(_i: int) -> None:
            raise RuntimeError("encode OOM")

        processor = AsyncSerdeProcessor(
            _FakeSerializer(transform=_boom),
            _FakeDeserializer(),
            serde_type="naive",
        )
        try:
            bus.start()
            task_id = processor.submit_serialize(
                [_SizedObject(4096)], [_SizedObject(2048)]
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is False
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        delta = counter_delta(before, read_counters())
        # bytes_in is always computed from src_objs sizes
        assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 4096


# ---------------------------------------------------------------------------
# Test: Decode success full wire path
# ---------------------------------------------------------------------------


class TestDecodeSuccessWirePath:
    """Verify decode START → END produces correct OTel metrics."""

    def test_decode_records_duration_histogram(self) -> None:
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = histogram_count("lmcache_blend.serde_decode_duration_seconds")
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="cachegen"
        )
        try:
            bus.start()
            task_id = processor.submit_deserialize(
                [_SizedObject(2048)], [_SizedObject(4096)]
            )
            assert _wait_for_fd(processor.get_deserialize_event_fd())
            assert processor.query_deserialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        after = histogram_count("lmcache_blend.serde_decode_duration_seconds")
        assert after - before == 1

    def test_decode_duration_carries_serde_type_attribute(self) -> None:
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="cachegen"
        )
        try:
            bus.start()
            task_id = processor.submit_deserialize(
                [_SizedObject(2048)], [_SizedObject(4096)]
            )
            assert _wait_for_fd(processor.get_deserialize_event_fd())
            assert processor.query_deserialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        attrs = _histogram_attrs_for("lmcache_blend.serde_decode_duration_seconds")
        cg_attrs = [a for a in attrs if a.get("serde_type") == "cachegen"]
        assert len(cg_attrs) >= 1, f"No cachegen attrs found in {attrs}"

    def test_decode_records_bytes_in_from_src(self) -> None:
        """bytes_in for decode is the compressed (source) size."""
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = read_counters()
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="cachegen"
        )
        try:
            bus.start()
            task_id = processor.submit_deserialize(
                [_SizedObject(2048)], [_SizedObject(4096)]
            )
            assert _wait_for_fd(processor.get_deserialize_event_fd())
            assert processor.query_deserialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        delta = counter_delta(before, read_counters())
        assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 2048

    def test_decode_records_bytes_out_from_dst(self) -> None:
        """bytes_out for decode is the decompressed (dst) size."""
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = read_counters()
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="cachegen"
        )
        try:
            bus.start()
            task_id = processor.submit_deserialize(
                [_SizedObject(2048)], [_SizedObject(4096)]
            )
            assert _wait_for_fd(processor.get_deserialize_event_fd())
            assert processor.query_deserialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        delta = counter_delta(before, read_counters())
        assert delta.get("lmcache_blend.serde_bytes_out", 0) >= 4096

    def test_decode_success_no_failure_counter(self) -> None:
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = read_counters()
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="cachegen"
        )
        try:
            bus.start()
            task_id = processor.submit_deserialize(
                [_SizedObject(2048)], [_SizedObject(4096)]
            )
            assert _wait_for_fd(processor.get_deserialize_event_fd())
            assert processor.query_deserialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        delta = counter_delta(before, read_counters())
        assert delta.get("lmcache_blend.serde_failures", 0) == 0


# ---------------------------------------------------------------------------
# Test: Decode failure wire path
# ---------------------------------------------------------------------------


class TestDecodeFailureWirePath:
    def test_decode_failure_increments_failure_counter(self) -> None:
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = read_counters()

        def _boom(_i: int) -> None:
            raise ValueError("corrupt data")

        processor = AsyncSerdeProcessor(
            _FakeSerializer(),
            _FakeDeserializer(transform=_boom),
            serde_type="fp8",
        )
        try:
            bus.start()
            task_id = processor.submit_deserialize(
                [_SizedObject(2048)], [_SizedObject(4096)]
            )
            assert _wait_for_fd(processor.get_deserialize_event_fd())
            assert processor.query_deserialize_result(task_id) is False
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        delta = counter_delta(before, read_counters())
        assert delta.get("lmcache_blend.serde_failures", 0) >= 1

    def test_decode_failure_carries_correct_attributes(self) -> None:
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())

        def _boom(_i: int) -> None:
            raise ValueError("corrupt data")

        processor = AsyncSerdeProcessor(
            _FakeSerializer(),
            _FakeDeserializer(transform=_boom),
            serde_type="fp8",
        )
        try:
            bus.start()
            task_id = processor.submit_deserialize(
                [_SizedObject(2048)], [_SizedObject(4096)]
            )
            assert _wait_for_fd(processor.get_deserialize_event_fd())
            assert processor.query_deserialize_result(task_id) is False
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        fail_count = _counter_sum_for(
            "lmcache_blend.serde_failures",
            serde_type="fp8",
            direction="decode",
            failure_reason="ValueError",
        )
        assert fail_count >= 1


# ---------------------------------------------------------------------------
# Test: Multi-object batch
# ---------------------------------------------------------------------------


class TestMultiObjectBatch:
    """Verify num_objects metadata reflects batch size."""

    def test_encode_batch_records_num_objects(self) -> None:
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="fp8"
        )
        try:
            bus.start()
            # 3 objects in the batch
            task_id = processor.submit_serialize(
                [_SizedObject(1024), _SizedObject(2048), _SizedObject(4096)],
                [_SizedObject(512), _SizedObject(1024), _SizedObject(2048)],
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        attrs = _histogram_attrs_for("lmcache_blend.serde_encode_duration_seconds")
        matching = [a for a in attrs if a.get("num_objects") == 3]
        assert len(matching) >= 1, f"Expected num_objects=3 in {attrs}"

    def test_encode_batch_sums_bytes_in(self) -> None:
        """bytes_in must be the sum of all source object sizes."""
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = read_counters()
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="fp8"
        )
        try:
            bus.start()
            task_id = processor.submit_serialize(
                [_SizedObject(1024), _SizedObject(2048), _SizedObject(4096)],
                [_SizedObject(512), _SizedObject(1024), _SizedObject(2048)],
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        delta = counter_delta(before, read_counters())
        # 1024 + 2048 + 4096 = 7168
        assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 7168

    def test_decode_batch_sums_bytes_out_from_dst(self) -> None:
        """bytes_out for decode must be the sum of dst object sizes."""
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = read_counters()
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="cachegen"
        )
        try:
            bus.start()
            task_id = processor.submit_deserialize(
                [_SizedObject(512), _SizedObject(1024)],
                [_SizedObject(2048), _SizedObject(4096)],
            )
            assert _wait_for_fd(processor.get_deserialize_event_fd())
            assert processor.query_deserialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        delta = counter_delta(before, read_counters())
        # 2048 + 4096 = 6144
        assert delta.get("lmcache_blend.serde_bytes_out", 0) >= 6144


# ---------------------------------------------------------------------------
# Test: Observability disabled → no events
# ---------------------------------------------------------------------------


class TestObservabilityDisabled:
    """When observability is off, no events should flow."""

    def test_no_events_when_observability_disabled(self) -> None:
        bus = init_event_bus(EventBusConfig(enabled=False))
        bus.register_subscriber(SerdeMetricsSubscriber())
        before = read_counters()
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="fp8"
        )
        try:
            # Don't start bus — it's disabled
            task_id = processor.submit_serialize(
                [_SizedObject(4096)], [_SizedObject(2048)]
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        delta = counter_delta(before, read_counters())
        # No bytes counters should have changed
        assert delta.get("lmcache_blend.serde_bytes_in", 0) == 0
        assert delta.get("lmcache_blend.serde_bytes_out", 0) == 0


# ---------------------------------------------------------------------------
# Test: Event session-id uniqueness
# ---------------------------------------------------------------------------


class TestEventSessionIdUniqueness:
    """Verify session IDs are unique across processors and tasks."""

    def test_different_processors_produce_different_session_ids(self) -> None:
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        session_ids: list[str] = []

        def _capture(event: Event) -> None:
            session_ids.append(event.session_id)

        bus.subscribe(EventType.CB_SERDE_ENCODE_START, _capture)
        proc_a = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="fp8"
        )
        proc_b = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="naive"
        )
        try:
            bus.start()
            task_a = proc_a.submit_serialize([_SizedObject(4096)], [_SizedObject(2048)])
            task_b = proc_b.submit_serialize([_SizedObject(4096)], [_SizedObject(2048)])
            assert _wait_for_fd(proc_a.get_serialize_event_fd())
            assert _wait_for_fd(proc_b.get_serialize_event_fd())
            assert proc_a.query_serialize_result(task_a) is True
            assert proc_b.query_serialize_result(task_b) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            proc_a.close()
            proc_b.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        assert len(session_ids) == 2
        assert session_ids[0] != session_ids[1]


# ---------------------------------------------------------------------------
# Test: serde_type metadata correctness per format
# ---------------------------------------------------------------------------


class TestSerdeTypeMetadata:
    """Verify serde_type flows from processor constructor into event metadata."""

    def test_serde_type_fp8_in_bytes_counters(self) -> None:
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="fp8"
        )
        try:
            bus.start()
            task_id = processor.submit_serialize(
                [_SizedObject(4096)], [_SizedObject(2048)]
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        # Check that bytes_in counter has an fp8-tagged data point
        fp8_in = _counter_sum_for(
            "lmcache_blend.serde_bytes_in",
            serde_type="fp8",
            direction="encode",
        )
        assert fp8_in >= 4096

    def test_serde_type_naive_in_bytes_counters(self) -> None:
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        processor = AsyncSerdeProcessor(
            _FakeSerializer(output_size_fraction=1.0),
            _FakeDeserializer(),
            serde_type="naive",
        )
        try:
            bus.start()
            task_id = processor.submit_serialize(
                [_SizedObject(4096)], [_SizedObject(4096)]
            )
            assert _wait_for_fd(processor.get_serialize_event_fd())
            assert processor.query_serialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        naive_in = _counter_sum_for(
            "lmcache_blend.serde_bytes_in",
            serde_type="naive",
            direction="encode",
        )
        assert naive_in >= 4096

    def test_serde_type_cachegen_in_decode_duration(self) -> None:
        bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(SerdeMetricsSubscriber())
        processor = AsyncSerdeProcessor(
            _FakeSerializer(), _FakeDeserializer(), serde_type="cachegen"
        )
        try:
            bus.start()
            task_id = processor.submit_deserialize(
                [_SizedObject(2048)], [_SizedObject(4096)]
            )
            assert _wait_for_fd(processor.get_deserialize_event_fd())
            assert processor.query_deserialize_result(task_id) is True
            time.sleep(_DRAIN_WAIT)
        finally:
            processor.close()
            bus.stop()
            init_event_bus(EventBusConfig(enabled=False))

        attrs = _histogram_attrs_for("lmcache_blend.serde_decode_duration_seconds")
        cg = [a for a in attrs if a.get("serde_type") == "cachegen"]
        assert len(cg) >= 1
