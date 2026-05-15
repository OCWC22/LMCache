# SPDX-License-Identifier: Apache-2.0

"""Tests for MPServerLoggingSubscriber."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.logging.mp_server import (
    MPServerLoggingSubscriber,
)


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = MPServerLoggingSubscriber()
    bus.register_subscriber(sub)
    return sub


class TestMPServerLoggingSubscriber:
    def test_subscriptions_cover_all_mp_server_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.MP_STORE_START in subs
        assert EventType.MP_STORE_END in subs
        assert EventType.MP_RETRIEVE_START in subs
        assert EventType.MP_RETRIEVE_END in subs
        assert EventType.MP_LOOKUP_PREFETCH_START in subs
        assert EventType.MP_LOOKUP_PREFETCH_END in subs

    def test_store_start_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id="req-1",
                metadata={"device": "cuda:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_store_end_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id="req-1",
                metadata={"device": "cuda:0", "stored_count": 5},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_retrieve_start_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id="req-2",
                metadata={"device": "cuda:1"},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_retrieve_end_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id="req-2",
                metadata={"device": "cuda:1", "retrieved_count": 3},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_lookup_prefetch_start_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id="req-3",
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_lookup_prefetch_end_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id="req-3",
                metadata={"found_count": 10},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_block_allocation_no_raw_block_ids(self, bus, subscriber):
        """Block allocation log must not contain raw block ID integers.

        Redaction policy: debug logs should summarise counts only, not
        leak raw block IDs (consistent with boundary-evidence and
        Prometheus redaction).
        """
        import logging
        from dataclasses import dataclass

        from lmcache.v1.mp_observability.event import Event, EventType

        @dataclass
        class _FakeRecord:
            req_id: str
            new_block_ids: list[int]
            new_token_ids: list[int]

        # Capture log output.
        messages: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                messages.append(self.format(record))

        cap = _Capture()
        sub_logger = logging.getLogger(
            "lmcache.v1.mp_observability.subscribers.logging.mp_server"
        )
        sub_logger.addHandler(cap)
        sub_logger.setLevel(logging.DEBUG)

        try:
            bus.start()
            bus.publish(
                Event(
                    event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
                    session_id="req-blk",
                    metadata={
                        "records": [
                            _FakeRecord(
                                req_id="req-blk",
                                new_block_ids=[100, 200, 300, 400, 500],
                                new_token_ids=[10, 20, 30, 40, 50],
                            )
                        ],
                    },
                )
            )
            time.sleep(0.15)
            bus.stop()

            # Find block allocation log messages.
            alloc_msgs = [m for m in messages if "block allocation" in m]
            assert alloc_msgs, "Expected at least one block allocation log message"

            # Raw block IDs (the integer values 100, 200, etc.) must NOT
            # appear in the log output.
            for msg in alloc_msgs:
                # The string repr of the list would be "[100, 200, 300, ...]"
                # or individual integers as standalone words.
                assert "[100" not in msg, (
                    f"Raw block ID list leaked in log: {msg}"
                )
                assert "num_blocks=" in msg, (
                    f"Expected num_blocks= summary, got: {msg}"
                )
        finally:
            sub_logger.removeHandler(cap)

    def test_multiple_events_no_crash(self, bus, subscriber):
        bus.start()
        for i in range(10):
            bus.publish(
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=f"req-{i}",
                    metadata={"device": "cuda:0"},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.MP_STORE_END,
                    session_id=f"req-{i}",
                    metadata={"device": "cuda:0", "stored_count": i},
                )
            )
        time.sleep(0.15)
        bus.stop()
