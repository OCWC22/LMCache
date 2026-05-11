# SPDX-License-Identifier: Apache-2.0

"""Opt-in redacted L0 block-allocation boundary evidence helpers."""

# Future
from __future__ import annotations

# Standard
import json
import os
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

L0_BLOCK_BOUNDARY_EVIDENCE_ENV = "INFERGUARD_L0_BLOCK_BOUNDARY_EVIDENCE_PATH"
_cached_l0_block_boundary_evidence_path = ""
_l0_block_boundary_evidence_path_initialized = False


def _get_l0_block_boundary_evidence_path() -> str:
    """Return the cached opt-in boundary evidence path."""
    global _cached_l0_block_boundary_evidence_path
    global _l0_block_boundary_evidence_path_initialized

    if not _l0_block_boundary_evidence_path_initialized:
        _cached_l0_block_boundary_evidence_path = os.environ.get(
            L0_BLOCK_BOUNDARY_EVIDENCE_ENV, ""
        ).strip()
        _l0_block_boundary_evidence_path_initialized = True
    return _cached_l0_block_boundary_evidence_path


def reset_l0_block_boundary_evidence_path_cache() -> None:
    """Reset cached boundary evidence path for tests that patch the env."""
    global _cached_l0_block_boundary_evidence_path
    global _l0_block_boundary_evidence_path_initialized

    _cached_l0_block_boundary_evidence_path = ""
    _l0_block_boundary_evidence_path_initialized = False


def append_l0_block_boundary_event(
    source: str,
    stage: str,
    records: list[object],
    *,
    metrics_updated_count: int | None = None,
) -> None:
    """Append redacted L0 block-allocation boundary evidence when requested.

    Args:
        source: Component emitting the boundary checkpoint.
        stage: Boundary stage name within the component.
        records: Block-allocation records. Only request ID and block count are
            emitted; token IDs and block IDs are intentionally redacted.
        metrics_updated_count: Optional count of metric instruments updated while
            processing the records.
    """
    path = _get_l0_block_boundary_evidence_path()
    if not path:
        return

    payload = {
        "schema_version": "inferguard-l0-block-boundary-event/v1",
        "source": source,
        "stage": stage,
        "timestamp_unix": time.time(),
        "records": [
            {
                "request_id": getattr(record, "req_id", ""),
                "block_count": len(getattr(record, "new_block_ids", []) or []),
            }
            for record in records
        ],
    }
    if metrics_updated_count is not None:
        payload["metrics_updated_count"] = metrics_updated_count

    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        logger.debug("Failed to append L0 block boundary evidence", exc_info=True)
