"""Quality-gate tests use existing TASK001 synthetic ingestion only in tmp dirs."""
import dataclasses
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.ai.price.artifacts import digest, write_new
from quant_platform.ai.price.audit import audit_bars, load_release, parent_review_disposition
from quant_platform.ai.price.config import TrainingConfig
from quant_platform.domain import Bar, QualityStatus

T = datetime(2024, 1, 1, tzinfo=timezone.utc)


def bar(t):
    return Bar(instrument_id="crypto:kraken:spot:BTC-USD", source="synthetic://schema-only", source_symbol="XXBTZUSD",
               event_time_utc=t, close_time_utc=t+timedelta(days=1), available_at_utc=t+timedelta(days=1, seconds=60),
               collected_at_utc=t+timedelta(days=2), timeframe="1d", raw_sha256="a"*64,
               clean_version="synthetic", quality_status=QualityStatus.REVIEW_REQUIRED,
               availability_basis="INFERRED", open=Decimal(100), high=Decimal(110), low=Decimal(90),
               close=Decimal(105), base_volume=Decimal(10), quote_volume=None, is_final=True)


def quality():
    return {"status": "REVIEW_REQUIRED", "synthetic_test_only": False, "invalid_rows": 0,
            "duplicate_rows": 0, "fatal_errors": [], "unfinished_rows": 0, "missing_candles": 1,
            "gaps": [{"after_utc": "2013-01-01T00:00:00Z", "before_utc": "2013-01-03T00:00:00Z", "missing_candles": 1}]}


def test_scoped_audit_checks_missing_endpoints_and_does_not_fill():
    bars = [bar(T+timedelta(days=i)) for i in range(3)]
    assert audit_bars(bars, T, T+timedelta(days=3))["status"] == "PASS"
    for incomplete in (bars[:-1], bars[1:], [bars[0], bars[2]], [bars[0], bars[0], bars[2]]):
        result = audit_bars(incomplete, T, T+timedelta(days=3))
        assert result["status"] == "FAIL" and result["interpolated_rows"] == 0


def test_synthetic_flags_prevent_release_even_with_complete_calendar():
    assert audit_bars([dataclasses.replace(bar(T), quality_flags=("SYNTHETIC_TEST_ONLY",))], T, T+timedelta(days=1))["status"] == "FAIL"


def test_parent_review_is_disposed_explicitly_without_mutation():
    original = quality()
    before = json.dumps(original, sort_keys=True)
    disposition = parent_review_disposition(original, T)
    assert disposition["missing_candles_excluded"] == 1
    assert disposition["parent_status_retained"] == "REVIEW_REQUIRED"
    assert json.dumps(original, sort_keys=True) == before


@pytest.mark.parametrize("change", [
    {"status": "FAIL"}, {"synthetic_test_only": True}, {"invalid_rows": 1},
    {"duplicate_rows": 1}, {"unfinished_rows": 1}, {"fatal_errors": ["broken"]},
    {"gaps": []}, {"missing_candles": 2},
])
def test_parent_unresolved_quality_cannot_be_silently_approved(change):
    with pytest.raises(ValueError):
        parent_review_disposition({**quality(), **change}, T)


def test_a_source_gap_inside_scope_is_blocking():
    with pytest.raises(ValueError, match="intersects"):
        parent_review_disposition(quality(), datetime(2013,1,2,tzinfo=timezone.utc))


@pytest.mark.parametrize("status,synthetic", [("FAIL", False), ("REVIEW_REQUIRED", False), ("PASS", True)])
def test_load_release_blocks_unapproved_quality_before_any_training(tmp_path, status, synthetic):
    write_new(tmp_path/"manifest.json", {"status": status, "synthetic": synthetic,
              "allowed_purpose": "PRICE_PREDICTION_RESEARCH_ONLY"})
    with pytest.raises(ValueError, match="requires"):
        load_release(tmp_path)


@pytest.mark.parametrize("changes", [{"window": 89}, {"horizons": (7,14)}, {"embargo_days": 27},
    {"validation_start": "2023-01-02T00:00:00Z"}, {"train_end": "2023-01-01"}, {"layers": 1}])
def test_protocol_rejects_shortcuts(changes):
    with pytest.raises(ValueError):
        TrainingConfig(**changes)


def test_new_research_artifact_is_append_only(tmp_path):
    write_new(tmp_path/"record.json", {"x": 1})
    with pytest.raises(FileExistsError):
        write_new(tmp_path/"record.json", {"x": 2})
