"""Portable JSON report, without calculating or inventing backtest measurements."""
import json
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Sequence

from quant_platform.domain import BacktestResult


def _encode(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Unsupported report type: {type(value).__name__}")


def render_results(results: Sequence[BacktestResult]) -> str:
    """Each run retains its instruments/venue, engine, hashes, costs and split."""
    if not all(isinstance(result, BacktestResult) for result in results):
        raise ValueError("Reports require BacktestResult objects")
    payload = {
        "schema_version": "multi-asset-research-v1", "research_only": True,
        "execution_authorization": "NOT_AUTHORIZED",
        "results": [asdict(result) for result in results],
    }
    return json.dumps(payload, default=_encode, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
