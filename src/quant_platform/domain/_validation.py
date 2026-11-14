"""Small standard-library validators shared by the public contracts."""
import re
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum


def text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")


def decimal(value: Decimal, name: str, *, minimum: Decimal | None = None) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def positive(value: Decimal, name: str) -> None:
    decimal(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def member(value: Enum, kind: type[Enum]) -> None:
    if not isinstance(value, kind):
        raise ValueError(f"Expected {kind.__name__}, got {value!r}")


def sha256(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected lowercase SHA-256")


def identity(value: str) -> tuple[str, str, str, str]:
    text(value, "instrument_id")
    parts = value.split(":", 3)
    if len(parts) != 4 or any(not p or any(c.isspace() for c in p) for p in parts):
        raise ValueError("instrument_id requires namespace:venue:asset_type:local_id")
    if parts[2] not in {"spot", "stock", "etf", "option"}:
        raise ValueError("Unknown instrument asset_type")
    return tuple(parts)
