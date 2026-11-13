"""One preregistered configuration; UTC boundaries refer to decision availability."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


def utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError("Explicit UTC required")
    return result


@dataclass(frozen=True)
class TrainingConfig:
    protocol: str = "kraken-price-transformer-v1"
    seed: int = 42
    window: int = 90
    horizons: tuple[int, int] = (14, 28)
    d_model: int = 64
    layers: int = 2
    heads: int = 4
    feedforward: int = 128
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 20
    patience: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    embargo_days: int = 28
    train_start: str = "2019-01-01T00:00:00Z"
    train_end: str = "2023-01-01T00:00:00Z"
    validation_start: str = "2023-01-29T00:00:00Z"
    validation_end: str = "2025-01-01T00:00:00Z"
    test_start: str = "2025-01-29T00:00:00Z"
    # Fixed before any performance inspection: no October archive is available.
    # Last eligible anchor is Sep 2 open / Sep 3 close+lag, target Sep 30.
    test_end: str = "2026-09-04T00:00:00Z"

    def __post_init__(self):
        if self.window != 90 or tuple(self.horizons) != (14, 28) or self.embargo_days < 28:
            raise ValueError("Protocol fixes 90 days, 14/28 targets and >=28-day embargo")
        if self.d_model not in (64, 128) or not 2 <= self.layers <= 4 or self.d_model % self.heads:
            raise ValueError("Invalid lightweight Transformer dimensions")
        if not (0 <= self.dropout < 1 and self.learning_rate > 0 and self.weight_decay >= 0):
            raise ValueError("Invalid optimizer/dropout settings")
        if min(self.batch_size, self.epochs, self.patience, self.feedforward) <= 0:
            raise ValueError("Positive training settings required")
        times = [utc(getattr(self, n)) for n in ("train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end")]
        if not all(a < b for a, b in zip(times, times[1:])):
            raise ValueError("Splits must be strictly chronological")
        for a, b in ((times[1], times[2]), (times[3], times[4])):
            if (b - a).days < self.embargo_days:
                raise ValueError("Split embargo too short")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        if "horizons" in data:
            data["horizons"] = tuple(data["horizons"])
        return cls(**data)
