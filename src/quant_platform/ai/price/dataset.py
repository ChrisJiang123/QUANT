"""Causal OHLCV features and purged, embargoed multi-horizon sample indexes."""
from dataclasses import dataclass
from datetime import timedelta
import json
from pathlib import Path

import numpy as np

from .artifacts import digest, verify
from .audit import load_release
from .config import utc

SYMBOLS = ("BTC", "ETH")
FEATURES = ("log_open_prev_close", "log_high_prev_close", "log_low_prev_close", "log_close_prev_close", "log1p_base_volume")


@dataclass
class Samples:
    x: np.ndarray
    y: np.ndarray
    asset: np.ndarray
    linear: np.ndarray
    momentum: np.ndarray
    metadata: list[dict]

    def subset(self, count):
        # Fixed earliest rows per asset, so functional smoke exercises both symbols.
        indices = np.concatenate([np.flatnonzero(self.asset == a)[:count//2] for a in (0, 1)])
        return Samples(self.x[indices], self.y[indices], self.asset[indices], self.linear[indices],
                       self.momentum[indices], [self.metadata[int(i)] for i in indices])


def build_samples(rows_by_asset, config, split):
    boundaries = {"train": (config.train_start, config.train_end),
                  "validation": (config.validation_start, config.validation_end),
                  "test": (config.test_start, config.test_end)}
    start, end = map(utc, boundaries[split])
    xs, ys, assets, linear, momentum, metadata = [], [], [], [], [], []
    for asset, symbol in enumerate(SYMBOLS):
        rows = rows_by_asset[symbol]
        times = [utc(r["event_time_utc"]) for r in rows]
        available = [utc(r["available_at_utc"]) for r in rows]
        if any(b - a != timedelta(days=1) for a, b in zip(times, times[1:])):
            raise ValueError("No interpolation: sample input must be daily and contiguous")
        if any(a < t + timedelta(days=1) for t, a in zip(times, available)):
            raise ValueError("Unfinished / premature-availability input")
        prices = np.array([[float(r[n]) for n in ("open", "high", "low", "close", "base_volume")] for r in rows], dtype=np.float64)
        if not np.isfinite(prices).all() or np.any(prices[:, :4] <= 0) or np.any(prices[:, 4] < 0):
            raise ValueError("Invalid OHLCV")
        closes = prices[:, 3]
        features = np.zeros((len(rows), 5), dtype=np.float64)
        features[1:, :4] = np.log(prices[1:, :4] / closes[:-1, None])
        features[:, 4] = np.log1p(prices[:, 4])
        for i in range(config.window, len(rows) - max(config.horizons)):
            decision = available[i]
            label_times = [available[i+h] for h in config.horizons]
            # Purge: both labels must be fully known strictly before split end.
            # Test upper decision boundary is distinct from the available label end.
            if not start <= decision < end or (split != "test" and max(label_times) >= end):
                continue
            window_start = i - config.window + 1
            if max(available[window_start:i+1]) > decision:
                raise ValueError("Future feature availability")
            trailing = [float(np.log(closes[i] / closes[i-h])) for h in config.horizons]
            volatility = float(np.std(features[i-27:i+1, 3], ddof=0))
            xs.append(features[window_start:i+1])
            ys.append([float(np.log(closes[i+h] / closes[i])) for h in config.horizons])
            assets.append(asset)
            linear.append([1.0, *trailing, volatility, features[i, 4], float(asset)])
            momentum.append(trailing)
            metadata.append({"symbol": symbol, "decision_time_utc": decision.isoformat(),
                             "feature_first_event_utc": times[window_start].isoformat(),
                             "feature_last_available_utc": available[i].isoformat(),
                             "label_available_utc": [v.isoformat() for v in label_times],
                             "trailing_28d_return": trailing[1], "trailing_28d_volatility": volatility})
    if not xs:
        raise ValueError(f"Empty {split} samples")
    # Same ordering and frozen mapping on both machines; no random time split.
    return Samples(np.asarray(xs), np.asarray(ys), np.asarray(assets, dtype=np.int64),
                   np.asarray(linear), np.asarray(momentum), metadata)


def load_development(directory, config):
    manifest = load_release(directory)
    if digest(config.to_dict()) != digest(manifest["recipe"]["config"]):
        raise ValueError("Config differs from preregistered data release")
    path = Path(directory) / "development.json"
    verify(path, manifest["files"]["development.json"])
    rows = json.loads(path.read_text(encoding="utf-8"))
    if any(utc(r["event_time_utc"]) >= utc(config.validation_end) for data in rows.values() for r in data):
        raise ValueError("Sealed observations leaked into development file")
    return manifest, build_samples(rows, config, "train"), build_samples(rows, config, "validation")


@dataclass
class Normalizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    @classmethod
    def fit(cls, train):
        flat = train.x.reshape(-1, train.x.shape[-1])
        return cls(flat.mean(0), np.maximum(flat.std(0), 1e-8), train.y.mean(0), np.maximum(train.y.std(0), 1e-8))

    def x(self, values):
        return ((values-self.x_mean)/self.x_std).astype(np.float32)

    def y(self, values):
        return ((values-self.y_mean)/self.y_std).astype(np.float32)

    def inverse_y(self, values):
        return np.asarray(values)*self.y_std+self.y_mean

    def to_dict(self):
        return {name: getattr(self, name).tolist() for name in ("x_mean", "x_std", "y_mean", "y_std")}

    @classmethod
    def from_dict(cls, value):
        return cls(**{k: np.array(v) for k, v in value.items()})
