"""Prediction errors in log-return units, never portfolio/profitability metrics."""
import numpy as np


def errors(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError("Expected matching [N,2] targets")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite evaluation output")
    return {str(h): {"n": len(truth), "mae": float(np.abs(truth[:, i]-prediction[:, i]).mean()),
                     "rmse": float(np.sqrt(np.square(truth[:, i]-prediction[:, i]).mean())),
                     "direction_accuracy": float((np.sign(truth[:, i]) == np.sign(prediction[:, i])).mean())}
            for i, h in enumerate((14, 28))} if len(truth) else None


def report_metrics(samples, predictions, volatility_threshold):
    groups = {"all": np.ones(len(samples.y), dtype=bool)}
    for symbol in ("BTC", "ETH"):
        groups[symbol] = np.array([m["symbol"] == symbol for m in samples.metadata])
    for direction in ("up", "down"):
        for vol in ("high_vol", "low_vol"):
            groups[f"{direction}_{vol}"] = np.array([
                (m["trailing_28d_return"] >= 0) == (direction == "up")
                and (m["trailing_28d_volatility"] >= volatility_threshold) == (vol == "high_vol")
                for m in samples.metadata])
    return {"metrics": {model: {group: errors(samples.y[mask], values[mask]) for group, mask in groups.items()}
                         for model, values in predictions.items()},
            "regime_rule": "Trailing 28d log-return sign x trailing 28d volatility; threshold median of train samples only",
            "volatility_threshold_train_only": float(volatility_threshold),
            "direction_rule": "sign(pred)==sign(actual); zero is flat, not an up/down call",
            "warnings": ["Overlapping daily 14/28d labels and correlated BTC/ETH observations are not independent",
                         "Prediction error is not a trading return; no fees, fills or profitability estimated"]}
