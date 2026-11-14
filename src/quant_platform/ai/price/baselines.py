"""Zero, same-horizon trailing momentum, and train-only ordinary least squares."""
import numpy as np


def fit_linear(train):
    coefficients, _, rank, singular = np.linalg.lstsq(train.linear, train.y, rcond=None)
    return {"coefficients": coefficients.tolist(), "rank": int(rank),
            "singular_values": singular.tolist(), "method": "ordinary_least_squares_train_only"}


def predict_baselines(samples, state):
    return {"zero": np.zeros_like(samples.y), "momentum": samples.momentum.copy(),
            "linear": samples.linear @ np.asarray(state["coefficients"])}
