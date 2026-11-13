"""One fixed research experiment; functional smoke runs never consume the seal."""
import json
import os
from pathlib import Path
import platform
import random

import numpy as np

from .artifacts import code_identity, digest, file_hash, git_commit, now, verify, write_new
from .audit import load_release
from .baselines import fit_linear, predict_baselines
from .config import TrainingConfig
from .dataset import FEATURES, Normalizer, build_samples, load_development
from .metrics import report_metrics


def runtime(seed, functional=False, device="cuda"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if not functional and device != "cuda":
        raise ValueError("Formal training/evaluation requires the RTX 4090 worker")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    if not functional and "RTX 4090" not in (gpu or ""):
        raise ValueError("Formal runs require verified RTX 4090, not the controller GPU")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return torch, {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                   "cuda_build": torch.version.cuda, "device": device, "gpu": gpu,
                   "deterministic_algorithms": True, "tf32": False,
                   "reproducibility_limit": "Same seed is not a cross-device/version bitwise guarantee"}


def tensor_data(samples, scaler, torch):
    return torch.utils.data.TensorDataset(torch.from_numpy(scaler.x(samples.x)),
                torch.from_numpy(samples.asset), torch.from_numpy(scaler.y(samples.y)))


def predict(model, samples, scaler, torch, device, batch_size):
    model.eval()
    outputs = []
    loader = torch.utils.data.DataLoader(tensor_data(samples, scaler, torch), batch_size=batch_size,
                                         shuffle=False, num_workers=0)
    with torch.no_grad():
        for x, asset, _ in loader:
            outputs.append(model(x.to(device), asset.to(device)).cpu().numpy())
    return scaler.inverse_y(np.concatenate(outputs))


def save_predictions(path, samples, predictions):
    with Path(path).open("x", encoding="utf-8") as stream:
        for index, metadata in enumerate(samples.metadata):
            stream.write(json.dumps({**metadata, "actual": samples.y[index].tolist(),
                         "predictions": {k: v[index].tolist() for k, v in predictions.items()}}, allow_nan=False) + "\n")


def fit(release_directory, output, *, functional=False, device="cuda", root=None):
    release_directory, output = Path(release_directory), Path(output)
    release = load_release(release_directory)  # FAIL/REVIEW blocks before importing torch or fitting anything.
    config = TrainingConfig.from_dict(release["recipe"]["config"])
    torch, environment = runtime(config.seed, functional, device)
    from .model import PriceReturnModel

    if not functional:
        write_new(release_directory / "development.claim.json", {"protocol": config.protocol,
                  "config_hash": digest(config.to_dict()), "output": str(output.resolve()), "started_at": now()})
    output.mkdir(parents=True, exist_ok=False)
    attempt = {"status": "RUNNING", "functional_only": functional, "release_id": release["release_id"],
               "data_hashes": release["files"], "config": config.to_dict(), "config_sha256": digest(config.to_dict()),
               "code": code_identity(), "git_commit": git_commit(root or Path.cwd()), "environment": environment,
               "started_at": now(), "features": FEATURES, "no_orders": True}
    write_new(output / "attempt.json", attempt)
    _, train, validation = load_development(release_directory, config)
    if functional:
        # Fixed first samples per asset; cannot freeze or evaluate the final test.
        train, validation = train.subset(128), validation.subset(64)
    scaler = Normalizer.fit(train)
    write_new(output / "normalizer.json", scaler.to_dict())
    linear_state = fit_linear(train)
    write_new(output / "linear.json", linear_state)
    volatility_threshold = float(np.median([m["trailing_28d_volatility"] for m in train.metadata]))
    model = PriceReturnModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    loader = torch.utils.data.DataLoader(tensor_data(train, scaler, torch), batch_size=config.batch_size,
                                         shuffle=True, generator=generator, num_workers=0)
    best, best_epoch, stale = float("inf"), 0, 0
    epochs = min(2, config.epochs) if functional else config.epochs
    log = output / "training.jsonl"
    with log.open("x", encoding="utf-8") as stream:
        for epoch in range(1, epochs+1):
            model.train()
            total = 0.0
            for x, asset, y in loader:
                optimizer.zero_grad(set_to_none=True)
                predicted = model(x.to(device), asset.to(device))
                loss = torch.nn.functional.mse_loss(predicted, y.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach().cpu()) * len(x)
            val_predictions = predict(model, validation, scaler, torch, device, config.batch_size)
            val_loss = float(np.square((val_predictions-validation.y)/scaler.y_std).mean())
            entry = {"epoch": epoch, "train_standardized_mse": total/len(train.y), "validation_standardized_mse": val_loss}
            stream.write(json.dumps(entry, allow_nan=False)+"\n")
            stream.flush()
            print(json.dumps(entry), flush=True)
            if val_loss < best:
                best, best_epoch, stale = val_loss, epoch, 0
                # Mutable checkpoint only during validation. Hash is sealed after selection.
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output / "model.pt")
            else:
                stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(torch.load(output / "model.pt", map_location=device, weights_only=True))
    predictions = {"transformer": predict(model, validation, scaler, torch, device, config.batch_size),
                   **predict_baselines(validation, linear_state)}
    metrics = report_metrics(validation, predictions, volatility_threshold)
    write_new(output / "validation_metrics.json", metrics)
    save_predictions(output / "validation_predictions.jsonl", validation, predictions)
    result = {"status": "FUNCTIONAL_SMOKE_ONLY" if functional else "VALIDATION_COMPLETED_FROZEN",
              "best_epoch": best_epoch, "epochs_run": epoch, "train_samples": len(train.y),
              "validation_samples": len(validation.y), "parameter_count": sum(p.numel() for p in model.parameters()),
              "environment": environment, "completed_at": now(), "final_test_accessed": False}
    write_new(output / "result.json", result)
    if not functional:
        if code_identity() != attempt["code"]:
            raise ValueError("Source changed during the experiment; cannot freeze")
        frozen_files = ("attempt.json", "model.pt", "normalizer.json", "linear.json", "validation_metrics.json",
                        "validation_predictions.jsonl", "training.jsonl", "result.json")
        write_new(output / "frozen.json", {"release_id": release["release_id"], "seal_id": release["seal_id"],
                  "config_sha256": digest(config.to_dict()), "code": code_identity(),
                  "files": {name: file_hash(output/name) for name in frozen_files},
                  "volatility_threshold_train_only": volatility_threshold, "frozen_at": now()})
    return result


def write_worker_report(output, destination):
    """Only called after real worker validation; never manufactures missing metrics."""
    output, destination = Path(output), Path(destination)
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    attempt = json.loads((output / "attempt.json").read_text(encoding="utf-8"))
    lines = ["# TRAINING REPORT — actual worker output", "",
             f"Status: {result['status']}; GPU: {result['environment']['gpu']}",
             f"Train/validation samples: {result['train_samples']} / {result['validation_samples']}",
             f"Epochs run / selected: {result['epochs_run']} / {result['best_epoch']}",
             f"Seed: {attempt['config']['seed']}; Git commit: {attempt['git_commit']}",
             f"Code SHA256: {attempt['code']['sha256']}; release: {attempt['release_id']}", "",
             "No orders or profitability metrics. Errors below are log-return prediction errors.", ""]
    for phase, name in (("Validation", "validation_metrics.json"), ("Final (one evaluation)", "final_metrics.json")):
        path = output / name
        lines.extend([f"## {phase}", ""])
        if not path.exists():
            lines.extend(["NOT RUN; final dataset remains sealed unless a prior failed claim exists.", ""])
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        lines.extend(["| Model | Group | Horizon | n | MAE | RMSE | Direction accuracy |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for model, groups in metrics["metrics"].items():
            for group, horizons in groups.items():
                if horizons:
                    for horizon, m in horizons.items():
                        lines.append(f"| {model} | {group} | {horizon} | {m['n']} | {m['mae']:.8f} | {m['rmse']:.8f} | {m['direction_accuracy']:.4f} |")
        lines.extend(["", "Regime threshold is fitted on training observations only.", ""])
    lines.extend(["## Limits", "", "Single seed/configuration; overlapping daily labels and correlated assets are not independent.",
                  "Parent TASK001 REVIEW_REQUIRED is retained; scoped research release passed a separate documented audit.",
                  "No October archive; test scope shortened before performance inspection to complete 28-day labels.",
                  "Historical availability is inferred; normalization cannot recover precision lost in upstream float64.",
                  "No Sentiment/GAT/RL training; no transaction-cost or strategy-profitability estimate.",
                  "Source/account/risk permissions do not change. No subsequent stage starts.", ""])
    with destination.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def verify_frozen(output, release, config):
    output = Path(output)
    frozen = json.loads((output / "frozen.json").read_text(encoding="utf-8"))
    if (frozen["release_id"] != release["release_id"] or frozen["seal_id"] != release["seal_id"]
            or frozen["config_sha256"] != digest(config.to_dict()) or frozen["code"] != code_identity()):
        raise ValueError("Frozen model/config/source identity mismatch")
    for name, expected in frozen["files"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe frozen artifact name")
        verify(output/name, expected)
    return frozen


def evaluate_final_once(release_directory, output, *, device="cuda"):
    release_directory, output = Path(release_directory), Path(output)
    release = load_release(release_directory)
    config = TrainingConfig.from_dict(release["recipe"]["config"])
    frozen = verify_frozen(output, release, config)
    torch, environment = runtime(config.seed, False, device)
    # Claim is shared by every model/output using this release; it is created
    # BEFORE opening test prices. A crash keeps it consumed; no auto retry/reset.
    claim_final(release_directory, release["seal_id"], file_hash(output / "frozen.json"))
    path = release_directory / "sealed.json"
    verify(path, release["files"]["sealed.json"])
    test = build_samples(json.loads(path.read_text(encoding="utf-8")), config, "test")
    scaler = Normalizer.from_dict(json.loads((output / "normalizer.json").read_text(encoding="utf-8")))
    linear = json.loads((output / "linear.json").read_text(encoding="utf-8"))
    from .model import PriceReturnModel
    model = PriceReturnModel(config).to(device)
    model.load_state_dict(torch.load(output / "model.pt", map_location=device, weights_only=True))
    predictions = {"transformer": predict(model, test, scaler, torch, device, config.batch_size), **predict_baselines(test, linear)}
    metrics = report_metrics(test, predictions, frozen["volatility_threshold_train_only"])
    write_new(output / "final_metrics.json", metrics)
    save_predictions(output / "final_predictions.jsonl", test, predictions)
    result = {"status": "FINAL_EVALUATED_ONCE", "samples": len(test.y), "environment": environment,
              "first_decision": min(m["decision_time_utc"] for m in test.metadata),
              "last_decision": max(m["decision_time_utc"] for m in test.metadata), "completed_at": now(),
              "frozen_sha256": file_hash(output / "frozen.json"), "final_metrics_sha256": file_hash(output / "final_metrics.json")}
    write_new(output / "final_receipt.json", result)
    return result


def claim_final(directory, seal_id, frozen_hash):
    """No retry or overwrite flag. An interrupted evaluation remains consumed."""
    write_new(Path(directory) / "final.claim.json", {"seal_id": seal_id,
              "frozen_sha256": frozen_hash, "claimed_at": now(), "policy": "ONCE_NO_AUTOMATIC_RETRY"})


def evaluate_development_baselines(release_directory, output):
    """Controller-side baseline research; never accesses the sealed file or GPU."""
    release_directory, output = Path(release_directory), Path(output)
    release = load_release(release_directory)
    config = TrainingConfig.from_dict(release["recipe"]["config"])
    _, train, validation = load_development(release_directory, config)
    output.mkdir(parents=True, exist_ok=False)
    state = fit_linear(train)
    threshold = float(np.median([m["trailing_28d_volatility"] for m in train.metadata]))
    predictions = predict_baselines(validation, state)
    write_new(output / "linear.json", state)
    write_new(output / "validation_metrics.json", report_metrics(validation, predictions, threshold))
    save_predictions(output / "validation_predictions.jsonl", validation, predictions)
    result = {"status": "DEVELOPMENT_BASELINES_COMPLETED", "release_id": release["release_id"],
              "data_hashes": release["files"], "config": config.to_dict(), "config_sha256": digest(config.to_dict()),
              "code": code_identity(), "git_commit": git_commit(Path.cwd()), "seed": config.seed,
              "train_samples": len(train.y), "validation_samples": len(validation.y),
              "numpy": np.__version__, "final_test_accessed": False, "completed_at": now()}
    write_new(output / "result.json", result)
    return result
