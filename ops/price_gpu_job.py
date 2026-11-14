"""Fixed one-off GPU job, not a shell agent, service or second scheduler."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description="Fixed RTX4090 price-prediction job; no orders")
    parser.add_argument("--evaluate-final", action="store_true", help="Consume sealed final evaluation exactly once after freeze")
    args = parser.parse_args()
    job = json.loads((ROOT / "job.json").read_text(encoding="utf-8"))
    release = (ROOT / job["release_path"]).resolve()
    release.relative_to(ROOT)
    from quant_platform.ai.price.train import evaluate_final_once, fit, write_worker_report
    output = ROOT / "results/price-v1-seed42"
    try:
        fit(release, output, root=ROOT)
        if args.evaluate_final:
            evaluate_final_once(release, output)
        write_worker_report(output, ROOT / "TRAINING_REPORT.md")
        print("Job finished; no subsequent phase will start.")
        return 0
    except Exception as exc:
        # Retain interruption evidence. Do not clear claims or automatically retry.
        from quant_platform.ai.price.artifacts import now, write_new
        path = ROOT / "job_failure.json"
        if not path.exists():
            write_new(path, {"status": "FAILED_NO_AUTOMATIC_RETRY", "exception": type(exc).__name__,
                            "message": str(exc), "at": now()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
