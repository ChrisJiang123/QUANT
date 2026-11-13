"""Explicit local CLI; no server, broker, generic shell or credential interface."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RESEARCH ONLY: price-return prediction, no orders")
    commands = parser.add_subparsers(dest="action", required=True)
    audit = commands.add_parser("audit-data")
    audit.add_argument("--root", type=Path, default=Path.cwd())
    for name in ("validate", "functional-smoke", "baselines", "evaluate-final-once"):
        sub = commands.add_parser(name)
        sub.add_argument("--release", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.action == "audit-data":
        from .audit import create_release
        result = {"release": str(create_release(args.root))}
    else:
        from .train import evaluate_final_once, evaluate_development_baselines, fit
        if args.action == "evaluate-final-once":
            result = evaluate_final_once(args.release, args.output, device=args.device)
        elif args.action == "baselines":
            result = evaluate_development_baselines(args.release, args.output)
        else:
            result = fit(args.release, args.output, functional=args.action == "functional-smoke", device=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
