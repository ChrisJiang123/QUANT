"""Replay fetched daily files using the retained download receipt (no network)."""
import argparse
import json
from pathlib import Path

from quant_crypto.pipeline import import_file
from quant_crypto.storage import verify

parser = argparse.ArgumentParser()
parser.add_argument("receipt", type=Path)
parser.add_argument("--root", type=Path, default=Path.cwd())
args = parser.parse_args()
root = args.root.resolve()
receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
for item in receipt["requests"] + receipt["members"]:
    verify(root / item["raw_path"], item["sha256"])
exit_code = 0
for item in receipt["members"]:
    if "symbol" not in item:
        continue
    manifest = import_file(root / item["raw_path"], root, symbol=item["symbol"],
                           source_url=receipt["urls"][0] + "#zip-member=" + item["member"],
                           collected_at=item["downloaded_at_utc"], expected_sha256=item["sha256"],
                           provenance=args.receipt.resolve().relative_to(root).as_posix())
    print(json.dumps(manifest["quality"], indent=2, sort_keys=True))
    if manifest["quality"]["status"] != "PASS":
        exit_code = 2
raise SystemExit(exit_code)
