import argparse
import json
from pathlib import Path

from .archive import fetch_daily
from .pipeline import import_file


def main():
    parser = argparse.ArgumentParser(description="RESEARCH ONLY: public archives/offline ingestion")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch-kraken-daily")
    importer = commands.add_parser("import")
    importer.add_argument("path", type=Path)
    importer.add_argument("--symbol", choices=["XBTUSD", "ETHUSD"], required=True)
    importer.add_argument("--source-url", required=True)
    importer.add_argument("--collected-at", required=True)
    importer.add_argument("--expected-sha256")
    importer.add_argument("--member")
    importer.add_argument("--as-of")
    importer.add_argument("--provenance")
    importer.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.command == "fetch-kraken-daily":
        path, receipt = fetch_daily(args.root)
        print(json.dumps({"download_manifest": str(path), "members": receipt["members"], "downloaded_bytes": receipt["downloaded_bytes"]}, indent=2))
    else:
        manifest = import_file(args.path, args.root, symbol=args.symbol, source_url=args.source_url,
                               collected_at=args.collected_at, expected_sha256=args.expected_sha256,
                               member=args.member, as_of=args.as_of, provenance=args.provenance, synthetic=args.synthetic)
        print(json.dumps(manifest["quality"], indent=2))
        return 0 if manifest["quality"]["status"] == "PASS" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
