"""Fetch only daily members from Kraken's official concatenated ZIP via Range.

Preserves exact response bytes and provenance; validates ZIP CRC. Does NOT claim
to verify the publisher's SHA256 of the entire 9.5GB archive without downloading it.
"""
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .public_http import get
from .storage import canonical, immutable_write, put_raw, sha256

BASE = "https://assets.kraken.com/marketing/institutions/"
PARTS = [BASE + f"Kraken_OHLCVT_Full.zip.part{i:02}" for i in range(5)]


def now():
    return datetime.now(timezone.utc).isoformat()


class RemoteParts(io.RawIOBase):
    def __init__(self, root: Path, urls=PARTS):
        self.root, self.urls, self.pos, self.parts, self.receipts = root, urls, 0, [], []
        for url in urls:
            data, status, headers = get(url, {"Range": "bytes=0-0"})
            match = re.fullmatch(r"bytes 0-0/(\d+)", headers.get("Content-Range", ""))
            if status != 206 or not match or len(data) != 1 or not headers.get("ETag"):
                raise ValueError("Server must support exact ranges and ETag")
            self.parts.append((int(match[1]), headers["ETag"]))
            self.record(url, data, headers)
        self.size = sum(p[0] for p in self.parts)

    def record(self, url, data, headers):
        path = put_raw(self.root, data, ".bin")
        self.receipts.append({"url": url, "range": headers.get("Content-Range"), "etag": headers.get("ETag"), "last_modified": headers.get("Last-Modified"), "downloaded_at_utc": now(), "sha256": sha256(data), "raw_path": path.relative_to(self.root).as_posix()})

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.pos = position
        return position

    def read(self, size=-1):
        size = min(self.size - self.pos, size if size >= 0 else self.size - self.pos)
        if size <= 0:
            return b""
        if size > 16 * 1024 * 1024:
            raise ValueError("Unexpectedly large archive member/directory")
        result, base = [], 0
        for url, (length, etag) in zip(self.urls, self.parts):
            if self.pos < base + length and size:
                start = self.pos - base
                end = min(length - 1, start + size - 1)
                data, status, headers = get(url, {"Range": f"bytes={start}-{end}", "If-Match": etag})
                if status != 206 or headers.get("Content-Range") != f"bytes {start}-{end}/{length}" or len(data) != end - start + 1 or headers.get("ETag") != etag:
                    raise ValueError("Range/ETag/length mismatch")
                self.record(url, data, headers)
                result.append(data)
                self.pos += len(data)
                size -= len(data)
            base += length
        return b"".join(result)


def fetch_daily(root: Path):
    root = root.resolve()
    remote = RemoteParts(root)
    members = []
    with zipfile.ZipFile(remote) as archive:
        for symbol in ("XBTUSD", "ETHUSD"):
            matches = [n for n in archive.namelist() if Path(n).name == f"{symbol}_1440.csv"]
            if len(matches) != 1:
                raise ValueError(f"Expected one daily member: {symbol}; found {matches}")
            name = matches[0]
            info = archive.getinfo(name)
            if info.file_size > 16 * 1024 * 1024:
                raise ValueError("Daily CSV exceeds safety bound")
            data = archive.read(name)  # zipfile verifies member CRC32.
            path = put_raw(root, data, ".csv")
            members.append({"member": name, "symbol": symbol, "sha256": sha256(data), "crc32": f"{info.CRC:08x}", "raw_path": path.relative_to(root).as_posix(), "downloaded_at_utc": now()})
        for name in archive.namelist():
            if Path(name).name == "MANIFEST.json":
                data = archive.read(name)
                path = put_raw(root, data, ".json")
                members.append({"member": name, "sha256": sha256(data), "raw_path": path.relative_to(root).as_posix()})
    sums, _, _ = get(BASE + "OHLCVT_Full_PARTS_SHA256SUMS.txt")
    checksum_path = put_raw(root, sums, ".txt")
    receipt = {"source_kind": "official_kraken_archive_range_extraction", "support_url": "https://support.kraken.com/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data", "urls": PARTS, "archive_total_bytes": remote.size, "downloaded_bytes": sum((root / r['raw_path']).stat().st_size for r in remote.receipts), "members": members, "requests": remote.receipts, "published_part_checksums_path": checksum_path.relative_to(root).as_posix(), "integrity": "HTTPS + stable ETag + exact Content-Range + ZIP CRC32 + local SHA256; full publisher archive/part SHA256 NOT verified"}
    path = root / "data/manifests" / f"download-{sha256(canonical(receipt))}.json"
    immutable_write(path, canonical(receipt))
    return path, receipt
