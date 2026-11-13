import re

import pytest

from quant_crypto import archive


def fake_server(blobs, *, wrong_etag=False, wrong_range=False, ignore_range=False):
    def get(url, headers=None, **kwargs):
        payload = blobs[url]
        a, b = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
        data = payload[a:b + 1]
        etag = '"changed"' if wrong_etag and headers.get("If-Match") else '"fixed"'
        content_range = f"bytes {a}-{b}/{len(payload)}"
        return data, 200 if ignore_range else 206, {"ETag": etag, "Content-Range": "bytes 1-1/100" if wrong_range else content_range}
    return get


def test_range_reader_crosses_part_boundary_and_records_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "get", fake_server({"a": b"abc", "b": b"defg"}))
    reader = archive.RemoteParts(tmp_path, ["a", "b"])
    reader.seek(2)
    assert reader.read(4) == b"cdef"
    assert reader.tell() == 6
    reader.seek(-2, 2)
    assert reader.read() == b"fg"
    assert all((tmp_path / item["raw_path"]).exists() for item in reader.receipts)


def test_range_etag_change_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "get", fake_server({"a": b"abc"}, wrong_etag=True))
    reader = archive.RemoteParts(tmp_path, ["a"])
    with pytest.raises(ValueError, match="mismatch"):
        reader.read(2)


def test_server_ignoring_range_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "get", fake_server({"a": b"abc"}, ignore_range=True))
    with pytest.raises(ValueError, match="exact ranges"):
        archive.RemoteParts(tmp_path, ["a"])


def test_invalid_content_range_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "get", fake_server({"a": b"abc"}, wrong_range=True))
    with pytest.raises(ValueError):
        archive.RemoteParts(tmp_path, ["a"])
