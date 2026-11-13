"""Only anonymous public GETs to narrowly allowlisted official resources."""
import re
import urllib.request
from urllib.parse import urlsplit

ARCHIVE_PATH = re.compile(r"/marketing/institutions/(?:Kraken_OHLCVT_(?:Full\.zip\.part0[0-4]|2026Q[1-3]\.zip)|OHLCVT_Full_PARTS_SHA256SUMS\.txt)$")


def authorize(url: str, method: str = "GET", headers=None) -> None:
    p = urlsplit(url)
    if method != "GET" or p.scheme != "https" or p.username or p.password or p.fragment or p.port not in (None, 443):
        raise PermissionError("Research-only anonymous HTTPS GET required")
    headers = headers or {}
    if set(k.lower() for k in headers) - {"range", "if-match"}:
        raise PermissionError("Credentials/custom headers forbidden")
    archive = p.hostname == "assets.kraken.com" and ARCHIVE_PATH.fullmatch(p.path) and not p.query
    api = p.hostname == "api.kraken.com" and p.path in {"/0/public/Time", "/0/public/AssetPairs", "/0/public/OHLC"} and not p.query
    if not (archive or api):
        raise PermissionError("URL outside public research allowlist")


class CheckedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        authorize(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def get(url: str, headers=None, limit=16 * 1024 * 1024):
    authorize(url, headers=headers)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 QuantCryptoResearch/0.1", **(headers or {})}, method="GET")
    with urllib.request.build_opener(CheckedRedirect()).open(request, timeout=45) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Response exceeds bounded download size")
        return data, response.status, dict(response.headers)
