import pytest

from quant_crypto import public_http


@pytest.mark.parametrize("url,method,headers", [
    ("https://api.kraken.com/0/private/Balance", "GET", {}),
    ("https://api.kraken.com/0/private/AddOrder", "POST", {}),
    ("https://api.kraken.com/0/private/CancelOrder", "GET", {}),
    ("https://api.kraken.com/0/private/Withdraw", "POST", {}),
    ("https://api.kraken.com/0/public/OHLC", "GET", {"API-Key": "SYNTHETIC"}),
    ("https://api.kraken.com/0/public/OHLC?api_key=SYNTHETIC", "GET", {}),
    ("https://user:pass@api.kraken.com/0/public/Time", "GET", {}),
    ("http://api.kraken.com/0/public/Time", "GET", {}),
    ("https://api.kraken.com.evil.test/0/public/Time", "GET", {}),
    ("https://api.kraken.com:444/0/public/Time", "GET", {}),
    ("https://api.kraken.com/0/public/%2e%2e/private/AddOrder", "GET", {}),
])
def test_forbidden_request_rejected_before_network(monkeypatch, url, method, headers):
    def must_not_open(*args, **kwargs):
        pytest.fail("Network must not be reached")
    monkeypatch.setattr(public_http.urllib.request, "build_opener", must_not_open)
    with pytest.raises(PermissionError):
        if method == "GET":
            public_http.get(url, headers)
        else:
            public_http.authorize(url, method, headers)


def test_redirect_to_private_endpoint_is_blocked():
    redirect = public_http.CheckedRedirect()
    with pytest.raises(PermissionError):
        redirect.redirect_request(None, None, 302, "Found", {}, "https://api.kraken.com/0/private/Balance")


def test_public_archive_allowlist():
    public_http.authorize("https://assets.kraken.com/marketing/institutions/Kraken_OHLCVT_Full.zip.part04")
