"""Synthetic architecture checks only; these are not strategy/backtest evidence."""
import dataclasses
import json
import subprocess
import sys
import tomllib
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.domain import (
    AssetClass, BacktestResult, BacktestRun, BacktestStatus, Bar, CostAssumptions,
    Direction, EquityPoint, FillSide, Instrument, MarketObservation, Measurement,
    OptionSpec, PutCall, QualityStatus, ResearchSignal, RiskDecision, RiskStatus,
    TestSplit, TimeWindow, TradeFill,
)
from quant_platform.research.reporting import render_results

ROOT = Path(__file__).resolve().parents[1]
T = datetime(2024, 1, 1, tzinfo=timezone.utc)
D = Decimal


def crypto():
    return Instrument(instrument_id="crypto:kraken:spot:BTC-USD", asset_class=AssetClass.CRYPTO_SPOT,
                      venue="kraken", symbol="BTC/USD", quote_currency="USD")


def stock():
    return Instrument(instrument_id="ibkr:SMART:stock:SYNTHETIC-1", asset_class=AssetClass.EQUITY,
                      venue="SMART", symbol="SYNTHETIC_STOCK", quote_currency="USD")


def option_spec():
    return OptionSpec(expiration=date(2024, 2, 16), strike=D("120"), put_call=PutCall.CALL,
                      contract_multiplier=D("10"))


def option():
    return Instrument(instrument_id="ibkr:SMART:option:SYNTHETIC-2", asset_class=AssetClass.LISTED_OPTION,
                      venue="SMART", symbol="SYNTHETIC_OPTION", quote_currency="USD",
                      underlying_id=stock().instrument_id, option_spec=option_spec())


def bar():
    return Bar(instrument_id=crypto().instrument_id, source="synthetic://NOT_MARKET_DATA",
               source_symbol="XXBTZUSD", event_time_utc=T, close_time_utc=T + timedelta(days=1),
               available_at_utc=T + timedelta(days=1, seconds=60), collected_at_utc=T + timedelta(days=4),
               timeframe="1d", raw_sha256="a" * 64, clean_version="synthetic-v1",
               quality_status=QualityStatus.REVIEW_REQUIRED, availability_basis="SYNTHETIC_TEST_ONLY",
               open=D(100), high=D(110), low=D(90), close=D(105), base_volume=D(10),
               quote_volume=None, is_final=True)


def signal():
    available = bar().available_at_utc
    return ResearchSignal(strategy_id="SYNTHETIC_SCHEMA_ONLY", instrument_id=crypto().instrument_id,
                          available_at_utc=available, decision_time_utc=available,
                          earliest_action_time_utc=available + timedelta(seconds=1),
                          direction=Direction.LONG, target_exposure=D("0.1"), data_version="synthetic-v1",
                          model_version=None, reason="Schema check, no strategy claim")


def run(instrument=None):
    return BacktestRun(
        run_id="SYNTHETIC_NOT_RUN", market="synthetic-contract-check",
        instruments=(instrument or crypto(),), strategy_version="SCHEMA_ONLY", data_hashes=("a" * 64,),
        git_commit=None, engine_name="NOT_RUN_SCHEMA_FIXTURE", engine_version=None,
        cost_assumptions=CostAssumptions(fee_rate=None, half_spread_rate=None, slippage_rate=None,
                                         fee_basis="UNKNOWN", fill_policy="NOT_RUN"),
        test_split=TestSplit(train=TimeWindow(start_utc=T, end_utc=T + timedelta(days=1)),
                            validation=TimeWindow(start_utc=T + timedelta(days=1), end_utc=T + timedelta(days=2)),
                            holdout=TimeWindow(start_utc=T + timedelta(days=2), end_utc=T + timedelta(days=3)),
                            holdout_seal_id="SYNTHETIC_SEAL_ONLY"), created_at_utc=T,
    )


def test_distinct_markets_and_nonstandard_option_multiplier():
    instruments = (crypto(), stock(), option(), Instrument(
        instrument_id="ibkr:SMART:etf:SYNTHETIC-3", asset_class=AssetClass.ETF,
        venue="SMART", symbol="SYNTHETIC_ETF", quote_currency="USD"))
    assert len({i.asset_class for i in instruments}) == 4
    assert option().underlying_id == stock().instrument_id
    assert option().option_spec.contract_multiplier == D(10)
    assert option().option_spec.deliverable is None
    assert crypto().price_increment is None


@pytest.mark.parametrize("missing", ["expiration", "strike", "put_call", "contract_multiplier"])
def test_option_required_contract_fields(missing):
    values = {f.name: getattr(option_spec(), f.name) for f in dataclasses.fields(OptionSpec)}
    del values[missing]
    with pytest.raises(TypeError):
        OptionSpec(**values)


@pytest.mark.parametrize("changes", [
    {"underlying_id": None}, {"option_spec": None},
    {"underlying_id": "ibkr:SMART:option:SYNTHETIC-2"},
    {"instrument_id": "ibkr:SMART:stock:SYNTHETIC-2"},
])
def test_option_identity_rejects_incomplete_or_conflicting_metadata(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(option(), **changes)


@pytest.mark.parametrize("changes", [
    {"contract_multiplier": D(0)}, {"strike": D("NaN")},
    {"contract_multiplier": 100.0}, {"put_call": "CALL"},
])
def test_option_numeric_and_enum_validation(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(option_spec(), **changes)


@pytest.mark.parametrize("changes", [
    {"instrument_id": "BTC/USD"}, {"venue": "OTHER"},
    {"option_spec": option_spec()}, {"price_increment": D(0)},
    {"quantity_increment": 0.001},
])
def test_instrument_identity_and_decimal_validation(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(crypto(), **changes)


def test_observation_provenance_and_nullable_measurements():
    observation = bar()
    assert isinstance(observation, MarketObservation)
    assert observation.quality_status is QualityStatus.REVIEW_REQUIRED
    assert observation.quote_volume is None
    assert isinstance(observation.close, Decimal)
    with pytest.raises(dataclasses.FrozenInstanceError):
        observation.quality_status = QualityStatus.PASS


@pytest.mark.parametrize("changes", [
    {"event_time_utc": T.replace(tzinfo=None)},
    {"collected_at_utc": T.astimezone(timezone(timedelta(hours=1)))},
    {"available_at_utc": T}, {"collected_at_utc": T},
    {"raw_sha256": "bad"}, {"quality_status": "PASS"},
    {"is_final": False}, {"high": D(99)}, {"close": 105.0},
])
def test_bad_bars_rejected(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(bar(), **changes)


def test_signal_rejects_future_information_and_early_simulation():
    original = signal()
    with pytest.raises(ValueError, match="available_at"):
        dataclasses.replace(original, decision_time_utc=bar().event_time_utc)
    with pytest.raises(ValueError, match="available_at"):
        dataclasses.replace(original, earliest_action_time_utc=original.decision_time_utc - timedelta(seconds=1))
    for when in (bar().event_time_utc, original.available_at_utc, original.earliest_action_time_utc - timedelta(microseconds=1)):
        with pytest.raises(ValueError, match="precedes"):
            original.validate_action_time(when)
    original.validate_action_time(original.earliest_action_time_utc)


@pytest.mark.parametrize("changes", [{"target_exposure": 0.1}, {"confidence": D("1.1")}, {"direction": "LONG"}])
def test_signal_rejects_invalid_numeric_or_untyped_intent(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(signal(), **changes)


@pytest.mark.parametrize("status", list(RiskStatus))
def test_every_risk_status_denies_every_order_mode(status):
    decision = RiskDecision(scope="SYNTHETIC", candidate=crypto().instrument_id,
                            rule_versions=("SCHEMA_ONLY",), status=status, reasons=("Not authorized",))
    assert decision.live_order_authorized is False
    with pytest.raises(TypeError, match="not order approval"):
        bool(decision)
    for mode in ("live", "paper", "dry_run", "unknown"):
        with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
            decision.require_execution_authorization(mode)


def test_risk_rejects_approval_strings():
    with pytest.raises(ValueError):
        RiskDecision(scope="SYNTHETIC", candidate="fixture", rule_versions=("v1",),
                     status="APPROVED", reasons=("fixture",))


def test_accounting_is_decimal_and_result_instruments_are_bound():
    fill = TradeFill(instrument_id=crypto().instrument_id, time_utc=T, side=FillSide.BUY,
                     quantity=D("0.0001"), price=D("100.12345"), fee=D("0.01"), fee_currency="USD")
    assert fill.quantity * fill.price == D("0.010012345")
    with pytest.raises(ValueError):
        dataclasses.replace(fill, fee=0.01)
    with pytest.raises(ValueError, match="instrument"):
        BacktestResult(run=run(stock()), status=BacktestStatus.REVIEW_REQUIRED, trade_ledger=(fill,))
    with pytest.raises(ValueError):
        EquityPoint(time_utc=T, equity=100.0, currency="USD")


def test_split_overlap_and_missing_run_provenance_are_rejected():
    split = run().test_split
    with pytest.raises(ValueError, match="nonoverlapping"):
        dataclasses.replace(split, validation=split.train)
    with pytest.raises(ValueError):
        dataclasses.replace(run(), data_hashes=())
    with pytest.raises(ValueError):
        dataclasses.replace(run(), cost_assumptions=None)


def test_not_run_rejects_pretend_metrics_or_completed_time():
    for outputs in ({"metrics": (Measurement(name="SYNTHETIC_NOT_PERFORMANCE", value=D(0)),)},
                    {"completed_at_utc": T}):
        with pytest.raises(ValueError, match="NOT_RUN"):
            BacktestResult(run=run(), status=BacktestStatus.NOT_RUN, **outputs)
    with pytest.raises(ValueError, match="completion timestamp"):
        BacktestResult(run=run(), status=BacktestStatus.COMPLETED)


def test_cross_market_report_records_identity_engine_costs_time_and_split_without_metrics():
    payload = json.loads(render_results([
        BacktestResult(run=run(i), status=BacktestStatus.NOT_RUN, warnings=("SYNTHETIC_SCHEMA_ONLY",))
        for i in (crypto(), stock(), option())
    ]))
    assert payload["execution_authorization"] == "NOT_AUTHORIZED"
    for result in payload["results"]:
        metadata = result["run"]
        assert metadata["instruments"][0]["instrument_id"]
        assert metadata["instruments"][0]["venue"]
        assert metadata["engine_name"] == "NOT_RUN_SCHEMA_FIXTURE"
        assert metadata["data_hashes"] == ["a" * 64]
        assert metadata["created_at_utc"].endswith("+00:00")
        assert metadata["cost_assumptions"]["fee_rate"] is None
        assert set(metadata["test_split"]) == {"train", "validation", "holdout", "holdout_seal_id"}
        assert result["status"] == "NOT_RUN"
        assert result["metrics"] == result["trade_ledger"] == result["equity_curve"] == []
    assert payload["results"][2]["run"]["instruments"][0]["option_spec"]["contract_multiplier"] == "10"


def test_domain_ports_and_reporting_import_without_optional_dependencies_or_network():
    code = r'''
import importlib.abc
import os
import sys
sys.path.insert(0, sys.argv[1])
blocked = {'freqtrade', 'ibapi', 'ib_insync', 'ccxt', 'torch', 'tensorflow', 'cupy', 'pyarrow', 'quant_crypto'}
class DenyOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Forbidden import: ' + fullname)
sys.meta_path.insert(0, DenyOptional())
def audit(event, args):
    if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system'}:
        raise AssertionError('Import side effect: ' + event)
sys.addaudithook(audit)
class NoEnvironment(dict):
    def __getitem__(self, key):
        raise AssertionError('Environment read: ' + key)
    def get(self, key, default=None):
        raise AssertionError('Environment read: ' + key)
os.environ = NoEnvironment()
import quant_platform.domain
import quant_platform.ports
import quant_platform.research.reporting
assert not blocked.intersection(sys.modules)
print('ISOLATED_IMPORT_OK')
'''
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(ROOT / "src")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ISOLATED_IMPORT_OK" in result.stdout


@pytest.mark.parametrize("command", ["live", "dry-run", "paper", "submit-order", "connect-ibkr"])
def test_existing_cli_has_no_order_or_broker_route(monkeypatch, command):
    from quant_crypto.cli import main
    monkeypatch.setattr(sys, "argv", ["quant-crypto", command])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_existing_config_packaging_and_docs_only_ibkr_boundary():
    config = (ROOT / "config/research.yaml").read_text(encoding="utf-8")
    for setting in ("execution_enabled: false", "live_order_permission: NOT_GRANTED",
                    "allow_api_credentials: false", "paper_trading_enabled: false",
                    "max_loss_authorized_cny: null"):
        assert setting in config
    package = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert package["project"]["scripts"] == {"quant-crypto": "quant_crypto.cli:main"}
    assert package["project"]["dependencies"] == ["pyarrow==24.0.0", "tzdata>=2025.2"]
    ibkr = ROOT / "src/quant_platform/adapters/ibkr"
    assert (ibkr / "README.md").exists()
    assert not list(ibkr.rglob("*.py"))
    assert not (ROOT / "src/quant_platform/execution").exists()
