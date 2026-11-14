"""Causality, baselines, metrics, model gradients, freeze and one-use seal tests."""
import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="Run in the isolated .venv-training environment")

from quant_platform.ai.price.artifacts import digest, write_new
from quant_platform.ai.price.baselines import fit_linear, predict_baselines
from quant_platform.ai.price.config import TrainingConfig, utc
from quant_platform.ai.price.dataset import Normalizer, Samples, build_samples
from quant_platform.ai.price.metrics import errors, report_metrics
from quant_platform.ai.price.train import claim_final


def rows(start=datetime(2019,1,1,tzinfo=timezone.utc), count=2192):
    data = []
    for index in range(count):
        day = start + timedelta(days=index)
        close = 100 * np.exp(0.0002*index + 0.05*np.sin(index/30))
        data.append({"event_time_utc": day.isoformat(), "available_at_utc": (day+timedelta(days=1, seconds=60)).isoformat(),
                     "open": close*.999, "high": close*1.01, "low": close*.99, "close": close,
                     "base_volume": 100+index%31})
    return {"BTC": data, "ETH": [dict(row) for row in data]}


@pytest.fixture(scope="module")
def prepared():
    data=rows(); config=TrainingConfig()
    return data,config,build_samples(data,config,"train"),build_samples(data,config,"validation")


def test_targets_are_log_returns_at_both_exact_calendar_horizons(prepared):
    data,config,train,_=prepared
    metadata=train.metadata[0]
    anchor=next(i for i,r in enumerate(data["BTC"]) if r["available_at_utc"]==metadata["decision_time_utc"])
    expected=[np.log(data["BTC"][anchor+h]["close"]/data["BTC"][anchor]["close"]) for h in (14,28)]
    np.testing.assert_allclose(train.y[0],expected)
    assert train.x.shape[1:]==(90,5)


def test_labels_are_purged_and_decisions_embargoed_at_every_boundary(prepared):
    _,config,train,validation=prepared
    for split,end,start in ((train,utc(config.train_end),utc(config.train_start)),
                            (validation,utc(config.validation_end),utc(config.validation_start))):
        for item in split.metadata:
            assert start <= utc(item["decision_time_utc"]) < end
            assert max(map(utc,item["label_available_utc"])) < end
            assert utc(item["feature_last_available_utc"]) <= utc(item["decision_time_utc"])
    assert min(utc(m["decision_time_utc"]) for m in validation.metadata)-utc(config.train_end)>=timedelta(days=28)
    assert max(utc(m["label_available_utc"][-1]) for m in train.metadata)<min(utc(m["decision_time_utc"]) for m in validation.metadata)


def test_modifying_future_prices_changes_labels_but_never_past_inputs(prepared):
    data,config,train,_=prepared
    changed={s:[dict(r) for r in values] for s,values in data.items()}
    anchor=next(i for i,r in enumerate(data["BTC"]) if r["available_at_utc"]==train.metadata[0]["decision_time_utc"])
    for r in changed["BTC"][anchor+1:]:
        for field in ("open","high","low","close"):
            r[field]*=2
    altered=build_samples(changed,config,"train")
    np.testing.assert_array_equal(train.x[0],altered.x[0])
    assert not np.array_equal(train.y[0],altered.y[0])


def test_a_missing_day_is_not_compressed_into_a_28_observation_target():
    data=rows(count=500)
    del data["BTC"][200]
    with pytest.raises(ValueError,match="contiguous"):
        build_samples(data,TrainingConfig(),"train")


def test_scaler_never_fits_validation_or_test(prepared):
    _,_,train,validation=prepared
    first=Normalizer.fit(train).to_dict()
    validation_copy=dataclasses.replace(validation,x=validation.x*1e6,y=validation.y*1e6)
    assert Normalizer.fit(train).to_dict()==first
    transformed=Normalizer.from_dict(first).x(validation_copy.x)
    assert abs(transformed).max()>1000


def test_linear_baseline_fits_train_only_and_zero_momentum_are_exact(prepared):
    _,_,train,validation=prepared
    state=fit_linear(train)
    predictions=predict_baselines(validation,state)
    np.testing.assert_array_equal(predictions["zero"],np.zeros_like(validation.y))
    np.testing.assert_array_equal(predictions["momentum"],validation.momentum)
    coefficients=np.array(state["coefficients"])
    np.testing.assert_allclose(predictions["linear"],validation.linear@coefficients)
    # Recover a known linear function without fitting on evaluation targets.
    synthetic=dataclasses.replace(train,y=train.linear@np.arange(12).reshape(6,2)/100)
    fitted=fit_linear(synthetic)
    np.testing.assert_allclose(np.array(fitted["coefficients"]),np.arange(12).reshape(6,2)/100,atol=1e-8)


def test_error_metrics_and_flat_direction_rule_are_auditable():
    truth=np.array([[1.,-1.],[-1.,1.]])
    result=errors(truth,np.zeros((2,2)))
    assert result["14"]=={"n":2,"mae":1.,"rmse":1.,"direction_accuracy":0.}
    assert result["28"]==result["14"]
    with pytest.raises(ValueError):
        errors(truth,np.full((2,2),np.nan))


def test_regime_definitions_use_only_trailing_information(prepared):
    _,_,train,validation=prepared
    threshold=float(np.median([m["trailing_28d_volatility"] for m in train.metadata]))
    report=report_metrics(validation,{"zero":np.zeros_like(validation.y)},threshold)
    groups=report["metrics"]["zero"]
    n=sum(groups[g]["14"]["n"] for g in ("up_high_vol","up_low_vol","down_high_vol","down_low_vol") if groups[g])
    assert n==len(validation.y)
    assert set(groups)=={"all","BTC","ETH","up_high_vol","up_low_vol","down_high_vol","down_low_vol"}


def test_seal_is_consumed_even_if_evaluation_crashes(tmp_path):
    claim_final(tmp_path,"SYNTHETIC_SEAL","a"*64)
    # Simulate interruption before any results; a second model/output cannot retry.
    with pytest.raises(FileExistsError):
        claim_final(tmp_path,"SYNTHETIC_SEAL","b"*64)
    assert json.loads((tmp_path/"final.claim.json").read_text())["frozen_sha256"]=="a"*64


def test_model_dual_heads_gradients_seed_and_distinct_layer_initialization():
    torch=pytest.importorskip("torch")
    from quant_platform.ai.price.model import PriceReturnModel
    config=TrainingConfig()
    torch.manual_seed(42)
    first=PriceReturnModel(config)
    torch.manual_seed(42)
    second=PriceReturnModel(config)
    for a,b in zip(first.parameters(),second.parameters()):
        assert torch.equal(a,b)
    assert not torch.equal(first.price_encoder.transformer.layers[0].linear1.weight,
                           first.price_encoder.transformer.layers[1].linear1.weight)
    x=torch.randn(4,90,5); asset=torch.tensor([0,1,0,1])
    output=first(x,asset)
    assert output.shape==(4,2) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert first.return_14d.weight.grad is not None and first.return_28d.weight.grad is not None


def test_formal_training_refuses_controller_or_cpu():
    torch=pytest.importorskip("torch")
    from quant_platform.ai.price.train import runtime
    with pytest.raises(ValueError,match="RTX 4090"):
        runtime(42,functional=False,device="cpu")


def test_functional_subset_exercises_both_assets_without_changing_time_order(prepared):
    small=prepared[2].subset(128)
    assert len(small.y)==128
    assert np.bincount(small.asset).tolist()==[64,64]
    assert small.metadata[0]["symbol"]=="BTC" and small.metadata[64]["symbol"]=="ETH"


def test_freeze_rejects_changed_weights_config_or_source(tmp_path):
    from quant_platform.ai.price.artifacts import code_identity, file_hash
    from quant_platform.ai.price.train import verify_frozen
    config=TrainingConfig()
    release={"release_id":"SYNTHETIC_RELEASE","seal_id":"SYNTHETIC_SEAL"}
    (tmp_path/"model.pt").write_bytes(b"SYNTHETIC_WEIGHT_IDENTITY_ONLY")
    frozen={**release,"config_sha256":digest(config.to_dict()),"code":code_identity(),
            "files":{"model.pt":file_hash(tmp_path/"model.pt")}}
    write_new(tmp_path/"frozen.json",frozen)
    assert verify_frozen(tmp_path,release,config)==frozen
    with pytest.raises(ValueError,match="identity mismatch"):
        verify_frozen(tmp_path,release,dataclasses.replace(config,seed=43))
    (tmp_path/"model.pt").write_bytes(b"TAMPERED")
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        verify_frozen(tmp_path,release,config)


def test_development_loader_never_opens_sealed_file(tmp_path,prepared,monkeypatch):
    data,config,_,_=prepared
    from quant_platform.ai.price.dataset import load_development
    from quant_platform.ai.price.artifacts import digest
    recipe={"config":config.to_dict(),"development_sha256":digest(data),"sealed_sha256":"b"*64,"audit":{"BTC":{"status":"PASS"},"ETH":{"status":"PASS"}}}
    manifest={"status":"PASS","synthetic":False,"allowed_purpose":"PRICE_PREDICTION_RESEARCH_ONLY",
              "recipe":recipe,"release_id":digest(recipe),"files":{"development.json":digest(data),"sealed.json":"b"*64}}
    write_new(tmp_path/"manifest.json",manifest);write_new(tmp_path/"development.json",data)
    # There deliberately is no sealed.json: training and validation must still work.
    _,train,validation=load_development(tmp_path,config)
    assert len(train.y)>0 and len(validation.y)>0


def test_formal_fit_rejects_review_required_before_runtime_import(tmp_path,monkeypatch):
    from quant_platform.ai.price import train as module
    write_new(tmp_path/"manifest.json",{"status":"REVIEW_REQUIRED"})
    monkeypatch.setattr(module,"runtime",lambda *a,**k:pytest.fail("Runtime reached before quality gate"))
    with pytest.raises(ValueError):
        module.fit(tmp_path,tmp_path/"output")


def test_final_evaluation_cannot_start_without_frozen_model(tmp_path):
    # Claim behavior and real freeze identities are exercised without fabricated final scores.
    with pytest.raises(FileNotFoundError):
        from quant_platform.ai.price.train import evaluate_final_once
        evaluate_final_once(tmp_path,tmp_path/"missing")
