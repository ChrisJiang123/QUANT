# QUANT — offline crypto research

Public source snapshot of an existing research system. It observes BTC/USD and
ETH/USD through free public data, computes causal factors and frozen rule weights,
and uses an isolated Freqtrade process for costed offline backtests. There are no
live accounts, trading credentials, order connections, leverage or online bots.

Current research verdict: **NO_GO_ALPHA**. The old terminal interval and the new
forward track remain **INCONCLUSIVE_DATA**. No additional training was justified
in this continuation. Functional software does not establish predictive value.

The receipt-time rule track records actual completed-response availability,
strictly future execution times, hash-chained observations and append-only
corrections. It settles real 14/28-day executable-open labels only after prices
have actually arrived. Unmatured results remain pending. Same-key runs are
idempotent. Missing or stale prices suppress new observations.

- [Daily entry, user scheduling, recovery and acceptance](docs/FORWARD_OPERATIONS.md)
- [Sanitized current report](reports/V4_NEXT_STEP_REPORT.md)
- [Machine-readable historical comparison](reports/V4_NEXT_STEP_METRICS.json)
- [Third-party attribution](THIRD_PARTY_NOTICES.md)

The source contains the original data/model/research modules as well as the new
rule observation entry. Historical source versions had execution-causality and
registration limitations; they remain research-negative and are not endorsed for
deployment. Only the rule DEV correction was rerun in this continuation. Old
model backtest numbers are explicitly historical exploratory results.

This public snapshot excludes the private Git ancestry, user planning documents,
historical machine reports, raw data, weights, caches, environments and connection
information. The original local history is retained unchanged. Local manifests
and the final-consumption gate refer to locally retained artifacts; they are not
reconstructible from this public snapshot alone. Reproducing historical financial
numbers requires those exact retained data/weights. The public source does not
claim a clean fresh terminal evaluation or a self-contained data distribution.

Dependencies are separately locked in `requirements-v4-core-lock.txt`,
`requirements-v4-freqtrade-lock.txt` and `requirements-v4-qlib-lock.txt`.
The forward entry reuses the existing local environments and archived metadata;
it does not install services or change system settings. Generic isolated tests:

```powershell
python -m pytest -q tests/test_rule_forward.py
```

Other integration tests require the retained private artifacts and isolated
Freqtrade/Qlib environments. Full historical tests cannot be claimed from a
source-only checkout. Freqtrade plugins carry GPL-3.0-or-later labels; retained
vendor licenses are under `third_party/licenses`. Publication does not assume
redistribution rights for market datasets or grant an unstated project license.
