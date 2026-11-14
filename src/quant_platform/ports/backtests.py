from typing import Protocol

from quant_platform.domain import BacktestResult, BacktestRun


class BacktestPort(Protocol):
    def run(self, specification: BacktestRun) -> BacktestResult:
        """Offline engine boundary; never place orders or promote data quality."""
        ...
