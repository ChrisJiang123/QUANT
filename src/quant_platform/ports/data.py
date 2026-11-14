from datetime import datetime
from typing import Iterable, Protocol

from quant_platform.domain import Instrument, MarketObservation


class MarketDataPort(Protocol):
    def observations(
        self, instrument: Instrument, *, start_utc: datetime, end_utc: datetime,
        decision_time_utc: datetime,
    ) -> Iterable[MarketObservation]:
        """[start,end) events with availability <= decision; retain quality flags."""
        ...
