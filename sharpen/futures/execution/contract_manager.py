"""Futures contract manager — front-month resolution and roll logic.

Gold futures (GC/MGC) have delivery months:
    Feb(G), Apr(J), Jun(M), Aug(Q), Oct(V), Dec(Z)

Each contract is front-month for ~2 months before its delivery month:
    Jan        -> G (Feb delivery)
    Feb-Mar    -> J (Apr delivery)
    Apr-May    -> M (Jun delivery)
    Jun-Jul    -> Q (Aug delivery)
    Aug-Sep    -> V (Oct delivery)
    Oct-Nov    -> Z (Dec delivery)
    Dec        -> G (Feb delivery, next year)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Gold delivery month codes and their calendar months
GOLD_MONTH_CODES = {
    2: "G",   # February
    4: "J",   # April
    6: "M",   # June
    8: "Q",   # August
    10: "V",  # October
    12: "Z",  # December
}

# Front-month mapping: calendar month -> delivery month number
# e.g., in January, the front-month is February (G)
FRONT_MONTH_MAP = {
    1: 2,    # Jan -> Feb delivery
    2: 4,    # Feb -> Apr delivery
    3: 4,    # Mar -> Apr delivery
    4: 6,    # Apr -> Jun delivery
    5: 6,    # May -> Jun delivery
    6: 8,    # Jun -> Aug delivery
    7: 8,    # Jul -> Aug delivery
    8: 10,   # Aug -> Oct delivery
    9: 10,   # Sep -> Oct delivery
    10: 12,  # Oct -> Dec delivery
    11: 12,  # Nov -> Dec delivery
    12: 2,   # Dec -> Feb delivery (next year)
}

# Contract multipliers and specs
CONTRACT_SPECS = {
    "GC": {"multiplier": 100, "tick_size": 0.10, "exchange": "COMEX", "currency": "USD"},
    "MGC": {"multiplier": 10, "tick_size": 0.10, "exchange": "COMEX", "currency": "USD"},
}


@dataclass
class ContractInfo:
    """Resolved futures contract details."""
    symbol: str          # "MGC" or "GC"
    local_symbol: str    # e.g., "MGCM6" (Micro Gold, June 2026)
    delivery_month: int  # e.g., 6
    delivery_year: int   # e.g., 2026
    expiry: Optional[date]  # Last trading day (set after IB qualification)
    multiplier: int      # 10 for MGC, 100 for GC
    tick_size: float     # 0.10
    exchange: str        # "COMEX"
    currency: str        # "USD"
    ib_contract: object | None = None  # ib_insync Future object (set after connect)


class FuturesContractManager:
    """Manages CME Gold futures contract lifecycle.

    Resolves the current front-month contract, detects when a roll
    is needed, and executes contract rolls.
    """

    def __init__(
        self,
        ib=None,
        symbol: str = "MGC",
        roll_days_before_expiry: int = 5,
    ):
        """
        Args:
            ib: Connected ib_insync.IB instance (None for offline testing).
            symbol: "MGC" (micro, 10 oz) or "GC" (full, 100 oz).
            roll_days_before_expiry: Roll N business days before first notice date.
        """
        if symbol not in CONTRACT_SPECS:
            raise ValueError(f"Unknown symbol: {symbol}. Must be one of {list(CONTRACT_SPECS)}")

        self._ib = ib
        self._symbol = symbol
        self._roll_days = roll_days_before_expiry
        self._spec = CONTRACT_SPECS[symbol]
        self._current: ContractInfo | None = None

    @property
    def contract(self) -> ContractInfo:
        """Currently active contract. Raises if not resolved."""
        if self._current is None:
            raise RuntimeError("Contract not resolved. Call resolve_front_month() first.")
        return self._current

    @property
    def multiplier(self) -> int:
        return self._spec["multiplier"]

    @property
    def tick_size(self) -> float:
        return self._spec["tick_size"]

    @property
    def ib_contract(self):
        """The ib_insync Future object for order placement."""
        return self.contract.ib_contract

    def compute_front_month(self, as_of: date | None = None) -> tuple[int, int]:
        """Compute the front-month delivery (month, year) for a given date.

        Returns:
            (delivery_month, delivery_year) e.g., (6, 2026) for June 2026.
        """
        if as_of is None:
            as_of = datetime.now(timezone.utc).date()

        delivery_month = FRONT_MONTH_MAP[as_of.month]
        delivery_year = as_of.year

        # December rolls to February of next year
        if as_of.month == 12:
            delivery_year += 1

        return delivery_month, delivery_year

    def _build_local_symbol(self, delivery_month: int, delivery_year: int) -> str:
        """Build the local symbol string, e.g., 'MGCM6' for Micro Gold June 2026."""
        month_code = GOLD_MONTH_CODES[delivery_month]
        year_digit = delivery_year % 10
        return f"{self._symbol}{month_code}{year_digit}"

    async def resolve_front_month(self, as_of: date | None = None) -> ContractInfo:
        """Resolve and qualify the current front-month contract via IB.

        Sets self._current and returns the ContractInfo.
        """
        delivery_month, delivery_year = self.compute_front_month(as_of)
        local_symbol = self._build_local_symbol(delivery_month, delivery_year)

        info = ContractInfo(
            symbol=self._symbol,
            local_symbol=local_symbol,
            delivery_month=delivery_month,
            delivery_year=delivery_year,
            expiry=None,
            multiplier=self._spec["multiplier"],
            tick_size=self._spec["tick_size"],
            exchange=self._spec["exchange"],
            currency=self._spec["currency"],
        )

        if self._ib is not None:
            info = await self._qualify_contract(info)

        self._current = info
        logger.info(
            f"Resolved front-month: {local_symbol} "
            f"(delivery {delivery_month}/{delivery_year}, "
            f"multiplier={info.multiplier})"
        )
        return info

    async def _qualify_contract(self, info: ContractInfo) -> ContractInfo:
        """Qualify the contract with IB to get exact expiry and conId."""
        from ib_insync import Future

        # Build IB contract object
        # IB uses YYYYMM format for lastTradeDateOrContractMonth
        contract_month = f"{info.delivery_year}{info.delivery_month:02d}"
        ib_fut = Future(
            symbol=info.symbol,
            lastTradeDateOrContractMonth=contract_month,
            exchange=info.exchange,
            currency=info.currency,
        )

        qualified = await self._ib.qualifyContractsAsync(ib_fut)
        if not qualified:
            raise RuntimeError(
                f"Failed to qualify contract: {info.local_symbol} "
                f"({info.symbol} {contract_month} on {info.exchange})"
            )

        ib_fut = qualified[0]
        info.ib_contract = ib_fut

        # Parse expiry from qualified contract
        if ib_fut.lastTradeDateOrContractMonth:
            expiry_str = ib_fut.lastTradeDateOrContractMonth
            if len(expiry_str) == 8:  # YYYYMMDD
                info.expiry = date(
                    int(expiry_str[:4]),
                    int(expiry_str[4:6]),
                    int(expiry_str[6:8]),
                )

        logger.info(
            f"Qualified: {ib_fut.localSymbol} conId={ib_fut.conId} "
            f"expiry={info.expiry}"
        )
        return info

    def check_roll_needed(self, as_of: date | None = None) -> bool:
        """Check if the current contract needs to be rolled.

        Roll triggers:
            1. Current contract is within roll_days of expiry.
            2. Front-month has changed (e.g., new month).
        """
        if self._current is None:
            return True  # No contract resolved yet

        if as_of is None:
            as_of = datetime.now(timezone.utc).date()

        # Check if front-month has changed
        new_month, new_year = self.compute_front_month(as_of)
        if (new_month != self._current.delivery_month
                or new_year != self._current.delivery_year):
            logger.info(
                f"Roll needed: front-month changed from "
                f"{self._current.local_symbol} to "
                f"{self._build_local_symbol(new_month, new_year)}"
            )
            return True

        # Check expiry proximity
        if self._current.expiry is not None:
            days_to_expiry = (self._current.expiry - as_of).days
            if days_to_expiry <= self._roll_days:
                logger.info(
                    f"Roll needed: {days_to_expiry} days to expiry "
                    f"(threshold: {self._roll_days})"
                )
                return True

        return False

    async def roll_contract(self) -> ContractInfo:
        """Roll to the next front-month contract.

        Caller is responsible for flattening the old contract position
        before calling this method.
        """
        old_symbol = self._current.local_symbol if self._current else "none"
        new_info = await self.resolve_front_month()
        logger.info(f"Rolled contract: {old_symbol} -> {new_info.local_symbol}")
        return new_info
