"""
NautilusTrader Backtest — LogReg BC Strategy on Gold 15-min
============================================================

Runs the winning LogReg BC model (from Stage 4 walk-forward) through
NautilusTrader's backtest engine with proper fill simulation.

The strategy:
  - Receives 15-min bars
  - Computes 18 OHLCV features (rolling window)
  - Runs LogReg inference (single dot product)
  - Submits market orders to switch Long <-> Short

Usage:
    python scripts/nautilus_backtest.py
    python scripts/nautilus_backtest.py --data data/processed/gc_2025_3min_front.parquet
    python scripts/nautilus_backtest.py --test_start 2025-11-01 --test_end 2026-01-01
"""

import argparse
import io
import sys
from contextlib import redirect_stdout
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import Bar, BarSpecification
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    BarAggregation,
    OmsType,
    OrderSide,
    PriceType,
)
from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.bc_feasibility_test import build_features


# ═══════════════════════════════════════════════════════════════════════
# Section 1: Feature Computer (batch-precomputed)
# ═══════════════════════════════════════════════════════════════════════

def precompute_features(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load full dataset, resample to 15-min, compute all features upfront.

    Returns (df_15min, features_df) where features_df is aligned 1:1 with df_15min.
    This avoids the O(n²) cost of recomputing features on every bar.
    """
    df = pd.read_parquet(data_path)
    print(f"  Loaded {len(df):,} raw bars from {data_path}")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    # Resample to 15-min
    df_15 = df.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna().reset_index()

    df_15['mid_price'] = (df_15['high'] + df_15['low']) / 2.0
    print(f"  Resampled to 15-min: {len(df_15):,} bars")

    # Compute features once (suppress verbose output)
    buf = io.StringIO()
    with redirect_stdout(buf):
        features_df = build_features(df_15)

    print(f"  Features computed: {features_df.shape[1]} dims")
    return df_15, features_df


# ═══════════════════════════════════════════════════════════════════════
# Section 2: LogReg BC Strategy
# ═══════════════════════════════════════════════════════════════════════

class LogRegStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for LogReg BC strategy."""
    instrument_id: str
    bar_type: str
    model_path: str = "checkpoints/bc_gold_15min/best_logreg.joblib"
    trade_size: str = "1"
    warmup_bars: int = 60


class LogRegStrategy(Strategy):
    """NautilusTrader strategy wrapping the LogReg BC model.

    Features are precomputed and indexed by bar count.
    On each bar: look up features, run LogReg, switch if needed.
    """

    def __init__(self, config: LogRegStrategyConfig,
                 features_array: np.ndarray, warmup_bars: int = 60) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)
        self.trade_size = Decimal(config.trade_size)

        # Load model
        ckpt = joblib.load(config.model_path)
        self.model = ckpt['model']
        self.feature_names = ckpt['feature_names']
        self.fee_bps = ckpt['fee_bps']

        # Precomputed features (aligned with bar index)
        self.features_array = features_array
        self.warmup = warmup_bars

        # State
        self.current_direction = None  # None=no position, 0=Short, 1=Long
        self.bar_count = 0
        self.signal_count = 0
        self.switch_count = 0

    def on_start(self) -> None:
        self.subscribe_bars(self.bar_type)
        self.log.info(f"LogReg BC strategy started — {self.instrument_id}")
        self.log.info(f"Model: {len(self.feature_names)} features, fee={self.fee_bps}bps")
        self.log.info(f"Precomputed features: {self.features_array.shape}")

    def on_bar(self, bar: Bar) -> None:
        self.bar_count += 1
        idx = self.bar_count - 1  # 0-based index into features array

        if idx < self.warmup:
            return

        if idx >= len(self.features_array):
            return

        # LogReg inference from precomputed features
        features = self.features_array[idx]
        direction = int(self.model.predict(features.reshape(1, -1))[0])
        self.signal_count += 1

        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            return

        qty = instrument.make_qty(self.trade_size)

        if self.current_direction is None:
            # Initial entry
            side = OrderSide.BUY if direction == 1 else OrderSide.SELL
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=qty,
            )
            self._pending_direction = direction
            self.submit_order(order)

        elif direction != self.current_direction:
            # Switch: with NETTING, 2x qty flips the position
            new_side = OrderSide.BUY if direction == 1 else OrderSide.SELL
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=new_side,
                quantity=instrument.make_qty(self.trade_size * 2),
            )
            self._pending_direction = direction
            self._pending_is_switch = True
            self.submit_order(order)

    def on_order_filled(self, event) -> None:
        if hasattr(self, '_pending_direction'):
            if hasattr(self, '_pending_is_switch') and self._pending_is_switch:
                self.switch_count += 1
                self._pending_is_switch = False
            self.current_direction = self._pending_direction
            del self._pending_direction

    def on_order_rejected(self, event) -> None:
        # Clear pending state on rejection
        if hasattr(self, '_pending_direction'):
            del self._pending_direction
        if hasattr(self, '_pending_is_switch'):
            del self._pending_is_switch

    def on_stop(self) -> None:
        self.close_all_positions(self.instrument_id)
        self.log.info(
            f"Strategy stopped. Bars={self.bar_count}, "
            f"Signals={self.signal_count}, Switches={self.switch_count}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Data Loading — Parquet to NautilusTrader Bars
# ═══════════════════════════════════════════════════════════════════════

def make_bars(
    df_15: pd.DataFrame,
    instrument_id: InstrumentId,
    bar_type: BarType,
    start_idx: int = 0,
) -> list[Bar]:
    """Convert DataFrame rows to NautilusTrader Bar objects."""
    bars = []
    for i in range(start_idx, len(df_15)):
        row = df_15.iloc[i]
        ts = pd.Timestamp(row['timestamp'])
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{row['open']:.2f}"),
            high=Price.from_str(f"{row['high']:.2f}"),
            low=Price.from_str(f"{row['low']:.2f}"),
            close=Price.from_str(f"{row['close']:.2f}"),
            volume=Quantity.from_str(f"{max(row['volume'], 0):.0f}"),
            ts_event=ts.value,
            ts_init=ts.value,
        )
        bars.append(bar)
    return bars


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Instrument Definition — Gold Futures (GC)
# ═══════════════════════════════════════════════════════════════════════

def create_gold_futures_instrument(venue: Venue, fee_bps: float = 6.5):
    """Create a Gold Futures (GC) instrument for backtesting."""
    from nautilus_trader.model.instruments import FuturesContract
    from nautilus_trader.model.identifiers import Symbol
    from nautilus_trader.model.enums import AssetClass

    instrument_id = InstrumentId(Symbol("GC"), venue)
    fee_frac = Decimal(str(fee_bps / 10000.0))

    # Set activation/expiration to cover our data range (2025-01-01 to 2026-06-01)
    activation_ns = int(pd.Timestamp("2024-01-01", tz="UTC").value)
    expiration_ns = int(pd.Timestamp("2027-01-01", tz="UTC").value)

    instrument = FuturesContract(
        instrument_id=instrument_id,
        raw_symbol=Symbol("GC"),
        asset_class=AssetClass.COMMODITY,
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.10"),  # GC tick size = $0.10
        multiplier=Quantity.from_int(100),  # GC multiplier = 100 troy oz
        lot_size=Quantity.from_int(1),
        underlying="GOLD",
        margin_init=Decimal("0.05"),   # 5% initial margin (~20x leverage)
        margin_maint=Decimal("0.04"),  # 4% maintenance margin
        maker_fee=fee_frac,
        taker_fee=fee_frac,
        activation_ns=activation_ns,
        expiration_ns=expiration_ns,
        ts_event=0,
        ts_init=0,
    )

    return instrument


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Results Analysis
# ═══════════════════════════════════════════════════════════════════════

def print_results(engine, venue, strategy):
    """Print comprehensive backtest results."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    # Account report
    report = engine.trader.generate_account_report(venue)
    if report is not None and len(report) > 0:
        first_balance = report.iloc[0]['total'] if 'total' in report.columns else None
        last_balance = report.iloc[-1]['total'] if 'total' in report.columns else None
        if first_balance is not None and last_balance is not None:
            pnl = float(last_balance) - float(first_balance)
            ret_pct = pnl / float(first_balance) * 100
            print(f"\n  Starting balance: ${float(first_balance):,.2f}")
            print(f"  Ending balance:   ${float(last_balance):,.2f}")
            print(f"  PnL:              ${pnl:+,.2f} ({ret_pct:+.2f}%)")

    # Order fills
    fills = engine.trader.generate_order_fills_report()
    n_fills = len(fills) if fills is not None else 0
    print(f"\n  Total fills:     {n_fills}")

    # Positions
    positions = engine.trader.generate_positions_report()
    if positions is not None and len(positions) > 0:
        # Compute PF from closed position PnLs
        if 'realized_pnl' in positions.columns:
            # NT returns Money strings like "1234.56 USD" — extract numeric part
            pnls = positions['realized_pnl'].apply(
                lambda x: float(str(x).split()[0]) if pd.notna(x) else 0.0
            )
            gross_profit = pnls[pnls > 0].sum()
            gross_loss = abs(pnls[pnls < 0].sum())
            pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            n_pos = len(positions)
            n_win = (pnls > 0).sum()
            n_lose = (pnls < 0).sum()
            win_rate = n_win / n_pos * 100 if n_pos > 0 else 0

            print(f"\n  Closed positions: {n_pos}")
            print(f"  Win/Loss:         {n_win}/{n_lose} ({win_rate:.1f}% win rate)")
            print(f"  Gross profit:     ${gross_profit:,.2f}")
            print(f"  Gross loss:       ${gross_loss:,.2f}")
            print(f"  Profit Factor:    {pf:.3f}")

            # Average win / average loss
            avg_win = pnls[pnls > 0].mean() if n_win > 0 else 0
            avg_loss = abs(pnls[pnls < 0].mean()) if n_lose > 0 else 0
            print(f"  Avg win:          ${avg_win:,.2f}")
            print(f"  Avg loss:         ${avg_loss:,.2f}")
        else:
            print(f"\n  Positions: {len(positions)} (no realized_pnl column)")
            print(positions.columns.tolist())

    # Strategy stats
    print("\n  Strategy Stats:")
    print(f"    Bars processed:  {strategy.bar_count}")
    print(f"    Signals emitted: {strategy.signal_count}")
    print(f"    Switches:        {strategy.switch_count}")
    print(f"    Switch rate:     {strategy.switch_count / max(strategy.signal_count, 1) * 100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Main — Backtest Runner
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NautilusTrader Backtest — LogReg BC on Gold 15-min")
    parser.add_argument("--data", default="data/processed/gc_2025_3min_front.parquet",
                        help="Path to 3-min OHLCV parquet")
    parser.add_argument("--model", default="checkpoints/bc_gold_15min/best_logreg.joblib",
                        help="Path to LogReg model checkpoint")
    parser.add_argument("--test_start", default=None,
                        help="Test period start (default: use all data)")
    parser.add_argument("--test_end", default=None,
                        help="Test period end")
    parser.add_argument("--balance", type=float, default=100_000.0,
                        help="Starting balance in USD (default: 100000)")
    parser.add_argument("--trade_size", type=int, default=1,
                        help="Number of contracts per trade (default: 1)")
    parser.add_argument("--fee_bps", type=float, default=6.5,
                        help="One-way fee in bps (default: 6.5)")
    args = parser.parse_args()

    print("=" * 70)
    print("NAUTILUS TRADER BACKTEST — LogReg BC Strategy")
    print("=" * 70)

    # --- Precompute features on full dataset ---
    print("\n--- Precomputing Features ---")
    df_15, features_df = precompute_features(args.data)
    features_array = features_df.values.astype(np.float32)

    # --- Determine bar range ---
    if args.test_start or args.test_end:
        ts = pd.to_datetime(df_15['timestamp'])
        mask = pd.Series(True, index=df_15.index)
        if args.test_start:
            mask &= ts >= args.test_start
        if args.test_end:
            mask &= ts < args.test_end
        bar_indices = df_15.index[mask].tolist()
        start_idx = bar_indices[0] if bar_indices else 0
        print(f"  Test window: {args.test_start} -> {args.test_end}")
        print(f"  Test bars: {len(bar_indices)} (indices {start_idx}..{bar_indices[-1]})")
        # Include warmup bars before test start
        warmup_start = max(0, start_idx - 60)
        print(f"  Including warmup from index {warmup_start}")
    else:
        warmup_start = 0
        print("  Using all data (full backtest)")

    # --- Setup venue and instrument ---
    venue = Venue("CME")
    instrument = create_gold_futures_instrument(venue, args.fee_bps)
    instrument_id = instrument.id
    print(f"\n  Instrument: {instrument_id}")
    print(f"  Multiplier: {instrument.multiplier}, Tick: {instrument.price_increment}")
    print(f"  Fee: {args.fee_bps} bps one-way")

    # --- Bar type ---
    bar_spec = BarSpecification(
        step=15,
        aggregation=BarAggregation.MINUTE,
        price_type=PriceType.LAST,
    )
    bar_type = BarType(
        instrument_id=instrument_id,
        bar_spec=bar_spec,
        aggregation_source=AggregationSource.EXTERNAL,
    )

    # --- Convert to NT bars ---
    bars = make_bars(df_15, instrument_id, bar_type, start_idx=warmup_start)
    print(f"  NautilusTrader bars: {len(bars)}")

    if not bars:
        print("ERROR: No bars. Check data path and date range.")
        return

    # --- Slice features to match bar range ---
    features_slice = features_array[warmup_start:]

    # --- Configure engine ---
    print("\n--- Configuring Backtest Engine ---")
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
    )
    engine = BacktestEngine(config=engine_config)

    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(args.balance, USD)],
    )

    engine.add_instrument(instrument)
    engine.add_data(bars)

    # --- Configure strategy ---
    # Warmup relative to the slice (first 60 bars of slice if starting from warmup_start)
    relative_warmup = 60 if warmup_start > 0 else 60

    strategy_config = LogRegStrategyConfig(
        instrument_id=str(instrument_id),
        bar_type=str(bar_type),
        model_path=args.model,
        trade_size=str(args.trade_size),
        warmup_bars=relative_warmup,
    )
    strategy = LogRegStrategy(
        config=strategy_config,
        features_array=features_slice,
        warmup_bars=relative_warmup,
    )
    engine.add_strategy(strategy)

    # --- Run ---
    print("\n--- Running Backtest ---")
    engine.run()

    # --- Results ---
    print_results(engine, venue, strategy)

    engine.dispose()
    print("\n  Done.")


if __name__ == "__main__":
    main()
