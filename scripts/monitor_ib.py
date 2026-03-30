#!/usr/bin/env python3
"""IB account monitor — read-only view of positions, P&L, and orders.

Connects to an already-running IB Gateway with a separate clientId (99)
so it does NOT interfere with the live trading session (clientId=1).

Due to IB Gateway TrustedIPs=127.0.0.1, the monitor must connect from
inside the Docker network. Use --docker mode (default) which runs the
query inside the gmgp1-gold container via `docker exec`.

Usage:
    # Default: runs via docker exec on remote desktop
    python scripts/monitor_ib.py

    # Watch mode — refresh every N seconds
    python scripts/monitor_ib.py --watch 60

    # JSON output (for piping to other tools)
    python scripts/monitor_ib.py --json

    # Direct mode (from inside the container or same network)
    python scripts/monitor_ib.py --direct --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap

# Python snippet executed inside the Docker container.
# Uses ib_insync 0.9.86 API (accountSummaryAsync, positions, etc.)
_CONTAINER_SCRIPT = textwrap.dedent(r'''
import asyncio
import json
from ib_insync import IB

async def snapshot():
    ib = IB()
    await ib.connectAsync("127.0.0.1", {port}, clientId={client_id}, timeout=15)

    result = {{"account": {{}}, "positions": [], "orders": [], "fills": []}}

    # Account summary
    summary = await ib.accountSummaryAsync()
    for s in summary:
        if s.tag in ("NetLiquidation", "TotalCashValue", "UnrealizedPnL",
                      "RealizedPnL", "AvailableFunds", "BuyingPower",
                      "GrossPositionValue", "MaintMarginReq", "InitMarginReq"):
            result["account"][s.tag] = s.value

    # Positions + portfolio
    for p in ib.portfolio():
        c = p.contract
        result["positions"].append({{
            "symbol": c.localSymbol or c.symbol,
            "exchange": c.exchange or "",
            "quantity": float(p.position),
            "avg_cost": float(p.averageCost),
            "market_price": float(p.marketPrice),
            "market_value": float(p.marketValue),
            "unrealized_pnl": float(p.unrealizedPNL),
            "realized_pnl": float(p.realizedPNL),
        }})

    if not result["positions"]:
        for pos in ib.positions():
            c = pos.contract
            result["positions"].append({{
                "symbol": c.localSymbol or c.symbol,
                "exchange": c.exchange or "",
                "quantity": float(pos.position),
                "avg_cost": float(pos.avgCost),
            }})

    # Open orders
    for t in ib.openTrades():
        o = t.order
        result["orders"].append({{
            "order_id": o.orderId,
            "symbol": t.contract.localSymbol or t.contract.symbol,
            "action": o.action,
            "quantity": float(o.totalQuantity),
            "order_type": o.orderType,
            "limit_price": float(getattr(o, "lmtPrice", 0)),
            "status": t.orderStatus.status if t.orderStatus else "unknown",
        }})

    # Recent fills
    for f in ib.fills():
        result["fills"].append({{
            "symbol": f.contract.localSymbol or f.contract.symbol,
            "time": str(f.execution.time) if f.execution.time else "",
            "action": f.execution.side,
            "quantity": float(f.execution.shares),
            "price": float(f.execution.price),
            "commission": float(f.commissionReport.commission) if f.commissionReport else 0,
        }})

    ib.disconnect()
    print(json.dumps(result))

asyncio.run(snapshot())
''')


def run_in_docker(container: str, port: int, client_id: int) -> dict | None:
    """Execute the monitor snippet inside a Docker container."""
    script = _CONTAINER_SCRIPT.format(port=port, client_id=client_id)
    cmd = ["docker", "exec", container, "python", "-c", script]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        print("ERROR: docker exec timed out (30s)", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"ERROR: Container script failed:\n{result.stderr}", file=sys.stderr)
        return None

    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON from container:\n{result.stdout}", file=sys.stderr)
        return None


def format_currency(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"${float(val):,.2f}"
    except (ValueError, TypeError):
        return str(val)


def format_pnl(val) -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
        return f"{'+' if v >= 0 else ''}${v:,.2f}"
    except (ValueError, TypeError):
        return str(val)


def print_snapshot(data: dict) -> None:
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"\n{'='*60}")
    print(f"  IB Account Monitor -- {ts}")
    print(f"{'='*60}")

    # Account
    acct = data.get("account", {})
    if acct:
        print("\n  Account Summary")
        print(f"  {'─'*40}")
        print(f"  Net Liquidation:  {format_currency(acct.get('NetLiquidation'))}")
        print(f"  Cash:             {format_currency(acct.get('TotalCashValue'))}")
        print(f"  Unrealized P&L:   {format_pnl(acct.get('UnrealizedPnL'))}")
        print(f"  Realized P&L:     {format_pnl(acct.get('RealizedPnL'))}")
        print(f"  Available Funds:  {format_currency(acct.get('AvailableFunds'))}")
        print(f"  Buying Power:     {format_currency(acct.get('BuyingPower'))}")
        print(f"  Gross Position:   {format_currency(acct.get('GrossPositionValue'))}")
        print(f"  Maint. Margin:    {format_currency(acct.get('MaintMarginReq'))}")
    else:
        print("\n  [No account data received]")

    # Positions
    positions = data.get("positions", [])
    print(f"\n  Positions ({len(positions)})")
    print(f"  {'─'*40}")
    if positions:
        for p in positions:
            sym = p["symbol"]
            qty = p["quantity"]
            direction = "LONG" if qty > 0 else "SHORT" if qty < 0 else "FLAT"
            avg = format_currency(p.get("avg_cost"))
            mkt = format_currency(p.get("market_price"))
            upnl = format_pnl(p.get("unrealized_pnl"))
            rpnl = format_pnl(p.get("realized_pnl"))
            print(f"  {sym:12s}  {direction:5s}  qty={qty:g}  avg={avg}  "
                  f"mkt={mkt}  uPnL={upnl}  rPnL={rpnl}")
    else:
        print("  (flat -- no positions)")

    # Open Orders
    orders = data.get("orders", [])
    print(f"\n  Open Orders ({len(orders)})")
    print(f"  {'─'*40}")
    if orders:
        for o in orders:
            lmt = format_currency(o.get("limit_price")) if o.get("limit_price") else ""
            price_str = f"@ {lmt}" if lmt else ""
            print(f"  #{o['order_id']:>6}  {o['symbol']:12s}  {o['action']:4s}  "
                  f"{o['quantity']:g}  {o['order_type']} {price_str}  [{o['status']}]")
    else:
        print("  (none)")

    # Recent Fills
    fills = data.get("fills", [])
    print(f"\n  Recent Fills ({len(fills)})")
    print(f"  {'─'*40}")
    if fills:
        for f in fills[-10:]:
            comm = f"  comm={format_currency(f['commission'])}" if f.get("commission") else ""
            print(f"  {f['time']}  {f['symbol']:12s}  {f['action']:4s}  "
                  f"{f['quantity']:g} @ {format_currency(f['price'])}{comm}")
    else:
        print("  (no fills this session)")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="IB account monitor (read-only, clientId=99)",
    )
    parser.add_argument(
        "--container", default="gmgp1-gold",
        help="Docker container to exec into (default: gmgp1-gold)",
    )
    parser.add_argument(
        "--port", type=int, default=4002,
        help="IB Gateway port (4002=paper, 4001=live)",
    )
    parser.add_argument(
        "--client-id", type=int, default=99,
        help="Client ID for monitor connection (default: 99)",
    )
    parser.add_argument(
        "--watch", type=int, default=0,
        help="Refresh interval in seconds (0=one-shot, 60=every minute)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Output as JSON",
    )
    args = parser.parse_args()

    import time
    while True:
        data = run_in_docker(args.container, args.port, args.client_id)
        if data is None:
            sys.exit(1)

        if args.as_json:
            print(json.dumps(data, indent=2))
        else:
            print_snapshot(data)

        if args.watch <= 0:
            break

        print(f"  [Next refresh in {args.watch}s -- Ctrl+C to exit]")
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break


if __name__ == "__main__":
    main()
