"""Fee drag analysis for PPO run yg8gjeey."""

btc_price = 93000
initial_balance = 100000

# From config
taker_fee = 0.0012  # 3x real
maker_fee = 0.0006  # 3x real

# From backtest summary
total_trades = 1752
test_steps = 22210
final_value = 81024.21

# From action config: signed_qty_proportions = [-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5]
# Non-zero proportions: 8 out of 9
# Average abs qty: mean(0.5, 0.2, 0.1, 0.05, 0.05, 0.1, 0.2, 0.5) = 0.2125
avg_qty = 0.2125  
avg_notional = btc_price * avg_qty
avg_fee = avg_notional * taker_fee  # assume taker (worst case)
total_estimated_fees = avg_fee * total_trades
actual_loss = initial_balance - final_value

print("=== FEE DRAG ANALYSIS ===")
print(f"BTC price: ${btc_price:,}")
print(f"Initial balance: ${initial_balance:,}")
print(f"Final value: ${final_value:,.2f}")
print(f"Actual loss: ${actual_loss:,.2f} ({actual_loss/initial_balance:.1%})")
print()
print(f"Total trades (test): {total_trades}")
print(f"Test steps (minutes): {test_steps}")
print(f"Trades per minute: {total_trades/test_steps:.3f}")
print()
print(f"Avg trade size: {avg_qty} BTC = ${avg_notional:,.0f} notional")
print(f"Avg fee per trade: ${avg_fee:.2f} (taker)")
print(f"Estimated total fees: ${total_estimated_fees:,.0f}")
print(f"Fee drag as % of capital: {total_estimated_fees/initial_balance:.1%}")
print(f"Fees as % of total loss: {total_estimated_fees/actual_loss:.0%}")
print()

# Breakeven analysis
breakeven_bps = avg_fee / avg_notional * 10000
print(f"Breakeven per trade: {breakeven_bps:.1f} bps (= {taker_fee*10000:.0f} bps = fee rate)")
print(f"At {total_trades/test_steps:.3f} trades/min, need to earn ${avg_fee:.2f}/trade (= fee) just to break even")
print()

# BTC volatility context
# 1-min BTC volatility is typically 5-15 bps
print("=== CONTEXT ===")
print(f"Fee per trade: {taker_fee*10000:.0f} bps")
print(f"Typical 1-min BTC volatility: ~5-15 bps")
print(f"Signal-to-noise ratio needed: fee/vol = {taker_fee*10000/10:.1f}x")
print("=> Agent needs to capture >12 bps per trade, but BTC only moves ~10 bps/min")
print("=> Almost impossible without leverage or lower fees!")
