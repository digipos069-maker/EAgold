import numpy as np
import MetaTrader5 as mt5

def run_trend_following_logic(worker, symbol, timeframe, signal_period, trend_period):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, trend_period + 5)
    if rates is None or len(rates) < trend_period: return
    close = np.array([rate['close'] for rate in rates])
    signal_ma = np.mean(close[-signal_period:]); trend_ma = np.mean(close[-trend_period:])
    prev_close = close[-2]; curr_close = close[-1]
    prev_signal_ma = np.mean(close[-signal_period-1:-1])
    worker.signals.update_status.emit(f"Checking Trend Following: Price={curr_close:.2f}, Trend MA={trend_ma:.2f}", "cyan")
    is_uptrend = curr_close > trend_ma
    if is_uptrend and worker.last_trade_action != "Buy":
        if prev_close < prev_signal_ma and curr_close > signal_ma:
            worker.last_trade_action = "Buy"; worker.execute_trade("Buy", is_auto=True)
    elif not is_uptrend and worker.last_trade_action != "Sell":
        if prev_close > prev_signal_ma and curr_close < signal_ma:
            worker.last_trade_action = "Sell"; worker.execute_trade("Sell", is_auto=True)
