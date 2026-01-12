import numpy as np
import MetaTrader5 as mt5

def run_ma_crossover_logic(worker, symbol, timeframe, short_period, long_period):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, long_period + 5)
    if rates is None or len(rates) < long_period: return
    close = np.array([rate['close'] for rate in rates])
    short_ma = np.mean(close[-short_period:]); long_ma = np.mean(close[-long_period:])
    prev_short_ma = np.mean(close[-short_period-1:-1]); prev_long_ma = np.mean(close[-long_period-1:-1])
    worker.signals.update_status.emit(f"Checking MA Crossover: Short MA={short_ma:.2f}, Long MA={long_ma:.2f}", "cyan")
    if prev_short_ma < prev_long_ma and short_ma > long_ma and worker.last_trade_action != "Buy":
        worker.last_trade_action = "Buy"; worker.execute_trade("Buy", is_auto=True)
    elif prev_short_ma > prev_long_ma and short_ma < long_ma and worker.last_trade_action != "Sell":
        worker.last_trade_action = "Sell"; worker.execute_trade("Sell", is_auto=True)
