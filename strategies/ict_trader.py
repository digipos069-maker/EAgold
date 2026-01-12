import numpy as np
import MetaTrader5 as mt5
from indicators import calculate_ema

def run_ict_trader_logic(worker, symbol, timeframe):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 205)
    if rates is None or len(rates) < 205:
        worker.signals.update_status.emit("ICT: Not enough data.", "orange")
        return

    close_prices = np.array([r['close'] for r in rates])
    high_prices = np.array([r['high'] for r in rates])
    low_prices = np.array([r['low'] for r in rates])
    
    ema200 = calculate_ema(close_prices, 200)
    if ema200 is None:
        worker.signals.update_status.emit("ICT: Could not calculate EMA 200.", "orange")
        return
        
    current_price = close_prices[-1]
    worker.signals.update_status.emit(f"ICT: Price={current_price:.2f}, EMA200={ema200:.2f}", "cyan")

    # FVG Detection
    fvg_top = None
    fvg_bottom = None
    
    # Look for the most recent FVG in the last 10 candles
    for i in range(len(rates) - 3, len(rates) - 13, -1):
        # Bullish FVG (Imbalance)
        if high_prices[i-1] < low_prices[i+1]:
            fvg_top = low_prices[i+1]
            fvg_bottom = high_prices[i-1]
            # Check if price is inside the FVG
            if current_price <= fvg_top and current_price >= fvg_bottom:
                if current_price > ema200 and worker.last_trade_action != "Buy":
                    worker.signals.update_status.emit(f"ICT: Bullish FVG detected and entered. Price={current_price:.2f}", "yellow")
                    worker.last_trade_action = "Buy"
                    worker.execute_trade("Buy", is_auto=True)
                    return
                
        # Bearish FVG (Imbalance)
        elif low_prices[i-1] > high_prices[i+1]:
            fvg_top = low_prices[i-1]
            fvg_bottom = high_prices[i+1]
            # Check if price is inside the FVG
            if current_price <= fvg_top and current_price >= fvg_bottom:
                if current_price < ema200 and worker.last_trade_action != "Sell":
                    worker.signals.update_status.emit(f"ICT: Bearish FVG detected and entered. Price={current_price:.2f}", "yellow")
                    worker.last_trade_action = "Sell"
                    worker.execute_trade("Sell", is_auto=True)
                    return
    worker.signals.update_status.emit(f"ICT: No FVG entry signal. Price={current_price:.2f}, EMA200={ema200:.2f}", "cyan")
