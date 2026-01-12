import numpy as np
import MetaTrader5 as mt5
from indicators import calculate_rsi, calculate_ema, calculate_adx, calculate_stochastic

def run_gold_scalper_logic(worker, symbol, max_spread, ema_period):
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        worker.signals.update_status.emit(f"Scalper: Symbol '{symbol}' not found!", "red")
        return
    if symbol_info.spread > max_spread: worker.signals.update_status.emit(f"Scalper: Spread too high ({symbol_info.spread}). Waiting...", "orange"); return
    
    # 1. H1 Trend Filter
    h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 205)
    h1_trend_bullish = True; h1_trend_bearish = True 
    h1_ema200 = 0.0
    
    if h1_rates is not None and len(h1_rates) >= 200:
        h1_close = np.array([r['close'] for r in h1_rates])
        h1_ema200 = calculate_ema(h1_close, 200)
        if h1_ema200:
            current_price_h1 = h1_close[-1]
            h1_trend_bullish = current_price_h1 > h1_ema200
            h1_trend_bearish = current_price_h1 < h1_ema200
    
    # 2. M5 Logic
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 250) 
    if rates is None or len(rates) < ema_period + 1: return
    
    close_prices = np.array([r['close'] for r in rates])
    high_prices = np.array([r['high'] for r in rates])
    low_prices = np.array([r['low'] for r in rates])
    open_prices = np.array([r['open'] for r in rates])

    rsi_values = calculate_rsi(close_prices, 14)
    adx_val = calculate_adx(high_prices, low_prices, close_prices, 14)
    stoch_k, stoch_d = calculate_stochastic(high_prices, low_prices, close_prices, 5, 3, 3)
    
    if len(rsi_values) < 2 or adx_val is None or stoch_k is None: return
    
    current_rsi, prev_rsi = rsi_values[-1], rsi_values[-2]
    current_price = close_prices[-1]
    ema_val = calculate_ema(close_prices, ema_period)
    
    # Candle Color
    is_bullish_candle = close_prices[-1] > open_prices[-1]
    is_bearish_candle = close_prices[-1] < open_prices[-1]
    
    if ema_val is None:
        worker.signals.update_status.emit(f"Scalper: Not enough data for EMA{ema_period}.", "orange")
        return
        
    worker.signals.update_status.emit(f"Scalper: Price={current_price:.2f} | H1 EMA={h1_ema200:.2f} | ADX={adx_val:.1f} | Stoch K/D={stoch_k:.1f}/{stoch_d:.1f}", "cyan")
    
    # Logic: Trend (H1 & M5) + Volatility (ADX) + Momentum (RSI Dip) + Trigger (Stoch + Candle)
    if h1_trend_bullish and current_price > ema_val and adx_val > 20:
        # RSI Dip Buy + Stoch Cross Up + Bullish Candle
        if prev_rsi < 30 and current_rsi >= 30:
            if stoch_k < 80 and stoch_k > stoch_d:
                if is_bullish_candle and worker.last_trade_action != "Buy":
                    worker.last_trade_action = "Buy"; worker.execute_trade("Buy", is_auto=True)
            
    elif h1_trend_bearish and current_price < ema_val and adx_val > 20:
        # RSI Peak Sell + Stoch Cross Down + Bearish Candle
        if prev_rsi > 70 and current_rsi <= 70:
            if stoch_k > 20 and stoch_k < stoch_d:
                if is_bearish_candle and worker.last_trade_action != "Sell":
                    worker.last_trade_action = "Sell"; worker.execute_trade("Sell", is_auto=True)
