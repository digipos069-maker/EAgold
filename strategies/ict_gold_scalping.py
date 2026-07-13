import numpy as np
import MetaTrader5 as mt5
from indicators import calculate_ema

def calculate_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return 0.0
    
    tr_list = []
    for i in range(1, len(closes)):
        h = highs[i]
        l = lows[i]
        pc = closes[i-1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_list.append(tr)
    
    # Simple moving average of TR
    atr = np.mean(tr_list[-period:])
    return float(atr)

def run_ict_gold_scalping_logic(worker, symbol, timeframe):
    # --- Strategy Parameters ---
    LOOKBACK_PERIOD = 40 # Number of candles to analyze for setups
    SWING_POINT_LOOKBACK = 5 # How many candles to look left and right for a swing point
    
    # Retrieve RR ratio from user config
    params = getattr(worker, "strategy_params", {})
    # Default RR ratio is 2.0 (1:2). User can configure via param2 in settings
    rr_ratio = params.get("param2", 2.0)
    try:
        rr_ratio = float(rr_ratio)
    except:
        rr_ratio = 2.0

    # --- Data Fetching ---
    # Higher timeframe for bias
    h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 201)
    # Lower timeframe for execution
    ltf_rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, LOOKBACK_PERIOD + SWING_POINT_LOOKBACK + 15)

    if h1_rates is None or ltf_rates is None or len(h1_rates) < 201 or len(ltf_rates) < LOOKBACK_PERIOD:
        worker.signals.update_status.emit(f"ICT Scalp: Not enough data.", "orange")
        return

    h1_close = np.array([r['close'] for r in h1_rates])
    ltf_high = np.array([r['high'] for r in ltf_rates])
    ltf_low = np.array([r['low'] for r in ltf_rates])
    ltf_close = np.array([r['close'] for r in ltf_rates])
    current_price = ltf_close[-1]

    # Calculate ATR for dynamic SL
    atr = calculate_atr(ltf_high, ltf_low, ltf_close, 14)
    if atr == 0.0:
        # Fallback if ATR calculation fails
        atr = 0.5

    # --- HTF Bias ---
    h1_ema200 = calculate_ema(h1_close, 200)
    if h1_ema200 is None:
        worker.signals.update_status.emit(f"ICT Scalp: Cannot calculate H1 EMA.", "orange")
        return
    
    is_bullish_bias = current_price > h1_ema200
    is_bearish_bias = current_price < h1_ema200
    
    status_msg = f"ICT Scalp: Price={current_price:.2f} | Bias={'Bullish' if is_bullish_bias else 'Bearish'}"
    worker.signals.update_status.emit(status_msg, "cyan")

    # --- Helper function to find swing points ---
    def find_swing_points(highs, lows, lookback):
        swings = []
        for i in range(len(highs) - lookback - 1, lookback - 1, -1):
            is_swing_high = all(highs[i] > highs[i-k] for k in range(1, lookback + 1)) and \
                            all(highs[i] > highs[i+k] for k in range(1, lookback + 1))
            is_swing_low = all(lows[i] < lows[i-k] for k in range(1, lookback + 1)) and \
                           all(lows[i] < lows[i+k] for k in range(1, lookback + 1))
            
            if is_swing_high:
                swings.append({'type': 'high', 'price': highs[i], 'index': i})
            elif is_swing_low:
                swings.append({'type': 'low', 'price': lows[i], 'index': i})
        return swings

    swing_points = find_swing_points(ltf_high, ltf_low, SWING_POINT_LOOKBACK)
    if len(swing_points) < 2:
        worker.signals.update_status.emit(f"{status_msg} | Waiting for market structure...", "cyan")
        return

    # Buffer for SL based on ATR. 0.5 * ATR is a good dynamic buffer for Gold scalping.
    sl_buffer = atr * 0.5

    # --- Core ICT Logic ---
    for i in range(len(swing_points) - 1):
        recent_swing = swing_points[i]
        prev_swing = swing_points[i+1]

        # === Potential Bearish Setup (Sell) ===
        if is_bearish_bias and recent_swing['type'] == 'high' and prev_swing['type'] == 'low':
            liquidity_grab_high = recent_swing['price']
            structure_low_to_break = prev_swing['price']
            
            mss_confirmed = False
            for j in range(recent_swing['index'], len(ltf_low)):
                if ltf_low[j] < structure_low_to_break:
                    mss_confirmed = True
                    break
            
            if mss_confirmed:
                for k in range(recent_swing['index'], len(ltf_close) - 2):
                    if ltf_low[k-1] > ltf_high[k+1]:
                        fvg_top = ltf_low[k-1]
                        fvg_bottom = ltf_high[k+1]
                        
                        # Check FVG Invalidation (has price closed above the FVG top after formation?)
                        invalidated = False
                        for p in range(k+2, len(ltf_close) - 1):
                            if ltf_close[p] > fvg_top:
                                invalidated = True
                                break
                        
                        if not invalidated and fvg_bottom <= current_price <= fvg_top and worker.last_trade_action != "Sell":
                            stop_loss_price = liquidity_grab_high + sl_buffer
                            risk_amount = stop_loss_price - current_price
                            take_profit_price = current_price - (risk_amount * rr_ratio)
                            
                            worker.signals.update_status.emit(f"ICT Scalp: Bearish FVG entry at {current_price:.2f} | SL={stop_loss_price:.2f} | TP={take_profit_price:.2f}", "yellow")
                            worker.last_trade_action = "Sell"
                            worker.execute_trade("Sell", is_auto=True, sl_price=stop_loss_price, tp_price=take_profit_price)
                            return

        # === Potential Bullish Setup (Buy) ===
        if is_bullish_bias and recent_swing['type'] == 'low' and prev_swing['type'] == 'high':
            liquidity_grab_low = recent_swing['price']
            structure_high_to_break = prev_swing['price']

            mss_confirmed = False
            for j in range(recent_swing['index'], len(ltf_high)):
                if ltf_high[j] > structure_high_to_break:
                    mss_confirmed = True
                    break

            if mss_confirmed:
                for k in range(recent_swing['index'], len(ltf_close) - 2):
                    if ltf_high[k-1] < ltf_low[k+1]:
                        fvg_top = ltf_low[k+1]
                        fvg_bottom = ltf_high[k-1]
                        
                        # Check FVG Invalidation (has price closed below the FVG bottom after formation?)
                        invalidated = False
                        for p in range(k+2, len(ltf_close) - 1):
                            if ltf_close[p] < fvg_bottom:
                                invalidated = True
                                break
                        
                        if not invalidated and fvg_bottom <= current_price <= fvg_top and worker.last_trade_action != "Buy":
                            stop_loss_price = liquidity_grab_low - sl_buffer
                            risk_amount = current_price - stop_loss_price
                            take_profit_price = current_price + (risk_amount * rr_ratio)
                            
                            worker.signals.update_status.emit(f"ICT Scalp: Bullish FVG entry at {current_price:.2f} | SL={stop_loss_price:.2f} | TP={take_profit_price:.2f}", "yellow")
                            worker.last_trade_action = "Buy"
                            worker.execute_trade("Buy", is_auto=True, sl_price=stop_loss_price, tp_price=take_profit_price)
                            return
    return
