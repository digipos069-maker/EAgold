import numpy as np
import MetaTrader5 as mt5
from indicators import calculate_ema

def run_ict_gold_scalping_logic(worker, symbol, timeframe):
    # --- Strategy Parameters ---
    LOOKBACK_PERIOD = 40 # Number of candles to analyze for setups
    SWING_POINT_LOOKBACK = 5 # How many candles to look left and right for a swing point

    # --- Data Fetching ---
    # Higher timeframe for bias
    h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 201)
    # Lower timeframe for execution
    ltf_rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, LOOKBACK_PERIOD + SWING_POINT_LOOKBACK)

    if h1_rates is None or ltf_rates is None or len(h1_rates) < 201 or len(ltf_rates) < LOOKBACK_PERIOD:
        worker.signals.update_status.emit(f"ICT Scalp: Not enough data.", "orange")
        return

    h1_close = np.array([r['close'] for r in h1_rates])
    ltf_high = np.array([r['high'] for r in ltf_rates])
    ltf_low = np.array([r['low'] for r in ltf_rates])
    ltf_close = np.array([r['close'] for r in ltf_rates])
    current_price = ltf_close[-1]

    # --- HTF Bias ---
    h1_ema200 = calculate_ema(h1_close, 200)
    if h1_ema200 is None:
        worker.signals.update_status.emit(f"ICT Scalp: Cannot calculate H1 EMA.", "orange")
        return
    
    is_bullish_bias = current_price > h1_ema200
    is_bearish_bias = current_price < h1_ema200
    
    status_msg = f"ICT Scalp: Price={current_price:.2f} | H1 EMA={h1_ema200:.2f} | Bias={'Bullish' if is_bullish_bias else 'Bearish'}"
    worker.signals.update_status.emit(status_msg, "cyan")

    # --- Helper function to find swing points ---
    def find_swing_points(highs, lows, lookback):
        swings = []
        # Corrected loop range.
        # It must stop 'lookback' candles from the end to prevent index out of bounds on the forward-looking check.
        for i in range(len(highs) - lookback - 1, lookback - 1, -1):
            # Swing High: high at index i is highest in window
            is_swing_high = all(highs[i] > highs[i-k] for k in range(1, lookback + 1)) and \
                            all(highs[i] > highs[i+k] for k in range(1, lookback + 1))
            # Swing Low: low at index i is lowest in window
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

    # --- Core ICT Logic: Look for (1) Liquidity Grab -> (2) MSS -> (3) Retracement to FVG ---
    # This logic iterates backwards from the most recent swing point
    for i in range(len(swing_points) - 1):
        recent_swing = swing_points[i]
        prev_swing = swing_points[i+1]

        # === Potential Bearish Setup (Sell) ===
        # Condition 1: Must have bearish bias. Recent swing must be a high, previous must be a low.
        if is_bearish_bias and recent_swing['type'] == 'high' and prev_swing['type'] == 'low':
            liquidity_grab_high = recent_swing['price']
            structure_low_to_break = prev_swing['price']
            
            # Condition 2: Check for MSS. Has price broken below the previous swing low?
            # We check candles from the liquidity grab high up to the current candle
            mss_confirmed = False
            for j in range(recent_swing['index'], len(ltf_low)):
                if ltf_low[j] < structure_low_to_break:
                    mss_confirmed = True
                    break
            
            if mss_confirmed:
                # Condition 3: Find FVG created during the MSS move
                # Look for FVG between the liquidity grab and the break of structure
                for k in range(recent_swing['index'], len(ltf_close) - 2):
                    # Bearish FVG: low of candle k-1 is higher than high of candle k+1
                    if ltf_low[k-1] > ltf_high[k+1]:
                        fvg_top = ltf_low[k-1]
                        fvg_bottom = ltf_high[k+1]
                        
                        # Condition 4: Check if current price has retraced into the FVG
                        if fvg_bottom <= current_price <= fvg_top and worker.last_trade_action != "Sell":
                            worker.signals.update_status.emit(f"ICT Scalp: Bearish FVG entry found at {current_price:.2f}", "yellow")
                            worker.last_trade_action = "Sell"
                            # Set SL above the liquidity grab high
                            stop_loss_price = liquidity_grab_high + 0.5 
                            worker.execute_trade("Sell", is_auto=True, sl_price=stop_loss_price)
                            return # Exit after finding a trade

        # === Potential Bullish Setup (Buy) ===
        # Condition 1: Must have bullish bias. Recent swing must be a low, previous must be a high.
        if is_bullish_bias and recent_swing['type'] == 'low' and prev_swing['type'] == 'high':
            liquidity_grab_low = recent_swing['price']
            structure_high_to_break = prev_swing['price']

        # Condition 2: Check for MSS. Has price broken above the previous swing high?
            mss_confirmed = False
            for j in range(recent_swing['index'], len(ltf_high)):
                if ltf_high[j] > structure_high_to_break:
                    mss_confirmed = True
                    break

            if mss_confirmed:
                # Condition 3: Find FVG created during the MSS move
                for k in range(recent_swing['index'], len(ltf_close) - 2):
                    # Bullish FVG: high of k-1 is lower than low of k+1
                    if ltf_high[k-1] < ltf_low[k+1]:
                        fvg_top = ltf_low[k+1]
                        fvg_bottom = ltf_high[k-1]
                        
                        # Condition 4: Check if current price has retraced into the FVG
                        if fvg_bottom <= current_price <= fvg_top and worker.last_trade_action != "Buy":
                            worker.signals.update_status.emit(f"ICT Scalp: Bullish FVG entry found at {current_price:.2f}", "yellow")
                            worker.last_trade_action = "Buy"
                            # Set SL below the liquidity grab low
                            stop_loss_price = liquidity_grab_low - 0.5
                            worker.execute_trade("Buy", is_auto=True, sl_price=stop_loss_price)
                            return # Exit after finding a trade
    return # No setup found
