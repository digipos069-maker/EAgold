import numpy as np
import MetaTrader5 as mt5

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
    
    atr = np.mean(tr_list[-period:])
    return float(atr)

def calculate_volume(symbol, sl_distance, risk_percent):
    try:
        account = mt5.account_info()
        symbol_info = mt5.symbol_info(symbol)
        if not account or not symbol_info or sl_distance <= 0: return 0.0

        risk_amount = account.balance * (risk_percent / 100.0)
        contract_size = symbol_info.trade_contract_size
        if contract_size <= 0: return 0.0

        raw_volume = risk_amount / (sl_distance * contract_size)
        step = symbol_info.volume_step or 0.01
        volume = round(raw_volume / step) * step
        volume = max(symbol_info.volume_min, min(symbol_info.volume_max, volume))
        return round(volume, 2)
    except Exception:
        return 0.0

def run_ict_gold_scalping_logic(worker, symbol, timeframe):
    # --- Strategy Parameters ---
    LOOKBACK_PERIOD = 40 
    SWING_POINT_LOOKBACK = 5 
    
    params = getattr(worker, "strategy_params", {})
    
    try:
        risk_percent = float(params.get("param1", 1.0))
    except:
        risk_percent = 1.0
        
    try:
        rr_ratio = float(params.get("param2", 2.0))
    except:
        rr_ratio = 2.0

    try:
        tp_mode = int(params.get("param3", 0)) # 0=RR, 1=Struct
    except:
        tp_mode = 0

    # --- Dual Data Fetching ---
    ltf_rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, LOOKBACK_PERIOD + SWING_POINT_LOOKBACK + 15)
    # Fetch enough M1 data to cover the higher timeframe lookback window
    m1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 3000)

    if ltf_rates is None or len(ltf_rates) < LOOKBACK_PERIOD or m1_rates is None or len(m1_rates) < 100:
        worker.signals.update_status.emit(f"ICT Scalp: Not enough data.", "orange")
        return

    ltf_open = np.array([r['open'] for r in ltf_rates])
    ltf_high = np.array([r['high'] for r in ltf_rates])
    ltf_low = np.array([r['low'] for r in ltf_rates])
    ltf_close = np.array([r['close'] for r in ltf_rates])
    
    m1_high_full = np.array([r['high'] for r in m1_rates])
    m1_low_full = np.array([r['low'] for r in m1_rates])
    m1_close_full = np.array([r['close'] for r in m1_rates])
    current_price = m1_close_full[-1]

    # ATR calculated on M1 for sniper stop loss padding
    atr = calculate_atr(m1_high_full, m1_low_full, m1_close_full, 14)
    if atr == 0.0:
        atr = 0.5
    sl_buffer = atr * 0.5

    status_msg = f"ICT Scalp: Price={current_price:.2f} | Scanning Structure..."

    # --- Helper function to find swing points ---
    def find_swing_points(rates_high, rates_low, lookback):
        swings = []
        for i in range(len(rates_high) - lookback - 1, lookback - 1, -1):
            is_swing_high = all(rates_high[i] > rates_high[i-k] for k in range(1, lookback + 1)) and \
                            all(rates_high[i] > rates_high[i+k] for k in range(1, lookback + 1))
            is_swing_low = all(rates_low[i] < rates_low[i-k] for k in range(1, lookback + 1)) and \
                           all(rates_low[i] < rates_low[i+k] for k in range(1, lookback + 1))
            
            if is_swing_high:
                swings.append({'type': 'high', 'price': rates_high[i], 'index': i})
            elif is_swing_low:
                swings.append({'type': 'low', 'price': rates_low[i], 'index': i})
        return swings

    swing_points = find_swing_points(ltf_high, ltf_low, SWING_POINT_LOOKBACK)
    if len(swing_points) < 4:
        worker.signals.update_status.emit(f"{status_msg} | Waiting for Market Structure", "cyan")
        return

    # --- Bias Detection ---
    # swing_points[0] is the most recent
    highs = [s for s in swing_points if s['type'] == 'high']
    lows = [s for s in swing_points if s['type'] == 'low']
    
    htf_bias = "Neutral"
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[0]['price'] > highs[1]['price'] and lows[0]['price'] >= lows[1]['price']:
            htf_bias = "Bullish"
        elif lows[0]['price'] < lows[1]['price'] and highs[0]['price'] <= highs[1]['price']:
            htf_bias = "Bearish"

    if htf_bias == "Neutral":
        worker.signals.update_status.emit(f"{status_msg} | HTF Neutral Bias", "cyan")
        return

    worker.signals.update_status.emit(f"{status_msg} | HTF Bias: {htf_bias}", "cyan")

    # --- Core MTF ICT Logic ---
    for i in range(len(swing_points) - 1):
        recent_swing = swing_points[i]
        prev_swing = swing_points[i+1]

        # === Potential Bearish Setup (Sell) ===
        if htf_bias == "Bearish" and recent_swing['type'] == 'high' and prev_swing['type'] == 'low':
            
            # 1. Buy-side Liquidity Sweep Filter
            older_highs = [h for h in highs if h['index'] < recent_swing['index']]
            swept = any(recent_swing['price'] > h['price'] for h in older_highs)
            
            if swept:
                liquidity_grab_high = recent_swing['price']
                structure_low_to_break = prev_swing['price']
                
                # 2. MSS with Displacement Filter
                mss_confirmed = False
                for j in range(recent_swing['index'], len(ltf_close)):
                    if ltf_close[j] < structure_low_to_break:
                        candle_range = ltf_high[j] - ltf_low[j]
                        body = abs(ltf_close[j] - ltf_open[j])
                        displacement = body > (candle_range * 0.5) if candle_range > 0 else False
                        
                        if displacement:
                            mss_confirmed = True
                            break
                
                if mss_confirmed:
                    grab_time = ltf_rates[recent_swing['index']]['time']
                    
                    # Switch to M1 for FVG
                    m1_subset = [r for r in m1_rates if r['time'] >= grab_time]
                    if len(m1_subset) < 3:
                        continue
                    
                    m1_sub_high = np.array([r['high'] for r in m1_subset])
                    m1_sub_low = np.array([r['low'] for r in m1_subset])
                    m1_sub_close = np.array([r['close'] for r in m1_subset])
                    
                    for k in range(1, len(m1_sub_close) - 2):
                        if m1_sub_low[k-1] > m1_sub_high[k+1]:
                            fvg_top = m1_sub_low[k-1]
                            fvg_bottom = m1_sub_high[k+1]
                            
                            invalidated = False
                            for p in range(k+2, len(m1_sub_close) - 1):
                                if m1_sub_close[p] > fvg_top:
                                    invalidated = True
                                    break
                            
                            if not invalidated and fvg_bottom <= current_price <= fvg_top and worker.last_trade_action != "Sell":
                                stop_loss_price = liquidity_grab_high + sl_buffer
                                risk_amount = stop_loss_price - current_price
                                
                                # TP Calculation
                                if tp_mode == 1: # Struct
                                    target_lows = [l['price'] for l in lows if l['price'] < current_price]
                                    if target_lows:
                                        take_profit_price = max(target_lows) # Nearest sell-side liquidity
                                    else:
                                        take_profit_price = current_price - (risk_amount * rr_ratio) # Fallback
                                else: # RR
                                    take_profit_price = current_price - (risk_amount * rr_ratio)
                                
                                volume = calculate_volume(symbol, risk_amount, risk_percent)
                                if volume <= 0:
                                    worker.signals.update_status.emit(f"Volume calculation failed.", "red")
                                    return
                                    
                                worker.signals.update_status.emit(f"ICT MTF: Bearish M1 FVG entry at {current_price:.2f} | SL={stop_loss_price:.2f} | TP={take_profit_price:.2f}", "yellow")
                                worker.last_trade_action = "Sell"
                                worker.execute_trade("Sell", is_auto=True, sl_price=stop_loss_price, tp_price=take_profit_price, volume=volume)
                                return

        # === Potential Bullish Setup (Buy) ===
        if htf_bias == "Bullish" and recent_swing['type'] == 'low' and prev_swing['type'] == 'high':
            
            # 1. Sell-side Liquidity Sweep Filter
            older_lows = [l for l in lows if l['index'] < recent_swing['index']]
            swept = any(recent_swing['price'] < l['price'] for l in older_lows)
            
            if swept:
                liquidity_grab_low = recent_swing['price']
                structure_high_to_break = prev_swing['price']
                
                # 2. MSS with Displacement Filter
                mss_confirmed = False
                for j in range(recent_swing['index'], len(ltf_close)):
                    if ltf_close[j] > structure_high_to_break:
                        candle_range = ltf_high[j] - ltf_low[j]
                        body = abs(ltf_close[j] - ltf_open[j])
                        displacement = body > (candle_range * 0.5) if candle_range > 0 else False
                        
                        if displacement:
                            mss_confirmed = True
                            break
                
                if mss_confirmed:
                    grab_time = ltf_rates[recent_swing['index']]['time']
                    
                    # Switch to M1
                    m1_subset = [r for r in m1_rates if r['time'] >= grab_time]
                    if len(m1_subset) < 3:
                        continue
                    
                    m1_sub_high = np.array([r['high'] for r in m1_subset])
                    m1_sub_low = np.array([r['low'] for r in m1_subset])
                    m1_sub_close = np.array([r['close'] for r in m1_subset])
                    
                    for k in range(1, len(m1_sub_close) - 2):
                        if m1_sub_high[k-1] < m1_sub_low[k+1]:
                            fvg_top = m1_sub_low[k+1]
                            fvg_bottom = m1_sub_high[k-1]
                            
                            invalidated = False
                            for p in range(k+2, len(m1_sub_close) - 1):
                                if m1_sub_close[p] < fvg_bottom:
                                    invalidated = True
                                    break
                            
                            if not invalidated and fvg_bottom <= current_price <= fvg_top and worker.last_trade_action != "Buy":
                                stop_loss_price = liquidity_grab_low - sl_buffer
                                risk_amount = current_price - stop_loss_price
                                
                                # TP Calculation
                                if tp_mode == 1: # Struct
                                    target_highs = [h['price'] for h in highs if h['price'] > current_price]
                                    if target_highs:
                                        take_profit_price = min(target_highs) # Nearest buy-side liquidity
                                    else:
                                        take_profit_price = current_price + (risk_amount * rr_ratio) # Fallback
                                else: # RR
                                    take_profit_price = current_price + (risk_amount * rr_ratio)
                                
                                volume = calculate_volume(symbol, risk_amount, risk_percent)
                                if volume <= 0:
                                    worker.signals.update_status.emit(f"Volume calculation failed.", "red")
                                    return
                                    
                                worker.signals.update_status.emit(f"ICT MTF: Bullish M1 FVG entry at {current_price:.2f} | SL={stop_loss_price:.2f} | TP={take_profit_price:.2f}", "yellow")
                                worker.last_trade_action = "Buy"
                                worker.execute_trade("Buy", is_auto=True, sl_price=stop_loss_price, tp_price=take_profit_price, volume=volume)
                                return
    return
