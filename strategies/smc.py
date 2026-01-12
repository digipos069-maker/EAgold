import numpy as np
import MetaTrader5 as mt5

def find_swings(highs, lows, lookback=5):
    swings = []
    for i in range(lookback, len(highs) - lookback):
        is_high = all(highs[i] > highs[i-k] for k in range(1, lookback+1)) and \
                  all(highs[i] > highs[i+k] for k in range(1, lookback+1))
        is_low = all(lows[i] < lows[i-k] for k in range(1, lookback+1)) and \
                 all(lows[i] < lows[i+k] for k in range(1, lookback+1))
        if is_high: swings.append({'type': 'high', 'price': highs[i], 'index': i})
        elif is_low: swings.append({'type': 'low', 'price': lows[i], 'index': i})
    return swings

def get_structure(swings):
    if len(swings) < 4: return "Uncertain"
    # Check last 2 highs and lows
    highs = [s for s in swings if s['type'] == 'high']
    lows = [s for s in swings if s['type'] == 'low']
    if len(highs) < 2 or len(lows) < 2: return "Uncertain"
    
    if highs[-1]['price'] > highs[-2]['price'] and lows[-1]['price'] > lows[-2]['price']:
        return "Bullish"
    if highs[-1]['price'] < highs[-2]['price'] and lows[-1]['price'] < lows[-2]['price']:
        return "Bearish"
    return "Ranging"

def run_smc_logic(worker, symbol, timeframe, risk_percent, rr_ratio):
    # Default params if not set
    if not risk_percent: risk_percent = 1
    if not rr_ratio: rr_ratio = 2
    
    # 1. HTF Bias (H1)
    h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
    if h1_rates is None or len(h1_rates) < 100: return
    h1_high = np.array([r['high'] for r in h1_rates])
    h1_low = np.array([r['low'] for r in h1_rates])
    h1_swings = find_swings(h1_high, h1_low)
    bias = get_structure(h1_swings)
    
    # 2. LTF Structure & Entry (Current Timeframe)
    ltf_rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
    if ltf_rates is None or len(ltf_rates) < 100: return
    
    ltf_open = np.array([r['open'] for r in ltf_rates])
    ltf_close = np.array([r['close'] for r in ltf_rates])
    ltf_high = np.array([r['high'] for r in ltf_rates])
    ltf_low = np.array([r['low'] for r in ltf_rates])
    
    current_price = ltf_close[-1]
    
    worker.signals.update_status.emit(f"SMC: Price={current_price:.2f} | Bias={bias}", "cyan")
    
    signal = None
    stop_loss = 0.0
    
    # Look for FVG/OB in recent candles (e.g., last 10)
    # Entry Condition: Bias matches + Tap into FVG/OB
    
    if bias == "Bullish":
        # Find Bullish FVG or OB below current price
        for i in range(len(ltf_rates)-2, len(ltf_rates)-15, -1):
            # Bullish FVG: High[i-1] < Low[i+1]
            if ltf_high[i-1] < ltf_low[i+1]:
                fvg_top = ltf_low[i+1]
                fvg_bottom = ltf_high[i-1]
                
                if fvg_bottom <= current_price <= fvg_top:
                    if ltf_close[-1] > ltf_open[-1]:
                        signal = "Buy"
                        stop_loss = fvg_bottom - (ltf_high[i-1] - ltf_low[i-1])*0.5 # Below FVG
                        break
                        
        # Check OB if no FVG signal
        if not signal:
             for i in range(len(ltf_rates)-3, len(ltf_rates)-15, -1):
                # Bullish OB: Bearish candle before strong move up
                if ltf_close[i] < ltf_open[i]: # Bearish
                    # Next candle strong bullish?
                    if ltf_close[i+1] > ltf_open[i+1] and ltf_close[i+1] > ltf_high[i]:
                         ob_top = ltf_high[i]
                         ob_bottom = ltf_low[i]
                         if ob_bottom <= current_price <= ob_top:
                              if ltf_close[-1] > ltf_open[-1]:
                                  signal = "Buy"
                                  stop_loss = ob_bottom - 0.5
                                  break

    elif bias == "Bearish":
        # Find Bearish FVG or OB above current price
        for i in range(len(ltf_rates)-2, len(ltf_rates)-15, -1):
            # Bearish FVG: Low[i-1] > High[i+1]
            if ltf_low[i-1] > ltf_high[i+1]:
                fvg_top = ltf_low[i-1]
                fvg_bottom = ltf_high[i+1]
                
                if fvg_bottom <= current_price <= fvg_top:
                    if ltf_close[-1] < ltf_open[-1]:
                        signal = "Sell"
                        stop_loss = fvg_top + (ltf_high[i-1] - ltf_low[i-1])*0.5
                        break
        
        if not signal:
            for i in range(len(ltf_rates)-3, len(ltf_rates)-15, -1):
                # Bearish OB: Bullish candle before strong move down
                if ltf_close[i] > ltf_open[i]:
                    if ltf_close[i+1] < ltf_open[i+1] and ltf_close[i+1] < ltf_low[i]:
                         ob_top = ltf_high[i]
                         ob_bottom = ltf_low[i]
                         if ob_bottom <= current_price <= ob_top:
                              if ltf_close[-1] < ltf_open[-1]:
                                  signal = "Sell"
                                  stop_loss = ob_top + 0.5
                                  break
    
    # Execute
    if signal:
        if signal == "Buy" and worker.last_trade_action != "Buy":
            sl_dist = current_price - stop_loss
            if sl_dist <= 0: sl_dist = 0.5 # Safety
            tp_dist = sl_dist * rr_ratio
            tp = current_price + tp_dist
            
            # Calc Volume based on Risk %
            calc_volume = 0.01
            try:
                account = mt5.account_info()
                symbol_info = mt5.symbol_info(symbol)
                if account and symbol_info and sl_dist > 0:
                    risk_amount = (risk_percent / 100.0) * account.balance
                    contract_size = symbol_info.trade_contract_size
                    raw_volume = risk_amount / (sl_dist * contract_size)
                    step = symbol_info.volume_step
                    calc_volume = round(raw_volume / step) * step
                    calc_volume = max(symbol_info.volume_min, min(symbol_info.volume_max, calc_volume))
            except: pass

            worker.last_trade_action = "Buy"
            worker.execute_trade("Buy", is_auto=True, sl_price=stop_loss, tp_price=tp, volume=calc_volume)
            worker.signals.update_status.emit(f"SMC: Buy Signal. Vol={calc_volume:.2f}, RR=1:{rr_ratio}", "green")

        elif signal == "Sell" and worker.last_trade_action != "Sell":
            sl_dist = stop_loss - current_price
            if sl_dist <= 0: sl_dist = 0.5
            tp_dist = sl_dist * rr_ratio
            tp = current_price - tp_dist
            
            # Calc Volume
            calc_volume = 0.01
            try:
                account = mt5.account_info()
                symbol_info = mt5.symbol_info(symbol)
                if account and symbol_info and sl_dist > 0:
                    risk_amount = (risk_percent / 100.0) * account.balance
                    contract_size = symbol_info.trade_contract_size
                    raw_volume = risk_amount / (sl_dist * contract_size)
                    step = symbol_info.volume_step
                    calc_volume = round(raw_volume / step) * step
                    calc_volume = max(symbol_info.volume_min, min(symbol_info.volume_max, calc_volume))
            except: pass

            worker.last_trade_action = "Sell"
            worker.execute_trade("Sell", is_auto=True, sl_price=stop_loss, tp_price=tp, volume=calc_volume)
            worker.signals.update_status.emit(f"SMC: Sell Signal. Vol={calc_volume:.2f}, RR=1:{rr_ratio}", "red")
