import numpy as np
import MetaTrader5 as mt5
from indicators import calculate_ema, calculate_macd, calculate_adx, calculate_atr, calculate_rsi, calculate_stochastic

def manage_scalping_trades(worker, symbol, timeframe):
    # Calculate dynamic ATR for management
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 20)
    if rates is None or len(rates) < 15: return
    
    high = np.array([r['high'] for r in rates])
    low = np.array([r['low'] for r in rates])
    close = np.array([r['close'] for r in rates])
    current_atr = calculate_atr(high, low, close, 14)
    if not current_atr: return

    open_positions = mt5.positions_get(symbol=symbol, magic=234000)
    if not open_positions: return

    tick = mt5.symbol_info_tick(symbol)
    if not tick: return

    for pos in open_positions:
        # 1. Break-Even Logic
        # If profit > 0.8 * ATR, move SL to Entry (+ small buffer)
        be_trigger = 0.8 * current_atr
        be_buffer = 0.1 * current_atr # Small profit to cover swap/commissions
        
        # 2. Trailing Stop Logic
        # If profit > 1.5 * ATR, trail by 1.0 * ATR
        trail_trigger = 1.5 * current_atr
        trail_dist = 1.0 * current_atr

        if pos.type == mt5.ORDER_TYPE_BUY:
            current_profit_dist = tick.bid - pos.price_open
            
            # Check Break-Even
            new_sl = pos.price_open + be_buffer
            if current_profit_dist > be_trigger and pos.sl < new_sl:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp,
                    "magic": 234000
                }
                mt5.order_send(request)
                worker.signals.update_status.emit(f"Manager: Moved Buy #{pos.ticket} to Breakeven.", "green")

            # Check Trailing
            if current_profit_dist > trail_trigger:
                potential_new_sl = tick.bid - trail_dist
                if potential_new_sl > pos.sl: # Only move SL up
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "sl": potential_new_sl,
                        "tp": pos.tp,
                        "magic": 234000
                    }
                    mt5.order_send(request)
                    worker.signals.update_status.emit(f"Manager: Trailing Buy #{pos.ticket} to {potential_new_sl:.2f}", "green")

        elif pos.type == mt5.ORDER_TYPE_SELL:
            current_profit_dist = pos.price_open - tick.ask
            
            # Check Break-Even
            new_sl = pos.price_open - be_buffer
            if current_profit_dist > be_trigger and (pos.sl == 0 or pos.sl > new_sl):
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp,
                    "magic": 234000
                }
                mt5.order_send(request)
                worker.signals.update_status.emit(f"Manager: Moved Sell #{pos.ticket} to Breakeven.", "green")

            # Check Trailing
            if current_profit_dist > trail_trigger:
                potential_new_sl = tick.ask + trail_dist
                if pos.sl == 0 or potential_new_sl < pos.sl: # Only move SL down
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "sl": potential_new_sl,
                        "tp": pos.tp,
                        "magic": 234000
                    }
                    mt5.order_send(request)
                    worker.signals.update_status.emit(f"Manager: Trailing Sell #{pos.ticket} to {potential_new_sl:.2f}", "green")

def run_gold_scalping_new_logic(worker, symbol):
    # 0. Manage Open Trades (Trailing/Breakeven)
    manage_scalping_trades(worker, symbol, mt5.TIMEFRAME_M15)

    # 1. Fetch Data
    # H1 for Trend Bias (EMA200)
    h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 205)
    # M15 for Entry/Confirmation (EMA50, MACD, ADX, ATR)
    m15_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    # M5 for Momentum (RSI)
    m5_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)

    if h1_rates is None or m15_rates is None or m5_rates is None:
        worker.signals.update_status.emit("Gold Scalping New: Not enough data.", "orange")
        return

    # 2. Indicators
    # H1
    h1_close = np.array([r['close'] for r in h1_rates])
    h1_ema200 = calculate_ema(h1_close, 200)
    if h1_ema200 is None: return

    # M15
    m15_close = np.array([r['close'] for r in m15_rates])
    m15_high = np.array([r['high'] for r in m15_rates])
    m15_low = np.array([r['low'] for r in m15_rates])
    
    m15_ema50 = calculate_ema(m15_close, 50)
    macd_line, signal_line, hist, prev_hist = calculate_macd(m15_close, 12, 26, 9)
    m15_adx = calculate_adx(m15_high, m15_low, m15_close, 14)
    m15_atr = calculate_atr(m15_high, m15_low, m15_close, 14)

    if any(x is None for x in [m15_ema50, macd_line, m15_adx, m15_atr]): return

    # M5
    m5_close = np.array([r['close'] for r in m5_rates])
    m5_high = np.array([r['high'] for r in m5_rates])
    m5_low = np.array([r['low'] for r in m5_rates])
    m5_open = np.array([r['open'] for r in m5_rates])
    
    m5_rsi = calculate_rsi(m5_close, 14)
    m5_stoch_k, m5_stoch_d = calculate_stochastic(m5_high, m5_low, m5_close, 5, 3, 3)
    
    if len(m5_rsi) < 1 or m5_stoch_k is None: return
    current_m5_rsi = m5_rsi[-1]
    
    # Candle Color (Price Action)
    is_bullish_candle = m5_close[-1] > m5_open[-1]
    is_bearish_candle = m5_close[-1] < m5_open[-1]

    current_price = m15_close[-1]

    # 3. Logic
    # Bias
    bias = "Long" if current_price > h1_ema200 else "Short"
    
    # Entry Conditions
    signal = None
    
    worker.signals.update_status.emit(f"Gold New: Price={current_price:.2f} | Bias={bias} | ADX={m15_adx:.1f} | Stoch K/D={m5_stoch_k:.1f}/{m5_stoch_d:.1f}", "cyan")

    if bias == "Long":
        # 1. Price above EMA50 on M15
        if current_price > m15_ema50:
            # 2. MACD Histogram flip positive (now > 0, prev <= 0)
            if hist > 0 and prev_hist <= 0:
                # 3. ADX > 18 (Trend Strength)
                if m15_adx > 18:
                    # 4. RSI M5 in 40-80
                    if 40 <= current_m5_rsi <= 80:
                        # 5. EXPERT: Stochastic < 80 (Not exhausted) AND Crossing UP
                        if m5_stoch_k < 80 and m5_stoch_k > m5_stoch_d:
                            # 6. EXPERT: Bullish Candle Confirmation
                            if is_bullish_candle:
                                signal = "Buy"

    elif bias == "Short":
        # 1. Price below EMA50 on M15
        if current_price < m15_ema50:
            # 2. MACD Histogram flip negative (now < 0, prev >= 0)
            if hist < 0 and prev_hist >= 0:
                    # 3. ADX > 18
                if m15_adx > 18:
                        # 4. RSI M5 in 20-60
                    if 20 <= current_m5_rsi <= 60:
                        # 5. EXPERT: Stochastic > 20 (Not exhausted) AND Crossing DOWN
                        if m5_stoch_k > 20 and m5_stoch_k < m5_stoch_d:
                            # 6. EXPERT: Bearish Candle Confirmation
                            if is_bearish_candle:
                                signal = "Sell"

    # 4. Execution
    if signal:
        # Calculate Volume (1% Risk)
        calc_volume = 0.01
        try:
            account = mt5.account_info()
            symbol_info = mt5.symbol_info(symbol)
            if account and symbol_info:
                risk_per_trade = 0.01 * account.balance
                sl_dist = 1.5 * m15_atr
                contract_size = symbol_info.trade_contract_size
                
                if sl_dist > 0 and contract_size > 0:
                    raw_volume = risk_per_trade / (sl_dist * contract_size)
                    step = symbol_info.volume_step
                    if step > 0:
                        calc_volume = round(raw_volume / step) * step
                        calc_volume = max(symbol_info.volume_min, min(symbol_info.volume_max, calc_volume))
        except Exception as e:
            worker.signals.update_status.emit(f"Volume Calc Error: {e}", "red")

        if signal == "Buy" and worker.last_trade_action != "Buy":
                sl_dist = 1.5 * m15_atr
                tp_dist = 1.0 * m15_atr
                sl_price = current_price - sl_dist
                tp_price = current_price + tp_dist
                worker.last_trade_action = "Buy"
                worker.execute_trade("Buy", is_auto=True, sl_price=sl_price, tp_price=tp_price, volume=calc_volume)
                worker.signals.update_status.emit(f"Gold Scalping New: Buy Signal. Vol={calc_volume:.2f}, SL={sl_price:.2f}", "green")

        elif signal == "Sell" and worker.last_trade_action != "Sell":
                sl_dist = 1.5 * m15_atr
                tp_dist = 1.0 * m15_atr
                sl_price = current_price + sl_dist
                tp_price = current_price - tp_dist
                worker.last_trade_action = "Sell"
                worker.execute_trade("Sell", is_auto=True, sl_price=sl_price, tp_price=tp_price, volume=calc_volume)
                worker.signals.update_status.emit(f"Gold Scalping New: Sell Signal. Vol={calc_volume:.2f}, SL={sl_price:.2f}", "red")
