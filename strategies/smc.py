import MetaTrader5 as mt5
import numpy as np
from datetime import datetime

# --- Constants & Helpers ---
MAX_LOOKBACK = 500  # Candles to analyze
SWING_LOOKBACK = 5  # Candles left/right to define a swing point

class SMCTrader:
    def __init__(self, worker, symbol, timeframe, risk_percent, rr_ratio):
        self.worker = worker
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_percent = float(risk_percent) if risk_percent else 1.0
        self.rr_ratio = float(rr_ratio) if rr_ratio else 2.0
        
    def get_data(self, tf, count=MAX_LOOKBACK):
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, count)
        if rates is None or len(rates) < count:
            return None
        
        # Convert to structured array/DataFrame-like dict for easier access
        data = {
            'time': [datetime.fromtimestamp(x['time']) for x in rates],
            'open': np.array([x['open'] for x in rates]),
            'high': np.array([x['high'] for x in rates]),
            'low': np.array([x['low'] for x in rates]),
            'close': np.array([x['close'] for x in rates]),
            'spread': np.array([x['spread'] for x in rates]),
        }
        return data

    def find_swings(self, data):
        """
        Identifies swing highs and lows.
        Returns a list of dicts: {'index': int, 'price': float, 'type': 'high'|'low'}
        """
        highs = data['high']
        lows = data['low']
        swings = []
        
        for i in range(SWING_LOOKBACK, len(highs) - SWING_LOOKBACK):
            # fractal high
            if all(highs[i] >= highs[i - k] for k in range(1, SWING_LOOKBACK + 1)) and \
               all(highs[i] > highs[i + k] for k in range(1, SWING_LOOKBACK + 1)):
                swings.append({'index': i, 'price': highs[i], 'type': 'high'})
            
            # fractal low
            if all(lows[i] <= lows[i - k] for k in range(1, SWING_LOOKBACK + 1)) and \
               all(lows[i] < lows[i + k] for k in range(1, SWING_LOOKBACK + 1)):
                swings.append({'index': i, 'price': lows[i], 'type': 'low'})
                
        return swings

    def get_market_structure(self, swings):
        """
        Determines trend based on recent Higher Highs/Lows vs Lower Highs/Lows.
        Simple logic: Compare last 2 same-type swings.
        """
        if len(swings) < 4:
            return "Uncertain", None, None # Bias, Last High, Last Low

        highs = [s for s in swings if s['type'] == 'high']
        lows = [s for s in swings if s['type'] == 'low']
        
        if len(highs) < 2 or len(lows) < 2:
            return "Uncertain", None, None

        last_h = highs[-1]
        prev_h = highs[-2]
        last_l = lows[-1]
        prev_l = lows[-2]

        bias = "Ranging"
        # Bullish: HH + HL
        if last_h['price'] > prev_h['price'] and last_l['price'] > prev_l['price']:
            bias = "Bullish"
        # Bearish: LH + LL
        elif last_h['price'] < prev_h['price'] and last_l['price'] < prev_l['price']:
            bias = "Bearish"
            
        return bias, last_h, last_l

    def detect_break_of_structure(self, data, swings, bias):
        """
        Detects if the most recent price action broke structure (BOS/CHoCH).
        Returns: 'BOS', 'CHoCH', or None
        """
        # Look at recent price action relative to last structural swing
        if not swings: return None
        
        last_high = [s for s in swings if s['type'] == 'high'][-1]
        last_low = [s for s in swings if s['type'] == 'low'][-1]
        
        current_close = data['close'][-1]
        
        # CHoCH Detection (Reversal Signs)
        # If Bullish, breaking below last HL is CHoCH
        if bias == "Bullish" and current_close < last_low['price']:
            return "CHoCH_Bearish"
        # If Bearish, breaking above last LH is CHoCH
        if bias == "Bearish" and current_close > last_high['price']:
            return "CHoCH_Bullish"
            
        return None

    def check_liquidity_sweep(self, data, swings, bias):
        """
        Checks if a recent high/low was swept (price poked through but reversed or simply took it out).
        For simplicity in this EA: We look for a recent candle that wicked beyond a Swing Point but closed back inside,
        OR just a clear take out of a swing point followed by reversal structure.
        
        Returns: True/False
        """
        # We need a sweep OPPOSITE to the intended trade direction
        # Buy Setup: Sweep of Sell-side Liquidity (Old Lows)
        # Sell Setup: Sweep of Buy-side Liquidity (Old Highs)
        
        lookback_candles = 20 # Look for sweep in last 20 candles
        current_idx = len(data['close']) - 1
        
        if bias == "Bullish": # Looking for Buy -> Need Sweep of Lows
            recent_lows = [s for s in swings if s['type'] == 'low' and s['index'] < current_idx - 1]
            if not recent_lows: return False
            
            # Check last significant low
            target_low = recent_lows[-1]['price']
            
            # Did we trade below it recently?
            lowest_recent = np.min(data['low'][current_idx-lookback_candles : current_idx])
            if lowest_recent < target_low:
                return True # Liquidity taken
                
        elif bias == "Bearish": # Looking for Sell -> Need Sweep of Highs
            recent_highs = [s for s in swings if s['type'] == 'high' and s['index'] < current_idx - 1]
            if not recent_highs: return False
            
            target_high = recent_highs[-1]['price']
            
            highest_recent = np.max(data['high'][current_idx-lookback_candles : current_idx])
            if highest_recent > target_high:
                return True # Liquidity taken
                
        return False

    def get_premium_discount(self, data, swings, bias):
        """
        Returns True if price is in Discount (for Buy) or Premium (for Sell).
        Uses range between Last Major High and Last Major Low.
        """
        if len(swings) < 2: return False
        
        # Simple Range: Max High to Min Low of recent structure
        # In a real impulse leg, we take the low that started the move to the high.
        
        last_high = [s for s in swings if s['type'] == 'high'][-1]['price']
        last_low = [s for s in swings if s['type'] == 'low'][-1]['price']
        
        current_price = data['close'][-1]
        r = last_high - last_low
        if r == 0: return False
        
        fib_level = (current_price - last_low) / r
        
        if bias == "Bullish":
            # Buy in Discount (< 0.5)
            return fib_level < 0.5
        elif bias == "Bearish":
            # Sell in Premium (> 0.5)
            return fib_level > 0.5
            
        return False

    def find_fvg(self, data, bias):
        """
        Finds the nearest VALID FVG to current price.
        Returns: (top, bottom, index) or None
        """
        # Scan backwards for unmitigated FVG
        l = len(data['close'])
        
        for i in range(l - 2, l - 50, -1):
            if bias == "Bullish":
                # Candle i is the FVG candle. Gap between High[i-1] and Low[i+1]?
                # Note: Array is chronological. i is "current" in loop. 
                # Gap is between i-1 (left) and i+1 (right)??? 
                # Standard FVG: Candle 1 High < Candle 3 Low. Gap is 1 High to 3 Low.
                # Let's map indices: 1=i-1, 2=i, 3=i+1.
                # Actually, if we loop backwards, 'i' is the middle candle.
                
                c1_high = data['high'][i-1]
                c3_low = data['low'][i+1]
                
                if c1_high < c3_low: # Gap exists
                    # Check mitigation
                    # Has price since 'i+1' traded into this zone?
                    # Zone: c1_high to c3_low.
                    # We only care if we are currently potentially IN or NEAR it.
                    return (c3_low, c1_high, i)
                    
            elif bias == "Bearish":
                # Candle 1 Low > Candle 3 High.
                c1_low = data['low'][i-1]
                c3_high = data['high'][i+1]
                
                if c1_low > c3_high:
                    return (c1_low, c3_high, i)
        
        return None

    def find_ob(self, data, bias):
        """
        Finds nearest Order Block.
        Bullish OB: Last Bearish candle before strong up move.
        Bearish OB: Last Bullish candle before strong down move.
        """
        l = len(data['close'])
        
        for i in range(l - 3, l - 50, -1):
            if bias == "Bullish":
                # Look for Bearish Candle
                if data['close'][i] < data['open'][i]:
                    # Check for strong displacement after (next candle Bullish and engulfing or strong)
                    if data['close'][i+1] > data['open'][i+1] and data['close'][i+1] > data['high'][i]:
                        return (data['high'][i], data['low'][i], i)
                        
            elif bias == "Bearish":
                # Look for Bullish Candle
                if data['close'][i] > data['open'][i]:
                    # Check for strong displacement down
                    if data['close'][i+1] < data['open'][i+1] and data['close'][i+1] < data['low'][i]:
                        return (data['high'][i], data['low'][i], i)
        return None

    def execute(self):
        # 1. HTF Analysis (H1) for Bias
        h1_data = self.get_data(mt5.TIMEFRAME_H1, 200)
        if not h1_data: return
        h1_swings = self.find_swings(h1_data)
        htf_bias, _, _ = self.get_market_structure(h1_swings)
        
        if htf_bias == "Uncertain" or htf_bias == "Ranging":
            self.worker.signals.update_status.emit(f"SMC: HTF Bias Uncertain ({htf_bias})", "orange")
            return

        # 2. LTF Analysis (Entry Timeframe)
        ltf_data = self.get_data(self.timeframe, 200)
        if not ltf_data: return
        ltf_swings = self.find_swings(ltf_data)
        
        # 3. Check Liquidity Sweep (Has liquidity been taken recently?)
        # For a Buy, we want to see Sell-side liquidity (Lows) swept.
        liquidity_swept = self.check_liquidity_sweep(ltf_data, ltf_swings, htf_bias)
        
        # 4. Check Premium/Discount
        in_zone = self.get_premium_discount(ltf_data, ltf_swings, htf_bias)
        
        current_price = ltf_data['close'][-1]
        self.worker.signals.update_status.emit(f"SMC: Bias={htf_bias} | LiqSweep={liquidity_swept} | Zone={in_zone}", "cyan")
        
        # CONDITIONS COMBINATION
        # We need: HTF Bias + Liquidity Sweep + Discount/Premium Zone + Tap into OB/FVG
        
        if not liquidity_swept: return # Strict liquidity rule
        if not in_zone: return # Strict P/D rule
        
        signal_type = None
        stop_loss = 0.0
        
        # 5. Find Entry Point (FVG or OB)
        if htf_bias == "Bullish":
            # Look for Bullish FVG or OB
            fvg = self.find_fvg(ltf_data, "Bullish")
            ob = self.find_ob(ltf_data, "Bullish")
            
            # Check if current price is inside/touching FVG or OB
            # Prioritize FVG
            entry_found = False
            
            if fvg:
                top, bottom, _ = fvg
                # If price dipped into FVG
                if bottom <= current_price <= top * 1.02: # Tolerance
                    signal_type = "Buy"
                    stop_loss = bottom - (top - bottom) # Buffer below
                    entry_found = True
            
            if not entry_found and ob:
                top, bottom, _ = ob
                if bottom <= current_price <= top * 1.02:
                    signal_type = "Buy"
                    stop_loss = bottom - (top-bottom)*0.5
                    entry_found = True

        elif htf_bias == "Bearish":
            fvg = self.find_fvg(ltf_data, "Bearish")
            ob = self.find_ob(ltf_data, "Bearish")
            
            entry_found = False
            
            if fvg:
                top, bottom, _ = fvg
                if bottom * 0.98 <= current_price <= top:
                    signal_type = "Sell"
                    stop_loss = top + (top-bottom)
                    entry_found = True
            
            if not entry_found and ob:
                top, bottom, _ = ob
                if bottom * 0.98 <= current_price <= top:
                    signal_type = "Sell"
                    stop_loss = top + (top-bottom)*0.5
                    entry_found = True

        # Execute
        if signal_type:
            # Check existing action to avoid spam
            if self.worker.last_trade_action == signal_type: return

            # Calculate TP
            dist_to_sl = abs(current_price - stop_loss)
            if dist_to_sl == 0: dist_to_sl = 0.5 # Safety
            
            tp_dist = dist_to_sl * self.rr_ratio
            tp = current_price + tp_dist if signal_type == "Buy" else current_price - tp_dist
            
            # Calculate Volume
            volume = self.calculate_volume(dist_to_sl)
            
            self.worker.last_trade_action = signal_type
            self.worker.execute_trade(signal_type, is_auto=True, sl_price=stop_loss, tp_price=tp, volume=volume)
            self.worker.signals.update_status.emit(f"SMC: {signal_type} Executed! Vol={volume}", "green")

    def calculate_volume(self, sl_distance):
        try:
            account = mt5.account_info()
            symbol_info = mt5.symbol_info(self.symbol)
            if not account or not symbol_info: return 0.01
            
            balance = account.balance
            risk_amount = balance * (self.risk_percent / 100.0)
            
            tick_value = symbol_info.trade_tick_value
            tick_size = symbol_info.trade_tick_size
            
            # Standard formula: Volume = Risk / (SL_Points * TickVal)
            # SL Distance is in Price. Points = SL_Dist / TickSize
            if sl_distance <= 0: return 0.01
            
            points_at_risk = sl_distance / tick_size
            # Rough approx for XAUUSD if tick_value is standard
            # Better generic: Risk / (SL_Dist * ContractSize) * ...
            
            contract_size = symbol_info.trade_contract_size
            # Profit = (Close - Open) * Volume * ContractSize
            # Risk = SL_Dist * Vol * Contract
            # Vol = Risk / (SL_Dist * Contract)
            
            raw_vol = risk_amount / (sl_distance * contract_size)
            
            step = symbol_info.volume_step
            vol = round(raw_vol / step) * step
            vol = max(symbol_info.volume_min, min(symbol_info.volume_max, vol))
            return vol
        except:
            return 0.01

def run_smc_logic(worker, symbol, timeframe, risk_percent, rr_ratio):
    try:
        trader = SMCTrader(worker, symbol, timeframe, risk_percent, rr_ratio)
        trader.execute()
    except Exception as e:
        worker.signals.update_status.emit(f"SMC Error: {str(e)}", "red")