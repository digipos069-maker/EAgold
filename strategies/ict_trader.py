import datetime
from zoneinfo import ZoneInfo
from strategies.smc import SMCTrader

class ICTTrader(SMCTrader):
    def __init__(self, worker, symbol, timeframe, risk_percent=1.0, rr_ratio=2.0):
        super().__init__(worker, symbol, timeframe, risk_percent, rr_ratio, name="ICT")
        self.ny_tz = ZoneInfo("America/New_York")

    def execute(self):
        if not self._is_in_killzone():
            self._status("Outside Killzone. Waiting.", "gray")
            return
            
        super().execute()

    def _is_in_killzone(self):
        """
        Check if current time is within an ICT Killzone (NY Time).
        London: 02:00 - 05:00
        NY AM: 08:30 - 11:00
        NY PM: 13:30 - 16:00
        """
        now_ny = datetime.datetime.now(self.ny_tz)
        t = now_ny.time()
        
        # London Open
        if datetime.time(2, 0) <= t <= datetime.time(5, 0):
            return True
            
        # NY AM Open
        if datetime.time(8, 30) <= t <= datetime.time(11, 0):
            return True
            
        # NY PM Session
        if datetime.time(13, 30) <= t <= datetime.time(16, 0):
            return True
            
        return False

    def in_premium_discount(self, data, swings, bias):
        """
        Enforce Optimal Trade Entry (OTE) for ICT.
        OTE is strictly the 62% to 79% Fibonacci retracement of the displacement leg.
        """
        highs = [s for s in swings if s.type == "high"]
        lows = [s for s in swings if s.type == "low"]
        if not highs or not lows:
            return False

        if bias == "Bullish":
            recent_high = highs[-1]
            prior_lows = [s for s in lows if s.index < recent_high.index]
            if not prior_lows:
                return False
            low = prior_lows[-1].price
            high = recent_high.price
        else:
            recent_low = lows[-1]
            prior_highs = [s for s in highs if s.index < recent_low.index]
            if not prior_highs:
                return False
            high = prior_highs[-1].price
            low = recent_low.price

        if high <= low:
            return False

        current = float(data.close[-1])
        position = (current - low) / (high - low)
        
        # OTE is 0.62 to 0.79 retracement of the move
        if bias == "Bullish":
            # Retracement from high (1.0) down to current
            # 62% retracement = 1 - 0.62 = 0.38
            # 79% retracement = 1 - 0.79 = 0.21
            return 0.21 <= position <= 0.38
        else:
            # Retracement from low (0.0) up to current
            # 62% retracement = 0.62
            # 79% retracement = 0.79
            return 0.62 <= position <= 0.79


def run_ict_trader_logic(worker, symbol, timeframe):
    params = getattr(worker, "strategy_params", {})
    risk_percent = float(params.get("param1", 1.0))
    rr_ratio = float(params.get("param2", 2.0))
    try:
        ICTTrader(worker, symbol, timeframe, risk_percent, rr_ratio).execute()
    except Exception as e:
        worker.signals.update_status.emit(f"ICT Error: {e}", "red")

