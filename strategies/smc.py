from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np


MAGIC = 234000
HTF_TIMEFRAME = mt5.TIMEFRAME_H1
MAX_LOOKBACK = 320
SWING_LOOKBACK = 3
SETUP_LOOKBACK = 90
MAX_SPREAD_POINTS = 300
MAX_TRADES_PER_DAY = 3
MIN_RR = 2.0


@dataclass
class MarketData:
    time: list
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    tick_volume: np.ndarray
    spread: np.ndarray


@dataclass
class Swing:
    index: int
    price: float
    type: str


@dataclass
class LiquidityEvent:
    side: str
    level: float
    sweep_index: int
    source: str


@dataclass
class StructureShift:
    direction: str
    break_index: int
    break_level: float
    kind: str


@dataclass
class Zone:
    kind: str
    direction: str
    top: float
    bottom: float
    index: int


class SMCTrader:
    def __init__(self, worker, symbol, timeframe, risk_percent=1.0, rr_ratio=2.0, name="SMC"):
        self.worker = worker
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_percent = self._safe_float(risk_percent, 1.0)
        self.rr_ratio = max(self._safe_float(rr_ratio, 2.0), MIN_RR)
        self.name = name

    def execute(self):
        if not self._market_is_tradeable():
            return

        htf = self.get_data(HTF_TIMEFRAME, MAX_LOOKBACK)
        ltf = self.get_data(self.timeframe, MAX_LOOKBACK)
        if htf is None or ltf is None:
            self._status("Waiting for enough candles.", "orange")
            return

        atr = self.calculate_atr(ltf)
        if atr is None or atr <= 0:
            self._status("Waiting for volatility data.", "orange")
            return

        if not self._passes_volatility_filter(ltf, atr):
            self._status("Low volatility. No trade.", "orange")
            return

        htf_swings = self.find_swings(htf)
        ltf_swings = self.find_swings(ltf)
        bias = self.get_market_bias(htf, htf_swings)
        if bias not in ("Bullish", "Bearish"):
            self._status(f"HTF bias not clean ({bias}).", "orange")
            return

        if self._daily_trade_count() >= MAX_TRADES_PER_DAY:
            self._status(f"Daily trade limit reached ({MAX_TRADES_PER_DAY}).", "orange")
            return

        liquidity = self.detect_liquidity_sweep(ltf, ltf_swings, bias, atr)
        mss = self.detect_mss_after_sweep(ltf, ltf_swings, liquidity) if liquidity else None
        in_pd = self.in_premium_discount(ltf, ltf_swings, bias)

        self._status(
            f"{self.name}: Bias={bias} | Sweep={bool(liquidity)} | MSS={bool(mss)} | PD={in_pd}",
            "cyan",
        )

        if not liquidity or not mss or not in_pd:
            return

        entry = self.select_entry_zone(ltf, liquidity, mss, bias, atr)
        if entry is None:
            return

        signal = "Buy" if bias == "Bullish" else "Sell"
        current_price = float(ltf.close[-1])
        stop_loss = self.calculate_stop_loss(entry, liquidity, signal, atr)
        if not self._valid_stop(signal, current_price, stop_loss):
            return

        tp = self.calculate_take_profit(ltf, ltf_swings, signal, current_price, stop_loss)
        volume = self.calculate_volume(abs(current_price - stop_loss))
        if volume <= 0:
            self._status("Volume calculation failed.", "red")
            return

        if self._has_open_direction(signal):
            self._status(f"{self.name}: Existing {signal} position. Waiting.", "orange")
            return

        if self.worker.last_trade_action == signal:
            return

        result = self.worker.execute_trade(
            signal,
            is_auto=True,
            sl_price=stop_loss,
            tp_price=tp,
            volume=volume,
        )
        if result is not False:
            self.worker.last_trade_action = signal
            self._status(
                f"{self.name}: {signal} {entry.kind} entry | Vol={volume:.2f} | SL={stop_loss:.2f} | TP={tp:.2f}",
                "green",
            )

    def get_data(self, timeframe, count):
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
        if rates is None or len(rates) < max(120, count // 2):
            return None
        return MarketData(
            time=[datetime.fromtimestamp(int(r["time"])) for r in rates],
            open=np.array([r["open"] for r in rates], dtype=float),
            high=np.array([r["high"] for r in rates], dtype=float),
            low=np.array([r["low"] for r in rates], dtype=float),
            close=np.array([r["close"] for r in rates], dtype=float),
            tick_volume=np.array([r["tick_volume"] for r in rates], dtype=float),
            spread=np.array([r["spread"] for r in rates], dtype=float),
        )

    def find_swings(self, data):
        swings = []
        for i in range(SWING_LOOKBACK, len(data.close) - SWING_LOOKBACK):
            left_high = data.high[i - SWING_LOOKBACK : i]
            right_high = data.high[i + 1 : i + SWING_LOOKBACK + 1]
            left_low = data.low[i - SWING_LOOKBACK : i]
            right_low = data.low[i + 1 : i + SWING_LOOKBACK + 1]
            if data.high[i] > np.max(left_high) and data.high[i] >= np.max(right_high):
                swings.append(Swing(i, float(data.high[i]), "high"))
            if data.low[i] < np.min(left_low) and data.low[i] <= np.min(right_low):
                swings.append(Swing(i, float(data.low[i]), "low"))
        return swings

    def get_market_bias(self, data, swings):
        highs = [s for s in swings if s.type == "high"]
        lows = [s for s in swings if s.type == "low"]
        if len(highs) < 2 or len(lows) < 2:
            return "Uncertain"

        hh = highs[-1].price > highs[-2].price
        hl = lows[-1].price > lows[-2].price
        lh = highs[-1].price < highs[-2].price
        ll = lows[-1].price < lows[-2].price

        if hh and hl:
            return "Bullish"
        if lh and ll:
            return "Bearish"

        current_close = float(data.close[-1])
        if current_close > highs[-1].price and lows[-1].price >= lows[-2].price:
            return "Bullish"
        if current_close < lows[-1].price and highs[-1].price <= highs[-2].price:
            return "Bearish"
        return "Ranging"

    def detect_liquidity_sweep(self, data, swings, bias, atr):
        start = max(10, len(data.close) - SETUP_LOOKBACK)
        tolerance = max(atr * 0.08, self._point() * 5)
        previous_day = self._previous_day_levels()

        levels = []
        if bias == "Bullish":
            levels.extend(("swing low", s.price) for s in swings if s.type == "low" and s.index < len(data.close) - 3)
            levels.extend(("equal lows", price) for price in self.find_equal_liquidity(swings, "low", tolerance))
            if previous_day:
                levels.append(("previous day low", previous_day[0]))

            for i in range(start, len(data.close)):
                for source, level in levels:
                    swept = data.low[i] < level - tolerance
                    reclaimed = data.close[i] > level
                    if swept and reclaimed:
                        return LiquidityEvent("sell-side", float(level), i, source)

        if bias == "Bearish":
            levels.extend(("swing high", s.price) for s in swings if s.type == "high" and s.index < len(data.close) - 3)
            levels.extend(("equal highs", price) for price in self.find_equal_liquidity(swings, "high", tolerance))
            if previous_day:
                levels.append(("previous day high", previous_day[1]))

            for i in range(start, len(data.close)):
                for source, level in levels:
                    swept = data.high[i] > level + tolerance
                    reclaimed = data.close[i] < level
                    if swept and reclaimed:
                        return LiquidityEvent("buy-side", float(level), i, source)

        return None

    def find_equal_liquidity(self, swings, swing_type, tolerance):
        matching = [s for s in swings if s.type == swing_type]
        levels = []
        for i in range(1, len(matching)):
            if abs(matching[i].price - matching[i - 1].price) <= tolerance:
                levels.append((matching[i].price + matching[i - 1].price) / 2.0)
        return levels[-4:]

    def detect_mss_after_sweep(self, data, swings, liquidity):
        if liquidity is None:
            return None

        post_sweep = [s for s in swings if s.index < liquidity.sweep_index]
        if liquidity.side == "sell-side":
            highs = [s for s in post_sweep if s.type == "high"]
            if not highs:
                return None
            break_level = highs[-1].price
            for i in range(liquidity.sweep_index + 1, len(data.close)):
                if data.close[i] > break_level:
                    return StructureShift("Bullish", i, break_level, "CHoCH/MSS")

        if liquidity.side == "buy-side":
            lows = [s for s in post_sweep if s.type == "low"]
            if not lows:
                return None
            break_level = lows[-1].price
            for i in range(liquidity.sweep_index + 1, len(data.close)):
                if data.close[i] < break_level:
                    return StructureShift("Bearish", i, break_level, "CHoCH/MSS")

        return None

    def in_premium_discount(self, data, swings, bias):
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
        if bias == "Bullish":
            return 0.21 <= position <= 0.50
        return 0.50 <= position <= 0.79

    def select_entry_zone(self, data, liquidity, mss, bias, atr):
        zones = self.find_fvg_zones(data, bias, liquidity.sweep_index, mss.break_index)
        zones.extend(self.find_order_blocks(data, bias, liquidity.sweep_index, mss.break_index, atr))
        current = float(data.close[-1])
        active = [z for z in zones if z.bottom <= current <= z.top]
        if not active:
            return None

        active.sort(key=lambda z: (0 if z.kind == "FVG" else 1, -z.index))
        return active[0]

    def find_fvg_zones(self, data, bias, sweep_index, break_index):
        zones = []
        start = max(2, sweep_index)
        end = min(len(data.close) - 2, max(break_index + 8, sweep_index + 3))
        for i in range(start, end):
            if bias == "Bullish" and data.high[i - 1] < data.low[i + 1]:
                bottom = float(data.high[i - 1])
                top = float(data.low[i + 1])
                if not self._zone_invalidated(data, "Bullish", top, bottom, i + 1):
                    zones.append(Zone("FVG", "Bullish", top, bottom, i))
            elif bias == "Bearish" and data.low[i - 1] > data.high[i + 1]:
                top = float(data.low[i - 1])
                bottom = float(data.high[i + 1])
                if not self._zone_invalidated(data, "Bearish", top, bottom, i + 1):
                    zones.append(Zone("FVG", "Bearish", top, bottom, i))
        return zones

    def find_order_blocks(self, data, bias, sweep_index, break_index, atr):
        zones = []
        displacement_min = atr * 0.8
        start = max(1, sweep_index)
        end = min(break_index + 1, len(data.close) - 1)
        for i in range(end - 1, start - 1, -1):
            body = abs(data.close[i + 1] - data.open[i + 1])
            if body < displacement_min:
                continue
            if bias == "Bullish" and data.close[i] < data.open[i] and data.close[i + 1] > data.high[i]:
                zones.append(Zone("OB", "Bullish", float(data.open[i]), float(data.low[i]), i))
                break
            if bias == "Bearish" and data.close[i] > data.open[i] and data.close[i + 1] < data.low[i]:
                zones.append(Zone("OB", "Bearish", float(data.high[i]), float(data.open[i]), i))
                break
        return zones

    def calculate_stop_loss(self, zone, liquidity, signal, atr):
        buffer = max(atr * 0.15, self._point() * 20)
        if signal == "Buy":
            return min(zone.bottom, liquidity.level) - buffer
        return max(zone.top, liquidity.level) + buffer

    def calculate_take_profit(self, data, swings, signal, entry, stop_loss):
        fixed_rr_target = entry + abs(entry - stop_loss) * self.rr_ratio if signal == "Buy" else entry - abs(entry - stop_loss) * self.rr_ratio
        if signal == "Buy":
            highs = [s.price for s in swings if s.type == "high" and s.price > entry]
            liquidity_target = min(highs) if highs else fixed_rr_target
            return max(fixed_rr_target, liquidity_target)
        lows = [s.price for s in swings if s.type == "low" and s.price < entry]
        liquidity_target = max(lows) if lows else fixed_rr_target
        return min(fixed_rr_target, liquidity_target)

    def calculate_atr(self, data, period=14):
        if len(data.close) < period + 2:
            return None
        prev_close = data.close[:-1]
        high = data.high[1:]
        low = data.low[1:]
        tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
        return float(np.mean(tr[-period:]))

    def calculate_volume(self, sl_distance):
        try:
            account = mt5.account_info()
            symbol_info = mt5.symbol_info(self.symbol)
            if not account or not symbol_info or sl_distance <= 0:
                return 0.0

            risk_amount = account.balance * (self.risk_percent / 100.0)
            contract_size = symbol_info.trade_contract_size
            if contract_size <= 0:
                return 0.0

            raw_volume = risk_amount / (sl_distance * contract_size)
            step = symbol_info.volume_step or 0.01
            volume = round(raw_volume / step) * step
            volume = max(symbol_info.volume_min, min(symbol_info.volume_max, volume))
            return round(volume, 2)
        except Exception:
            return 0.0

    def _market_is_tradeable(self):
        symbol_info = mt5.symbol_info(self.symbol)
        tick = mt5.symbol_info_tick(self.symbol)
        if not symbol_info or not tick:
            self._status(f"Symbol not available: {self.symbol}", "red")
            return False
        if not symbol_info.visible:
            mt5.symbol_select(self.symbol, True)
        if symbol_info.spread > MAX_SPREAD_POINTS:
            self._status(f"Spread too high ({symbol_info.spread}).", "orange")
            return False
        return True

    def _passes_volatility_filter(self, data, atr):
        recent_range = float(np.mean(data.high[-10:] - data.low[-10:]))
        return recent_range >= atr * 0.35

    def _zone_invalidated(self, data, direction, top, bottom, created_index):
        if created_index >= len(data.close) - 1:
            return False
        later_closes = data.close[created_index + 1 : -1]
        if len(later_closes) == 0:
            return False
        if direction == "Bullish":
            return bool(np.any(later_closes < bottom))
        return bool(np.any(later_closes > top))

    def _previous_day_levels(self):
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_D1, 1, 1)
        if rates is None or len(rates) == 0:
            return None
        return float(rates[0]["low"]), float(rates[0]["high"])

    def _daily_trade_count(self):
        try:
            now = datetime.now(timezone.utc)
            start = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
            deals = mt5.history_deals_get(start, now + timedelta(minutes=1))
            if not deals:
                return 0
            return sum(1 for d in deals if getattr(d, "magic", None) == MAGIC and getattr(d, "symbol", "") == self.symbol)
        except Exception:
            return 0

    def _has_open_direction(self, signal):
        positions = mt5.positions_get(symbol=self.symbol, magic=MAGIC)
        if not positions:
            return False
        target_type = mt5.ORDER_TYPE_BUY if signal == "Buy" else mt5.ORDER_TYPE_SELL
        return any(pos.type == target_type for pos in positions)

    def _valid_stop(self, signal, entry, stop_loss):
        if signal == "Buy":
            return stop_loss < entry
        return stop_loss > entry

    def _point(self):
        info = mt5.symbol_info(self.symbol)
        return float(info.point) if info and info.point else 0.01

    def _safe_float(self, value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _status(self, message, color):
        self.worker.signals.update_status.emit(f"{self.name}: {message}" if not message.startswith(self.name) else message, color)


class ICTTrader(SMCTrader):
    def __init__(self, worker, symbol, timeframe, risk_percent=1.0, rr_ratio=2.0):
        super().__init__(worker, symbol, timeframe, risk_percent, rr_ratio, name="ICT")


def run_smc_logic(worker, symbol, timeframe, risk_percent, rr_ratio):
    try:
        SMCTrader(worker, symbol, timeframe, risk_percent, rr_ratio).execute()
    except Exception as e:
        worker.signals.update_status.emit(f"SMC Error: {e}", "red")


def run_ict_logic(worker, symbol, timeframe, risk_percent=1.0, rr_ratio=2.0):
    try:
        ICTTrader(worker, symbol, timeframe, risk_percent, rr_ratio).execute()
    except Exception as e:
        worker.signals.update_status.emit(f"ICT Error: {e}", "red")
