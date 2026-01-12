import numpy as np

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1: return np.array([])
    deltas = np.diff(prices)
    seed = deltas[:period]
    gains = seed[seed >= 0].sum() / period
    losses = -seed[seed < 0].sum() / period
    rs = gains / losses if losses != 0 else np.inf
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0: gain = delta; loss = 0
        else: gain = 0; loss = -delta
        gains = (gains * (period - 1) + gain) / period
        losses = (losses * (period - 1) + loss) / period
        rs = gains / losses if losses != 0 else np.inf
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def calculate_ema(prices, period):
    if len(prices) < period: return None
    return np.mean(prices[-period:])

def calculate_atr(high, low, close, period=14):
    if len(close) < period + 1: return None
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    # First ATR is simple average
    atr = np.zeros(len(tr))
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr[-1]

def calculate_adx(high, low, close, period=14):
    if len(close) < period * 2: return None
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    
    # Smooth TR, +DM, -DM
    def smooth(data, period):
        smoothed = np.zeros(len(data))
        smoothed[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1]/period) + data[i]
        return smoothed

    tr_s = smooth(tr, period)
    plus_dm_s = smooth(plus_dm, period)
    minus_dm_s = smooth(minus_dm, period)
    
    # Avoid division by zero
    tr_s[tr_s == 0] = 1e-9
    
    plus_di = 100 * (plus_dm_s / tr_s)
    minus_di = 100 * (minus_dm_s / tr_s)
    
    sum_di = plus_di + minus_di
    sum_di[sum_di == 0] = 1e-9
    
    dx = 100 * np.abs(plus_di - minus_di) / sum_di
    adx = smooth(dx, period)
    return adx[-1]

def calculate_macd(close, fast=12, slow=26, signal=9):
    if len(close) < slow + signal: return None, None, None, None
    
    def get_ema_series(prices, period):
        ema = np.zeros(len(prices))
        ema[period-1] = np.mean(prices[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema

    ema_fast = get_ema_series(close, fast)
    ema_slow = get_ema_series(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = get_ema_series(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line[-1], signal_line[-1], histogram[-1], histogram[-2]

def calculate_stochastic(high, low, close, k_period=5, d_period=3, slowing=3):
    if len(close) < k_period + d_period + slowing: return None, None
    
    # Calculate %K Line
    lowest_low = np.zeros(len(low))
    highest_high = np.zeros(len(high))
    
    for i in range(k_period - 1, len(low)):
        lowest_low[i] = np.min(low[i - k_period + 1:i + 1])
        highest_high[i] = np.max(high[i - k_period + 1:i + 1])
        
    denom = highest_high - lowest_low
    denom[denom == 0] = 1e-9 # Avoid div by zero
    
    raw_k = 100 * ((close - lowest_low) / denom)
    
    # Apply Slowing to %K
    k_line = np.zeros(len(raw_k))
    for i in range(slowing - 1, len(raw_k)):
        k_line[i] = np.mean(raw_k[i - slowing + 1:i + 1])
        
    # Calculate %D Line (SMA of %K)
    d_line = np.zeros(len(k_line))
    for i in range(d_period - 1, len(k_line)):
        d_line[i] = np.mean(k_line[i - d_period + 1:i + 1])
        
    return k_line[-1], d_line[-1]
