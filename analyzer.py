import yfinance as yf
import pandas as pd
import numpy as np
import ta
from config import TA_CONFIG


def fetch_data(symbol, period="5d", interval="15m"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"  [!] Error fetching {symbol}: {e}")
        return None


def fetch_higher_timeframe(symbol, period="1mo", interval="1h"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def compute_indicators(df):
    cfg = TA_CONFIG

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=cfg["rsi_period"]).rsi()

    # MACD
    macd = ta.trend.MACD(df["Close"], window_slow=cfg["macd_slow"], window_fast=cfg["macd_fast"], window_sign=cfg["macd_signal"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # EMAs
    df["ema_short"] = ta.trend.EMAIndicator(df["Close"], window=cfg["ema_short"]).ema_indicator()
    df["ema_long"] = ta.trend.EMAIndicator(df["Close"], window=cfg["ema_long"]).ema_indicator()

    # 200 SMA — trend filter
    df["sma_200"] = df["Close"].rolling(window=min(200, len(df))).mean()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"], window=cfg["bb_period"], window_dev=cfg["bb_std"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=cfg["atr_period"]).average_true_range()

    # Volume SMA
    df["vol_sma"] = df["Volume"].rolling(window=20).mean()

    # Stochastic RSI
    stoch = ta.momentum.StochRSIIndicator(df["Close"], window=cfg["rsi_period"])
    df["stoch_k"] = stoch.stochrsi_k()
    df["stoch_d"] = stoch.stochrsi_d()

    # ADX
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # Supertrend
    df = compute_supertrend(df, period=10, multiplier=3)

    # Support / Resistance
    df = compute_support_resistance(df)

    # Candlestick patterns
    df = detect_candle_patterns(df)

    return df


def compute_supertrend(df, period=10, multiplier=3):
    hl2 = (df["High"] + df["Low"]) / 2
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=period).average_true_range()

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(df)):
        if df["Close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["Close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i - 1]) if direction.iloc[i - 1] == 1 else lower_band.iloc[i]
        else:
            supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i - 1]) if direction.iloc[i - 1] == -1 else upper_band.iloc[i]

    df["supertrend"] = supertrend
    df["supertrend_dir"] = direction
    return df


def compute_support_resistance(df, lookback=50):
    window = min(lookback, len(df) - 1)
    if window < 10:
        df["support"] = df["Low"].min()
        df["resistance"] = df["High"].max()
        return df

    recent = df.tail(window)

    # Swing lows = support
    lows = recent["Low"].values
    swing_lows = []
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
            swing_lows.append(lows[i])

    # Swing highs = resistance
    highs = recent["High"].values
    swing_highs = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
            swing_highs.append(highs[i])

    price = df["Close"].iloc[-1]

    # Nearest support below price
    supports_below = [s for s in swing_lows if s < price]
    support = max(supports_below) if supports_below else recent["Low"].min()

    # Nearest resistance above price
    resistances_above = [r for r in swing_highs if r > price]
    resistance = min(resistances_above) if resistances_above else recent["High"].max()

    df["support"] = support
    df["resistance"] = resistance

    return df


def detect_candle_patterns(df):
    o = df["Open"]
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    body = abs(c - o)
    candle_range = h - l

    # Bullish Engulfing
    prev_bearish = df["Close"].shift(1) < df["Open"].shift(1)
    curr_bullish = c > o
    engulfs = (o <= df["Close"].shift(1)) & (c >= df["Open"].shift(1))
    df["bullish_engulfing"] = prev_bearish & curr_bullish & engulfs

    # Bearish Engulfing
    prev_bullish = df["Close"].shift(1) > df["Open"].shift(1)
    curr_bearish = c < o
    engulfs_bear = (o >= df["Close"].shift(1)) & (c <= df["Open"].shift(1))
    df["bearish_engulfing"] = prev_bullish & curr_bearish & engulfs_bear

    # Pin Bar / Hammer (bullish)
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)
    df["hammer"] = (lower_wick > 2 * body) & (upper_wick < body * 0.5) & (candle_range > 0)

    # Shooting Star (bearish)
    df["shooting_star"] = (upper_wick > 2 * body) & (lower_wick < body * 0.5) & (candle_range > 0)

    # Doji
    df["doji"] = body < (candle_range * 0.1)

    # Morning Star (3-candle bullish reversal)
    c1_bearish = df["Close"].shift(2) < df["Open"].shift(2)
    c2_small = abs(df["Close"].shift(1) - df["Open"].shift(1)) < (abs(df["Close"].shift(2) - df["Open"].shift(2)) * 0.3)
    c3_bullish = c > o
    c3_closes_above = c > (df["Open"].shift(2) + df["Close"].shift(2)) / 2
    df["morning_star"] = c1_bearish & c2_small & c3_bullish & c3_closes_above

    # Evening Star (3-candle bearish reversal)
    c1_bullish = df["Close"].shift(2) > df["Open"].shift(2)
    c3_bearish_es = c < o
    c3_closes_below = c < (df["Open"].shift(2) + df["Close"].shift(2)) / 2
    df["evening_star"] = c1_bullish & c2_small & c3_bearish_es & c3_closes_below

    return df


def get_higher_tf_trend(symbol):
    df_1h = fetch_higher_timeframe(symbol)
    if df_1h is None or len(df_1h) < 30:
        return {"trend": "UNKNOWN", "rsi": None, "ema_trend": None}

    rsi = ta.momentum.RSIIndicator(df_1h["Close"], window=14).rsi().iloc[-1]
    ema9 = ta.trend.EMAIndicator(df_1h["Close"], window=9).ema_indicator().iloc[-1]
    ema21 = ta.trend.EMAIndicator(df_1h["Close"], window=21).ema_indicator().iloc[-1]

    macd_obj = ta.trend.MACD(df_1h["Close"])
    macd_val = macd_obj.macd().iloc[-1]
    macd_sig = macd_obj.macd_signal().iloc[-1]

    bullish_count = 0
    bearish_count = 0

    if ema9 > ema21:
        bullish_count += 1
    else:
        bearish_count += 1

    if macd_val > macd_sig:
        bullish_count += 1
    else:
        bearish_count += 1

    if rsi > 50:
        bullish_count += 1
    else:
        bearish_count += 1

    if bullish_count >= 2:
        trend = "BULLISH"
    elif bearish_count >= 2:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    return {
        "trend": trend,
        "rsi": round(rsi, 1) if pd.notna(rsi) else None,
        "ema_trend": "BULLISH" if ema9 > ema21 else "BEARISH",
    }


def compute_market_structure(df, lookback=30):
    window = min(lookback, len(df) - 1)
    if window < 10:
        return "UNKNOWN"

    recent = df.tail(window)
    highs = recent["High"].values
    lows = recent["Low"].values

    swing_highs = []
    swing_lows = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            swing_lows.append(lows[i])

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1] > swing_highs[-2]
        hl = swing_lows[-1] > swing_lows[-2]
        lh = swing_highs[-1] < swing_highs[-2]
        ll = swing_lows[-1] < swing_lows[-2]

        if hh and hl:
            return "UPTREND"
        elif lh and ll:
            return "DOWNTREND"

    return "RANGE"


def calculate_position_size(capital, risk_pct, entry, stop_loss):
    risk_amount = capital * (risk_pct / 100)
    risk_per_unit = abs(entry - stop_loss)
    if risk_per_unit == 0:
        return 0
    qty = int(risk_amount / risk_per_unit)
    return max(qty, 0)


def generate_signal(df, name, symbol=None):
    if df is None or len(df) < 30:
        return None

    df = compute_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    cfg = TA_CONFIG

    # ── Higher timeframe trend ──
    htf = {"trend": "UNKNOWN", "rsi": None, "ema_trend": None}
    if symbol:
        htf = get_higher_tf_trend(symbol)

    # ── Market structure ──
    structure = compute_market_structure(df)

    # ── 200 SMA trend filter ──
    above_200sma = last["Close"] > last["sma_200"] if pd.notna(last["sma_200"]) else True
    below_200sma = last["Close"] < last["sma_200"] if pd.notna(last["sma_200"]) else True

    # ── Score indicators ──
    buy_score = 0
    sell_score = 0
    reasons_buy = []
    reasons_sell = []

    # 1. RSI
    if last["rsi"] < cfg["rsi_oversold"]:
        buy_score += 1
        reasons_buy.append(f"RSI oversold ({last['rsi']:.1f})")
    elif last["rsi"] > cfg["rsi_overbought"]:
        sell_score += 1
        reasons_sell.append(f"RSI overbought ({last['rsi']:.1f})")

    # 2. MACD crossover
    if prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]:
        buy_score += 1.5
        reasons_buy.append("MACD bullish crossover")
    elif prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]:
        sell_score += 1.5
        reasons_sell.append("MACD bearish crossover")

    # 3. EMA crossover
    if last["ema_short"] > last["ema_long"] and prev["ema_short"] <= prev["ema_long"]:
        buy_score += 1.5
        reasons_buy.append(f"EMA {cfg['ema_short']} crossed above EMA {cfg['ema_long']}")
    elif last["ema_short"] < last["ema_long"] and prev["ema_short"] >= prev["ema_long"]:
        sell_score += 1.5
        reasons_sell.append(f"EMA {cfg['ema_short']} crossed below EMA {cfg['ema_long']}")

    # 4. Bollinger Band touch
    if last["Close"] <= last["bb_lower"]:
        buy_score += 1
        reasons_buy.append("Price at lower Bollinger Band")
    elif last["Close"] >= last["bb_upper"]:
        sell_score += 1
        reasons_sell.append("Price at upper Bollinger Band")

    # 5. Stochastic RSI
    if last["stoch_k"] < 0.2 and last["stoch_d"] < 0.2:
        buy_score += 1
        reasons_buy.append(f"Stoch RSI oversold ({last['stoch_k']:.2f})")
    elif last["stoch_k"] > 0.8 and last["stoch_d"] > 0.8:
        sell_score += 1
        reasons_sell.append(f"Stoch RSI overbought ({last['stoch_k']:.2f})")

    # 6. Volume confirmation
    if pd.notna(last["vol_sma"]) and last["vol_sma"] > 0 and last["Volume"] > last["vol_sma"] * 1.5:
        buy_score += 0.5
        sell_score += 0.5
        reasons_buy.append("High volume confirmation")
        reasons_sell.append("High volume confirmation")

    # ── NEW: 7. Supertrend ──
    if pd.notna(last["supertrend_dir"]):
        if last["supertrend_dir"] == 1 and prev.get("supertrend_dir", 0) == -1:
            buy_score += 1.5
            reasons_buy.append("Supertrend flipped BULLISH")
        elif last["supertrend_dir"] == 1:
            buy_score += 0.5
            reasons_buy.append("Supertrend bullish")
        elif last["supertrend_dir"] == -1 and prev.get("supertrend_dir", 0) == 1:
            sell_score += 1.5
            reasons_sell.append("Supertrend flipped BEARISH")
        elif last["supertrend_dir"] == -1:
            sell_score += 0.5
            reasons_sell.append("Supertrend bearish")

    # ── NEW: 8. Candlestick patterns ──
    if last.get("bullish_engulfing", False):
        buy_score += 1
        reasons_buy.append("Bullish Engulfing candle")
    if last.get("bearish_engulfing", False):
        sell_score += 1
        reasons_sell.append("Bearish Engulfing candle")
    if last.get("hammer", False):
        buy_score += 0.5
        reasons_buy.append("Hammer candle (bullish reversal)")
    if last.get("shooting_star", False):
        sell_score += 0.5
        reasons_sell.append("Shooting Star candle (bearish reversal)")
    if last.get("morning_star", False):
        buy_score += 1
        reasons_buy.append("Morning Star pattern (strong bullish)")
    if last.get("evening_star", False):
        sell_score += 1
        reasons_sell.append("Evening Star pattern (strong bearish)")

    # ── NEW: 9. Support/Resistance proximity ──
    price = last["Close"]
    support = last.get("support", price * 0.99)
    resistance = last.get("resistance", price * 1.01)
    price_range = resistance - support if resistance > support else price * 0.01

    near_support = (price - support) / price_range < 0.15 if price_range > 0 else False
    near_resistance = (resistance - price) / price_range < 0.15 if price_range > 0 else False

    if near_support:
        buy_score += 1
        reasons_buy.append(f"Price near support ({support:.2f})")
    if near_resistance:
        sell_score += 1
        reasons_sell.append(f"Price near resistance ({resistance:.2f})")

    # ── NEW: 10. 200 SMA trend filter — GATE ──
    trend_aligned_buy = above_200sma
    trend_aligned_sell = below_200sma

    # ── NEW: 11. Higher timeframe confirmation — BONUS ──
    htf_buy_aligned = htf["trend"] in ("BULLISH", "UNKNOWN")
    htf_sell_aligned = htf["trend"] in ("BEARISH", "UNKNOWN")

    if htf["trend"] == "BULLISH":
        buy_score += 0.5
        reasons_buy.append(f"1H timeframe is BULLISH (RSI {htf['rsi']})")
    elif htf["trend"] == "BEARISH":
        sell_score += 0.5
        reasons_sell.append(f"1H timeframe is BEARISH (RSI {htf['rsi']})")

    # ── NEW: 12. Market structure bonus ──
    if structure == "UPTREND":
        buy_score += 0.5
        reasons_buy.append("Market structure: Higher Highs + Higher Lows")
    elif structure == "DOWNTREND":
        sell_score += 0.5
        reasons_sell.append("Market structure: Lower Highs + Lower Lows")

    # ── Trend strength ──
    trend_strong = last["adx"] > 25 if pd.notna(last["adx"]) else False

    # ── ATR for stop-loss ──
    atr = last["atr"] if pd.notna(last["atr"]) else price * 0.01

    # ── Final signal decision ──
    # NEW TOTAL: max possible ~12 points. Threshold = 4 for signal.
    min_score = cfg.get("strong_signal_min_score", 4)
    signal = None

    if buy_score >= min_score and trend_aligned_buy:
        sl = round(price - 1.5 * atr, 2)

        # Smart SL: use support if it's tighter
        if near_support and support > sl:
            sl = round(support - 0.2 * atr, 2)

        risk = abs(price - sl)
        t1 = round(price + 1.5 * risk, 2)
        t2 = round(price + 3 * risk, 2)

        # Position sizing (default 1L capital, 1% risk)
        qty = calculate_position_size(100000, 1, price, sl)

        signal = {
            "name": name,
            "action": "BUY",
            "strength": "STRONG" if buy_score >= 6 else "MODERATE",
            "price": round(price, 2),
            "stop_loss": sl,
            "target_1": t1,
            "target_2": t2,
            "risk_per_unit": round(risk, 2),
            "reward_t1": round(t1 - price, 2),
            "reward_t2": round(t2 - price, 2),
            "rr_t1": f"1:{round((t1 - price) / risk, 1)}" if risk > 0 else "N/A",
            "rr_t2": f"1:{round((t2 - price) / risk, 1)}" if risk > 0 else "N/A",
            "qty_suggestion": qty,
            "score": round(buy_score, 1),
            "max_score": 12,
            "reasons": reasons_buy,
            "trend_strong": trend_strong,
            "rsi": round(last["rsi"], 1) if pd.notna(last["rsi"]) else None,
            "adx": round(last["adx"], 1) if pd.notna(last["adx"]) else None,
            "htf_trend": htf["trend"],
            "structure": structure,
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "supertrend_dir": "BULLISH" if last.get("supertrend_dir") == 1 else "BEARISH",
            "filters_passed": {
                "200_sma": trend_aligned_buy,
                "htf_aligned": htf_buy_aligned,
                "structure": structure,
            },
        }

    elif sell_score >= min_score and trend_aligned_sell:
        sl = round(price + 1.5 * atr, 2)

        if near_resistance and resistance < sl:
            sl = round(resistance + 0.2 * atr, 2)

        risk = abs(sl - price)
        t1 = round(price - 1.5 * risk, 2)
        t2 = round(price - 3 * risk, 2)

        qty = calculate_position_size(100000, 1, price, sl)

        signal = {
            "name": name,
            "action": "SELL",
            "strength": "STRONG" if sell_score >= 6 else "MODERATE",
            "price": round(price, 2),
            "stop_loss": sl,
            "target_1": t1,
            "target_2": t2,
            "risk_per_unit": round(risk, 2),
            "reward_t1": round(price - t1, 2),
            "reward_t2": round(price - t2, 2),
            "rr_t1": f"1:{round((price - t1) / risk, 1)}" if risk > 0 else "N/A",
            "rr_t2": f"1:{round((price - t2) / risk, 1)}" if risk > 0 else "N/A",
            "qty_suggestion": qty,
            "score": round(sell_score, 1),
            "max_score": 12,
            "reasons": reasons_sell,
            "trend_strong": trend_strong,
            "rsi": round(last["rsi"], 1) if pd.notna(last["rsi"]) else None,
            "adx": round(last["adx"], 1) if pd.notna(last["adx"]) else None,
            "htf_trend": htf["trend"],
            "structure": structure,
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "supertrend_dir": "BULLISH" if last.get("supertrend_dir") == 1 else "BEARISH",
            "filters_passed": {
                "200_sma": trend_aligned_sell,
                "htf_aligned": htf_sell_aligned,
                "structure": structure,
            },
        }

    # ── Summary (always returned) ──
    summary = {
        "name": name,
        "price": round(price, 2),
        "rsi": round(last["rsi"], 1) if pd.notna(last["rsi"]) else None,
        "macd_hist": round(last["macd_hist"], 4) if pd.notna(last["macd_hist"]) else None,
        "ema_trend": "BULLISH" if last["ema_short"] > last["ema_long"] else "BEARISH",
        "bb_position": "NEAR LOWER" if last["Close"] < last["bb_mid"] else "NEAR UPPER",
        "adx": round(last["adx"], 1) if pd.notna(last["adx"]) else None,
        "supertrend": "BULLISH" if last.get("supertrend_dir") == 1 else "BEARISH",
        "structure": structure,
        "htf_trend": htf["trend"],
        "above_200sma": above_200sma,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "buy_score": round(buy_score, 1),
        "sell_score": round(sell_score, 1),
        "buy_reasons": reasons_buy,
        "sell_reasons": reasons_sell,
    }

    return {"signal": signal, "summary": summary}
