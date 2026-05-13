import pandas as pd
import numpy as np
import ta
from curl_cffi import requests as curl_requests

_SESSION = curl_requests.Session(impersonate="chrome110")


def _fetch_yahoo(symbol, period="1d", interval="5m"):
    """Fetch OHLCV data directly from Yahoo Finance v8 API."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "interval": interval,
        "range": period,
        "includePrePost": "false",
    }
    try:
        resp = _SESSION.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        chart = data.get("chart", {})
        if chart.get("error"):
            print(f"  [!] Yahoo error for {symbol}: {chart['error']}")
            return None
        result = chart.get("result")
        if not result:
            return None
        result = result[0]
        timestamps = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        df = pd.DataFrame({
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        }, index=pd.to_datetime(timestamps, unit="s", utc=True))
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"  [!] Scalp fetch error {symbol}: {e}")
        return None


def fetch_scalp_data(symbol, period="1d", interval="5m"):
    return _fetch_yahoo(symbol, period=period, interval=interval)


def compute_scalp_indicators(df):
    df["ema5"] = ta.trend.EMAIndicator(df["Close"], window=5).ema_indicator()
    df["ema13"] = ta.trend.EMAIndicator(df["Close"], window=13).ema_indicator()
    df["ema34"] = ta.trend.EMAIndicator(df["Close"], window=34).ema_indicator()

    df["rsi7"] = ta.momentum.RSIIndicator(df["Close"], window=7).rsi()

    macd = ta.trend.MACD(df["Close"], window_slow=17, window_fast=8, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df["Close"], window=14, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    df["atr"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=7).average_true_range()

    df["vol_sma"] = df["Volume"].rolling(window=10).mean()

    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=5, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["momentum"] = df["Close"].pct_change(3) * 100

    df["vwap"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()

    df["body"] = abs(df["Close"] - df["Open"])
    df["candle_range"] = df["High"] - df["Low"]

    # ADX for trend strength — only scalp in trending market
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=10)
    df["adx"] = adx.adx()

    return df


SCALP_CONFIG = {
    "GC=F": {
        "name": "Gold (XAU/USD)",
        "target_points": 10,
        "sl_points": 5,
        "label": "$",
    },
    "^NSEI": {
        "name": "NIFTY 50",
        "target_options": [20, 30, 50],
        "default_target": 30,
        "sl_points": 15,
        "label": "",
    },
    "^NSEBANK": {
        "name": "BANK NIFTY",
        "target_options": [30, 50, 75],
        "default_target": 50,
        "sl_points": 25,
        "label": "",
    },
}


def generate_scalp_signal(symbol):
    cfg = SCALP_CONFIG.get(symbol)
    if not cfg:
        return None

    df = fetch_scalp_data(symbol)
    if df is None or len(df) < 25:
        return None

    df = compute_scalp_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) > 3 else prev
    price = last["Close"]

    buy_score = 0
    sell_score = 0
    reasons_buy = []
    reasons_sell = []

    # ══════════════════════════════════════════════════
    # RULE 1: TREND DIRECTION MUST BE CLEAR
    # EMA stack is the primary filter — no trade without it
    # ══════════════════════════════════════════════════

    ema_bullish_stack = last["ema5"] > last["ema13"] > last["ema34"]
    ema_bearish_stack = last["ema5"] < last["ema13"] < last["ema34"]

    if not ema_bullish_stack and not ema_bearish_stack:
        # EMAs are mixed — NO SCALP, market is choppy
        return _build_result(cfg, df, last, 0, 0, [], [], None)

    if ema_bullish_stack:
        buy_score += 2
        reasons_buy.append("EMA 5/13/34 bullish stack")
    if ema_bearish_stack:
        sell_score += 2
        reasons_sell.append("EMA 5/13/34 bearish stack")

    # ══════════════════════════════════════════════════
    # RULE 2: MOMENTUM MUST CONFIRM DIRECTION
    # If EMA says BUY, momentum must also be up (and vice versa)
    # ══════════════════════════════════════════════════

    # MACD histogram must agree with direction
    if ema_bullish_stack:
        if last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]:
            buy_score += 1.5
            reasons_buy.append("MACD histogram expanding bullish")
        elif last["macd_hist"] < 0:
            # MACD against EMA direction — KILL the signal
            buy_score -= 2
            reasons_buy.append("MACD disagrees (bearish) — weak setup")
    elif ema_bearish_stack:
        if last["macd_hist"] < 0 and last["macd_hist"] < prev["macd_hist"]:
            sell_score += 1.5
            reasons_sell.append("MACD histogram expanding bearish")
        elif last["macd_hist"] > 0:
            sell_score -= 2
            reasons_sell.append("MACD disagrees (bullish) — weak setup")

    # MACD crossover (fresh cross = strong)
    if prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]:
        if ema_bullish_stack:
            buy_score += 1.5
            reasons_buy.append("Fresh MACD bullish crossover")
    elif prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]:
        if ema_bearish_stack:
            sell_score += 1.5
            reasons_sell.append("Fresh MACD bearish crossover")

    # ══════════════════════════════════════════════════
    # RULE 3: RSI MUST NOT BE EXHAUSTED
    # Don't buy when overbought, don't sell when oversold
    # ══════════════════════════════════════════════════

    rsi = last["rsi7"]
    if ema_bullish_stack:
        if rsi > 75:
            # Overbought — DON'T BUY, momentum exhausted
            buy_score -= 3
            reasons_buy.append(f"RSI(7) overbought at {rsi:.0f} — DON'T ENTER")
        elif 45 < rsi < 70 and rsi > prev["rsi7"]:
            buy_score += 1
            reasons_buy.append(f"RSI(7) rising in sweet spot ({rsi:.0f})")
        elif rsi < 35:
            # RSI too low for a bullish scalp — trend may be reversing
            buy_score -= 1
    elif ema_bearish_stack:
        if rsi < 25:
            sell_score -= 3
            reasons_sell.append(f"RSI(7) oversold at {rsi:.0f} — DON'T ENTER")
        elif 30 < rsi < 55 and rsi < prev["rsi7"]:
            sell_score += 1
            reasons_sell.append(f"RSI(7) falling in sweet spot ({rsi:.0f})")
        elif rsi > 65:
            sell_score -= 1

    # ══════════════════════════════════════════════════
    # RULE 4: PRICE ACTION CONFIRMATION
    # Need candle momentum — not just indicators
    # ══════════════════════════════════════════════════

    # Consecutive candles in same direction
    green_count = sum(1 for i in range(-3, 0) if df["Close"].iloc[i] > df["Open"].iloc[i])
    red_count = sum(1 for i in range(-3, 0) if df["Close"].iloc[i] < df["Open"].iloc[i])

    if ema_bullish_stack and green_count >= 2:
        buy_score += 1
        reasons_buy.append(f"{green_count}/3 green candles — momentum confirmed")
    elif ema_bearish_stack and red_count >= 2:
        sell_score += 1
        reasons_sell.append(f"{red_count}/3 red candles — momentum confirmed")

    # Current candle must be in the right direction
    current_green = last["Close"] > last["Open"]
    current_red = last["Close"] < last["Open"]

    if ema_bullish_stack and current_red:
        buy_score -= 1  # Current candle is against our direction
    if ema_bearish_stack and current_green:
        sell_score -= 1

    # ══════════════════════════════════════════════════
    # RULE 5: VWAP CONFIRMATION
    # Price must be on the right side of VWAP
    # ══════════════════════════════════════════════════

    if pd.notna(last["vwap"]):
        if ema_bullish_stack and price > last["vwap"]:
            buy_score += 0.5
            reasons_buy.append(f"Above VWAP ({last['vwap']:.2f})")
        elif ema_bullish_stack and price < last["vwap"]:
            buy_score -= 1  # Below VWAP = don't buy
            reasons_buy.append("Below VWAP — risky buy")
        elif ema_bearish_stack and price < last["vwap"]:
            sell_score += 0.5
            reasons_sell.append(f"Below VWAP ({last['vwap']:.2f})")
        elif ema_bearish_stack and price > last["vwap"]:
            sell_score -= 1
            reasons_sell.append("Above VWAP — risky sell")

    # ══════════════════════════════════════════════════
    # RULE 6: STOCHASTIC — ONLY FOR TIMING, NOT DIRECTION
    # ══════════════════════════════════════════════════

    if ema_bullish_stack and last["stoch_k"] > last["stoch_d"] and last["stoch_k"] < 80:
        buy_score += 0.5
        reasons_buy.append(f"Stoch K crossing up ({last['stoch_k']:.0f})")
    elif ema_bearish_stack and last["stoch_k"] < last["stoch_d"] and last["stoch_k"] > 20:
        sell_score += 0.5
        reasons_sell.append(f"Stoch K crossing down ({last['stoch_k']:.0f})")

    # ══════════════════════════════════════════════════
    # RULE 7: VOLUME MUST BE PRESENT
    # No scalp on low volume — slippage risk
    # ══════════════════════════════════════════════════

    if pd.notna(last["vol_sma"]) and last["vol_sma"] > 0:
        if last["Volume"] > last["vol_sma"] * 1.3:
            buy_score += 0.5
            sell_score += 0.5
            if ema_bullish_stack:
                reasons_buy.append("High volume — confirms move")
            else:
                reasons_sell.append("High volume — confirms move")
        elif last["Volume"] < last["vol_sma"] * 0.5:
            # Very low volume — avoid
            buy_score -= 1
            sell_score -= 1

    # ══════════════════════════════════════════════════
    # RULE 8: ADX — ONLY SCALP IN TRENDING MARKET
    # ══════════════════════════════════════════════════

    if pd.notna(last["adx"]):
        if last["adx"] > 20:
            buy_score += 0.5
            sell_score += 0.5
            if ema_bullish_stack:
                reasons_buy.append(f"ADX {last['adx']:.0f} — trending market")
            else:
                reasons_sell.append(f"ADX {last['adx']:.0f} — trending market")
        elif last["adx"] < 15:
            # Range-bound — bad for scalping
            buy_score -= 1.5
            sell_score -= 1.5

    # ══════════════════════════════════════════════════
    # FINAL DECISION — STRICT THRESHOLD
    # Only ONE direction allowed. If both have score, take NONE.
    # ══════════════════════════════════════════════════

    threshold = 5.0  # Must score 5+ out of ~8.5 possible

    # CRITICAL: Only allow signal in ONE direction
    signal = None
    if buy_score >= threshold and sell_score < 2:
        sl_pts = cfg["sl_points"]
        sl = round(price - sl_pts, 2)
        target_pts = cfg.get("target_points", cfg.get("default_target", 10))

        targets = []
        if "target_options" in cfg:
            for tp in cfg["target_options"]:
                targets.append({"points": tp, "price": round(price + tp, 2)})
        else:
            targets.append({"points": target_pts, "price": round(price + target_pts, 2)})

        # Filter out negative reasons
        clean_reasons = [r for r in reasons_buy if "DON'T" not in r and "weak" not in r and "risky" not in r]

        signal = {
            "type": "SCALP",
            "action": "BUY",
            "name": cfg["name"],
            "symbol": symbol,
            "price": round(price, 2),
            "stop_loss": sl,
            "sl_points": sl_pts,
            "targets": targets,
            "score": round(buy_score, 1),
            "max_score": 10,
            "reasons": clean_reasons,
            "rsi": round(last["rsi7"], 1) if pd.notna(last["rsi7"]) else None,
            "ema_stack": "BULLISH",
            "momentum": round(last["momentum"], 2) if pd.notna(last["momentum"]) else None,
            "vwap": round(last["vwap"], 2) if pd.notna(last["vwap"]) else None,
            "atr": round(last["atr"], 2) if pd.notna(last["atr"]) else None,
            "label": cfg["label"],
        }

    elif sell_score >= threshold and buy_score < 2:
        sl_pts = cfg["sl_points"]
        sl = round(price + sl_pts, 2)
        target_pts = cfg.get("target_points", cfg.get("default_target", 10))

        targets = []
        if "target_options" in cfg:
            for tp in cfg["target_options"]:
                targets.append({"points": tp, "price": round(price - tp, 2)})
        else:
            targets.append({"points": target_pts, "price": round(price - target_pts, 2)})

        clean_reasons = [r for r in reasons_sell if "DON'T" not in r and "weak" not in r and "risky" not in r]

        signal = {
            "type": "SCALP",
            "action": "SELL",
            "name": cfg["name"],
            "symbol": symbol,
            "price": round(price, 2),
            "stop_loss": sl,
            "sl_points": sl_pts,
            "targets": targets,
            "score": round(sell_score, 1),
            "max_score": 10,
            "reasons": clean_reasons,
            "rsi": round(last["rsi7"], 1) if pd.notna(last["rsi7"]) else None,
            "ema_stack": "BEARISH",
            "momentum": round(last["momentum"], 2) if pd.notna(last["momentum"]) else None,
            "vwap": round(last["vwap"], 2) if pd.notna(last["vwap"]) else None,
            "atr": round(last["atr"], 2) if pd.notna(last["atr"]) else None,
            "label": cfg["label"],
        }

    return _build_result(cfg, df, last, buy_score, sell_score, reasons_buy, reasons_sell, signal)


def _build_result(cfg, df, last, buy_score, sell_score, reasons_buy, reasons_sell, signal):
    price = last["Close"]
    momentum_data = _get_momentum_info(df)

    summary = {
        "name": cfg["name"],
        "symbol": cfg.get("symbol", ""),
        "price": round(price, 2),
        "rsi7": round(last["rsi7"], 1) if pd.notna(last.get("rsi7")) else None,
        "ema5": round(last["ema5"], 2) if pd.notna(last.get("ema5")) else None,
        "ema13": round(last["ema13"], 2) if pd.notna(last.get("ema13")) else None,
        "ema34": round(last["ema34"], 2) if pd.notna(last.get("ema34")) else None,
        "ema_stack": (
            "BULLISH" if last.get("ema5") and last["ema5"] > last["ema13"] > last["ema34"]
            else "BEARISH" if last.get("ema5") and last["ema5"] < last["ema13"] < last["ema34"]
            else "MIXED — NO SCALP"
        ),
        "macd_hist": round(last["macd_hist"], 4) if pd.notna(last.get("macd_hist")) else None,
        "stoch_k": round(last["stoch_k"], 1) if pd.notna(last.get("stoch_k")) else None,
        "momentum": round(last["momentum"], 2) if pd.notna(last.get("momentum")) else None,
        "vwap": round(last["vwap"], 2) if pd.notna(last.get("vwap")) else None,
        "atr": round(last["atr"], 2) if pd.notna(last.get("atr")) else None,
        "adx": round(last["adx"], 1) if pd.notna(last.get("adx")) else None,
        "volume_spike": bool(pd.notna(last.get("vol_sma")) and last["vol_sma"] > 0 and last["Volume"] > last["vol_sma"] * 1.3),
        "green_streak": momentum_data["green_streak"],
        "red_streak": momentum_data["red_streak"],
        "buy_score": round(max(buy_score, 0), 1),
        "sell_score": round(max(sell_score, 0), 1),
        "buy_reasons": [r for r in reasons_buy if r],
        "sell_reasons": [r for r in reasons_sell if r],
        "target_config": cfg.get("target_options", [cfg.get("target_points")]),
        "sl_points": cfg["sl_points"],
        "label": cfg["label"],
    }

    return {"signal": signal, "summary": summary}


def _get_momentum_info(df):
    try:
        green_streak = all(df["Close"].iloc[i] > df["Open"].iloc[i] for i in range(-3, 0))
        red_streak = all(df["Close"].iloc[i] < df["Open"].iloc[i] for i in range(-3, 0))
    except Exception:
        green_streak = False
        red_streak = False
    return {"green_streak": green_streak, "red_streak": red_streak}
