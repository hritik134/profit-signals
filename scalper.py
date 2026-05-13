import yfinance as yf
import pandas as pd
import numpy as np
import ta


def fetch_scalp_data(symbol, period="1d", interval="5m"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"  [!] Scalp fetch error {symbol}: {e}")
        return None


def compute_scalp_indicators(df):
    # Fast EMAs for scalping
    df["ema5"] = ta.trend.EMAIndicator(df["Close"], window=5).ema_indicator()
    df["ema13"] = ta.trend.EMAIndicator(df["Close"], window=13).ema_indicator()
    df["ema34"] = ta.trend.EMAIndicator(df["Close"], window=34).ema_indicator()

    # RSI short period
    df["rsi7"] = ta.momentum.RSIIndicator(df["Close"], window=7).rsi()

    # MACD fast
    macd = ta.trend.MACD(df["Close"], window_slow=17, window_fast=8, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Bollinger Bands tight
    bb = ta.volatility.BollingerBands(df["Close"], window=14, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # ATR for volatility
    df["atr"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=7).average_true_range()

    # Volume
    df["vol_sma"] = df["Volume"].rolling(window=10).mean()

    # Stochastic fast
    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=5, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Momentum (rate of change)
    df["momentum"] = df["Close"].pct_change(3) * 100

    # VWAP approximation
    df["vwap"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()

    # Candle body and wick
    df["body"] = abs(df["Close"] - df["Open"])
    df["candle_range"] = df["High"] - df["Low"]
    df["body_ratio"] = df["body"] / df["candle_range"].replace(0, np.nan)

    return df


def detect_scalp_momentum(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) > 3 else prev

    # Check for 3 consecutive green/red candles
    green_streak = all(df["Close"].iloc[i] > df["Open"].iloc[i] for i in range(-3, 0))
    red_streak = all(df["Close"].iloc[i] < df["Open"].iloc[i] for i in range(-3, 0))

    # Price acceleration: each candle bigger than last
    increasing_momentum_up = (
        df["Close"].iloc[-1] - df["Open"].iloc[-1] >
        df["Close"].iloc[-2] - df["Open"].iloc[-2] > 0
    )
    increasing_momentum_down = (
        df["Open"].iloc[-1] - df["Close"].iloc[-1] >
        df["Open"].iloc[-2] - df["Close"].iloc[-2] > 0
    )

    return {
        "green_streak": green_streak,
        "red_streak": red_streak,
        "increasing_up": increasing_momentum_up,
        "increasing_down": increasing_momentum_down,
    }


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
    if df is None or len(df) < 20:
        return None

    df = compute_scalp_indicators(df)
    momentum = detect_scalp_momentum(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last["Close"]

    buy_score = 0
    sell_score = 0
    reasons_buy = []
    reasons_sell = []

    # 1. EMA alignment (EMA5 > EMA13 > EMA34 = bullish stack)
    if last["ema5"] > last["ema13"] > last["ema34"]:
        buy_score += 2
        reasons_buy.append("EMA 5/13/34 bullish stack")
    elif last["ema5"] < last["ema13"] < last["ema34"]:
        sell_score += 2
        reasons_sell.append("EMA 5/13/34 bearish stack")

    # 2. EMA5 crossover EMA13 (fresh cross)
    if prev["ema5"] <= prev["ema13"] and last["ema5"] > last["ema13"]:
        buy_score += 1.5
        reasons_buy.append("EMA5 just crossed above EMA13")
    elif prev["ema5"] >= prev["ema13"] and last["ema5"] < last["ema13"]:
        sell_score += 1.5
        reasons_sell.append("EMA5 just crossed below EMA13")

    # 3. RSI 7 momentum
    if 40 < last["rsi7"] < 65 and last["rsi7"] > prev["rsi7"]:
        buy_score += 1
        reasons_buy.append(f"RSI(7) rising: {last['rsi7']:.1f}")
    elif 35 < last["rsi7"] < 60 and last["rsi7"] < prev["rsi7"]:
        sell_score += 1
        reasons_sell.append(f"RSI(7) falling: {last['rsi7']:.1f}")

    # 4. MACD histogram growing
    if last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]:
        buy_score += 1
        reasons_buy.append("MACD histogram expanding (bullish)")
    elif last["macd_hist"] < 0 and last["macd_hist"] < prev["macd_hist"]:
        sell_score += 1
        reasons_sell.append("MACD histogram expanding (bearish)")

    # 5. MACD crossover
    if prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]:
        buy_score += 1.5
        reasons_buy.append("MACD bullish cross")
    elif prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]:
        sell_score += 1.5
        reasons_sell.append("MACD bearish cross")

    # 6. Green/Red candle streak (momentum)
    if momentum["green_streak"]:
        buy_score += 1
        reasons_buy.append("3 consecutive green candles")
    if momentum["red_streak"]:
        sell_score += 1
        reasons_sell.append("3 consecutive red candles")

    # 7. Increasing candle size (acceleration)
    if momentum["increasing_up"]:
        buy_score += 0.5
        reasons_buy.append("Candles getting bigger (accelerating up)")
    if momentum["increasing_down"]:
        sell_score += 0.5
        reasons_sell.append("Candles getting bigger (accelerating down)")

    # 8. Volume spike
    if pd.notna(last["vol_sma"]) and last["vol_sma"] > 0 and last["Volume"] > last["vol_sma"] * 1.3:
        buy_score += 0.5
        sell_score += 0.5
        reasons_buy.append("Volume above average")
        reasons_sell.append("Volume above average")

    # 9. Price above/below VWAP
    if pd.notna(last["vwap"]):
        if last["Close"] > last["vwap"]:
            buy_score += 0.5
            reasons_buy.append(f"Price above VWAP ({last['vwap']:.2f})")
        else:
            sell_score += 0.5
            reasons_sell.append(f"Price below VWAP ({last['vwap']:.2f})")

    # 10. Stochastic confirmation
    if last["stoch_k"] > last["stoch_d"] and last["stoch_k"] < 80:
        buy_score += 0.5
        reasons_buy.append(f"Stoch K > D ({last['stoch_k']:.0f})")
    elif last["stoch_k"] < last["stoch_d"] and last["stoch_k"] > 20:
        sell_score += 0.5
        reasons_sell.append(f"Stoch K < D ({last['stoch_k']:.0f})")

    # Build signal
    signal = None
    threshold = 3.5

    label = cfg["label"]
    sl_pts = cfg["sl_points"]

    if "target_points" in cfg:
        target_pts = cfg["target_points"]
    else:
        target_pts = cfg["default_target"]

    if buy_score >= threshold:
        sl = round(price - sl_pts, 2)
        t1 = round(price + target_pts, 2)

        # For NIFTY/BANKNIFTY, multiple targets
        targets = []
        if "target_options" in cfg:
            for tp in cfg["target_options"]:
                targets.append({"points": tp, "price": round(price + tp, 2)})
        else:
            targets.append({"points": target_pts, "price": t1})

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
            "reasons": reasons_buy,
            "rsi": round(last["rsi7"], 1) if pd.notna(last["rsi7"]) else None,
            "ema_stack": "BULLISH" if last["ema5"] > last["ema13"] > last["ema34"] else "MIXED",
            "momentum": round(last["momentum"], 2) if pd.notna(last["momentum"]) else None,
            "vwap": round(last["vwap"], 2) if pd.notna(last["vwap"]) else None,
            "atr": round(last["atr"], 2) if pd.notna(last["atr"]) else None,
            "label": label,
        }

    elif sell_score >= threshold:
        sl = round(price + sl_pts, 2)

        targets = []
        if "target_options" in cfg:
            for tp in cfg["target_options"]:
                targets.append({"points": tp, "price": round(price - tp, 2)})
        else:
            targets.append({"points": target_pts, "price": round(price - target_pts, 2)})

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
            "reasons": reasons_sell,
            "rsi": round(last["rsi7"], 1) if pd.notna(last["rsi7"]) else None,
            "ema_stack": "BEARISH" if last["ema5"] < last["ema13"] < last["ema34"] else "MIXED",
            "momentum": round(last["momentum"], 2) if pd.notna(last["momentum"]) else None,
            "vwap": round(last["vwap"], 2) if pd.notna(last["vwap"]) else None,
            "atr": round(last["atr"], 2) if pd.notna(last["atr"]) else None,
            "label": label,
        }

    # Summary always returned
    summary = {
        "name": cfg["name"],
        "symbol": symbol,
        "price": round(price, 2),
        "rsi7": round(last["rsi7"], 1) if pd.notna(last["rsi7"]) else None,
        "ema5": round(last["ema5"], 2) if pd.notna(last["ema5"]) else None,
        "ema13": round(last["ema13"], 2) if pd.notna(last["ema13"]) else None,
        "ema34": round(last["ema34"], 2) if pd.notna(last["ema34"]) else None,
        "ema_stack": (
            "BULLISH" if last["ema5"] > last["ema13"] > last["ema34"]
            else "BEARISH" if last["ema5"] < last["ema13"] < last["ema34"]
            else "MIXED"
        ),
        "macd_hist": round(last["macd_hist"], 4) if pd.notna(last["macd_hist"]) else None,
        "stoch_k": round(last["stoch_k"], 1) if pd.notna(last["stoch_k"]) else None,
        "momentum": round(last["momentum"], 2) if pd.notna(last["momentum"]) else None,
        "vwap": round(last["vwap"], 2) if pd.notna(last["vwap"]) else None,
        "atr": round(last["atr"], 2) if pd.notna(last["atr"]) else None,
        "volume_spike": bool(pd.notna(last["vol_sma"]) and last["vol_sma"] > 0 and last["Volume"] > last["vol_sma"] * 1.3),
        "green_streak": momentum["green_streak"],
        "red_streak": momentum["red_streak"],
        "buy_score": round(buy_score, 1),
        "sell_score": round(sell_score, 1),
        "buy_reasons": reasons_buy,
        "sell_reasons": reasons_sell,
        "target_config": cfg.get("target_options", [cfg.get("target_points")]),
        "sl_points": sl_pts,
        "label": cfg["label"],
    }

    return {"signal": signal, "summary": summary}
