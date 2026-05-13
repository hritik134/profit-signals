WATCHLIST = {
    "indian": [
        {"symbol": "^NSEI", "name": "NIFTY 50"},
        {"symbol": "^NSEBANK", "name": "BANK NIFTY"},
    ],
    "gold": [
        {"symbol": "GC=F", "name": "Gold Futures (XAU/USD)"},
    ],
}

# Technical analysis settings
TA_CONFIG = {
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "ema_short": 9,
    "ema_long": 21,
    "sma_200": 200,
    "bb_period": 20,
    "bb_std": 2,
    "atr_period": 14,
    "vwap_enabled": True,
}

# Scan interval in seconds
SCAN_INTERVAL = 300

# Signal strength thresholds
STRONG_SIGNAL_MIN_SCORE = 4  # out of 6 indicators agreeing
