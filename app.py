import json
import threading
import time
import numpy as np
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, Response

from analyzer import fetch_data, generate_signal
from scalper import generate_scalp_signal, fetch_scalp_data


IST = timezone(timedelta(hours=5, minutes=30))


def is_nse_open():
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def nse_next_open():
    now = datetime.now(IST)
    weekday = now.weekday()
    if weekday == 5:
        days_ahead = 2
    elif weekday == 6:
        days_ahead = 1
    elif now.hour >= 15 and now.minute > 30:
        days_ahead = 1
        if weekday == 4:
            days_ahead = 3
    else:
        days_ahead = 0
    next_day = now + timedelta(days=days_ahead)
    return next_day.replace(hour=9, minute=15, second=0, microsecond=0).strftime("%a %d %b, 9:15 AM IST")


def get_live_price(symbol):
    try:
        df = fetch_scalp_data(symbol, period="1d", interval="1m")
        if df is not None and len(df) > 0:
            last = df.iloc[-1]
            prev_close = df.iloc[0]["Open"]
            price = float(last["Close"])
            change = round(price - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2) if prev_close else 0
            return {"price": round(price, 2), "change": change, "change_pct": change_pct}
    except Exception:
        pass
    return None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app = Flask(__name__, template_folder="templates", static_folder="static")

INSTRUMENTS = {
    "gold": [{"symbol": "GC=F", "name": "Gold Futures (XAU/USD)"}],
    "india": [
        {"symbol": "^NSEI", "name": "NIFTY 50"},
        {"symbol": "^NSEBANK", "name": "BANK NIFTY"},
    ],
}

latest_data = {
    "signals": [],
    "summaries": [],
    "scalp_signals": [],
    "scalp_summaries": [],
    "active_trades": [],
    "live_prices": {},
    "last_scan": None,
    "scan_count": 0,
    "nse_open": False,
    "nse_next_open": "",
}
data_lock = threading.Lock()

active_trades = []


def check_active_trades(live_prices):
    global active_trades
    closed = []
    still_open = []

    for trade in active_trades:
        sym = trade["symbol"]
        lp = live_prices.get(sym)
        if not lp:
            still_open.append(trade)
            continue

        current_price = lp["price"]
        trade["current_price"] = current_price

        is_buy = trade["action"] == "BUY"

        if is_buy:
            trade["pnl"] = round(current_price - trade["entry_price"], 2)
            trade["pnl_pct"] = round((trade["pnl"] / trade["entry_price"]) * 100, 3)
            hit_sl = current_price <= trade["stop_loss"]
            hit_t1 = current_price >= trade["targets"][0]["price"]
        else:
            trade["pnl"] = round(trade["entry_price"] - current_price, 2)
            trade["pnl_pct"] = round((trade["pnl"] / trade["entry_price"]) * 100, 3)
            hit_sl = current_price >= trade["stop_loss"]
            hit_t1 = current_price <= trade["targets"][0]["price"]

        if hit_sl:
            trade["status"] = "SL HIT"
            trade["exit_price"] = current_price
            trade["exit_time"] = datetime.now().strftime("%H:%M:%S")
            closed.append(trade)
        elif hit_t1:
            trade["status"] = "TARGET HIT"
            trade["exit_price"] = current_price
            trade["exit_time"] = datetime.now().strftime("%H:%M:%S")
            closed.append(trade)
        else:
            trade["status"] = "ACTIVE"
            still_open.append(trade)

    active_trades = still_open
    return still_open, closed


def add_trade(signal):
    global active_trades

    for t in active_trades:
        if t["symbol"] == signal["symbol"] and t["action"] == signal["action"]:
            return

    trade = {
        "id": f"{signal['symbol']}_{int(time.time())}",
        "type": signal.get("type", "SWING"),
        "action": signal["action"],
        "name": signal["name"],
        "symbol": signal["symbol"],
        "entry_price": signal["price"],
        "stop_loss": signal["stop_loss"],
        "sl_points": signal.get("sl_points", abs(signal["price"] - signal["stop_loss"])),
        "targets": signal.get("targets", [{"points": 0, "price": signal.get("target_1", signal["price"])}]),
        "current_price": signal["price"],
        "pnl": 0,
        "pnl_pct": 0,
        "status": "ACTIVE",
        "entry_time": datetime.now().strftime("%H:%M:%S"),
        "reasons": signal.get("reasons", []),
        "score": signal.get("score", 0),
        "label": signal.get("label", ""),
    }
    active_trades.append(trade)


def background_scanner():
    global active_trades

    while True:
        signals = []
        summaries = []
        scalp_signals = []
        scalp_summaries = []
        live_prices = {}
        closed_trades = []

        nse_live = is_nse_open()

        # Fetch live prices for all instruments
        price_symbols = [("GC=F", "Gold", "$")]
        if nse_live:
            price_symbols += [("^NSEI", "NIFTY", ""), ("^NSEBANK", "BANKNIFTY", "")]

        for sym, name, label in price_symbols:
            lp = get_live_price(sym)
            if lp:
                lp["name"] = name
                lp["label"] = label
                live_prices[sym] = lp

        # Check active trades against live prices
        still_open, closed_trades = check_active_trades(live_prices)

        # Swing signals
        all_instruments = INSTRUMENTS["india"] + INSTRUMENTS["gold"]
        for item in all_instruments:
            is_indian = item["symbol"] in ("^NSEI", "^NSEBANK")
            if is_indian and not nse_live:
                continue
            try:
                df = fetch_data(item["symbol"])
                if df is None:
                    continue
                result = generate_signal(df, item["name"], symbol=item["symbol"])
                if result is None:
                    continue
                if result["signal"]:
                    result["signal"]["symbol"] = item["symbol"]
                    signals.append(result["signal"])
                    add_trade(result["signal"])
                summaries.append(result["summary"])
            except Exception as e:
                print(f"Error scanning {item['name']}: {e}")

        # Scalp signals
        scalp_symbols = ["GC=F"]
        if nse_live:
            scalp_symbols += ["^NSEI", "^NSEBANK"]

        for sym in scalp_symbols:
            try:
                result = generate_scalp_signal(sym)
                if result is None:
                    continue
                if result["signal"]:
                    scalp_signals.append(result["signal"])
                    add_trade(result["signal"])
                scalp_summaries.append(result["summary"])
            except Exception as e:
                print(f"Error scalp scanning {sym}: {e}")

        with data_lock:
            latest_data["signals"] = signals
            latest_data["summaries"] = summaries
            latest_data["scalp_signals"] = scalp_signals
            latest_data["scalp_summaries"] = scalp_summaries
            latest_data["active_trades"] = [t for t in active_trades]
            latest_data["closed_trades"] = closed_trades
            latest_data["live_prices"] = live_prices
            latest_data["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            latest_data["scan_count"] += 1
            latest_data["nse_open"] = nse_live
            latest_data["nse_next_open"] = nse_next_open() if not nse_live else ""

        time.sleep(30)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def get_data():
    with data_lock:
        return Response(
            json.dumps(latest_data, cls=NumpyEncoder),
            mimetype="application/json",
        )


if __name__ == "__main__":
    scanner_thread = threading.Thread(target=background_scanner, daemon=True)
    scanner_thread.start()
    print("\n  Trade Signal Web App running!")
    print("  Open on your phone: http://<your-pc-ip>:5000")
    print("  Open on this PC:    http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
