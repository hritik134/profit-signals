import time
import os
import sys
import winsound
from datetime import datetime

from analyzer import fetch_data, generate_signal
from config import SCAN_INTERVAL

INSTRUMENTS = [
    {"symbol": "GC=F", "name": "Gold Futures (XAU/USD)"},
]

LOG_FILE = os.path.join(os.path.dirname(__file__), "gold_signals.log")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def beep_alert(action):
    try:
        freq = 1000 if action == "BUY" else 600
        for _ in range(5):
            winsound.Beep(freq, 200)
            time.sleep(0.05)
    except Exception:
        pass


def log_signal(sig):
    with open(LOG_FILE, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'='*60}\n")
        f.write(f"[{ts}] {sig['strength']} {sig['action']} — {sig['name']}\n")
        f.write(f"Price: ${sig['price']}  |  SL: ${sig['stop_loss']}\n")
        f.write(f"T1: ${sig['target_1']} ({sig['rr_t1']})  |  T2: ${sig['target_2']} ({sig['rr_t2']})\n")
        f.write(f"Score: {sig['score']}/{sig['max_score']}  |  RSI: {sig['rsi']}  |  ADX: {sig['adx']}\n")
        f.write(f"HTF: {sig['htf_trend']}  |  Structure: {sig['structure']}  |  Supertrend: {sig['supertrend_dir']}\n")
        f.write(f"Reasons: {', '.join(sig['reasons'])}\n")
        f.write(f"Filters: {sig['filters_passed']}\n")


def print_header():
    print(f"\n{BOLD}{YELLOW}{'='*65}")
    print(f"   GOLD MONITOR  |  XAU/USD Futures  |  UPGRADED v2.0")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}{RESET}\n")


def print_signal(sig):
    action = sig["action"]
    color = GREEN if action == "BUY" else RED
    icon = ">>>" if sig["strength"] == "STRONG" else ">>"

    print(f"\n{BOLD}{color}{'='*60}")
    print(f"  {icon} {sig['strength']} {action} SIGNAL: {sig['name']}")
    print(f"{'='*60}{RESET}")

    print(f"\n  {BOLD}{WHITE}---- ENTRY ----{RESET}")
    print(f"  Price:          ${sig['price']}")

    print(f"\n  {BOLD}{WHITE}---- RISK MANAGEMENT ----{RESET}")
    print(f"  Stop Loss:      ${sig['stop_loss']}")
    print(f"  Risk/unit:      ${sig['risk_per_unit']}")
    print(f"  Suggested Qty:  {sig['qty_suggestion']} units (1% of 1L capital)")

    print(f"\n  {BOLD}{WHITE}---- TARGETS ----{RESET}")
    print(f"  Target 1:       ${sig['target_1']}  |  R:R = {sig['rr_t1']}  |  Profit: ${sig['reward_t1']}/unit")
    print(f"  Target 2:       ${sig['target_2']}  |  R:R = {sig['rr_t2']}  |  Profit: ${sig['reward_t2']}/unit")

    print(f"\n  {BOLD}{WHITE}---- SIGNAL STRENGTH ----{RESET}")
    print(f"  Score:          {sig['score']}/{sig['max_score']}")
    print(f"  RSI:            {sig['rsi']}")
    print(f"  ADX:            {sig['adx']} ({'Strong trend' if sig['trend_strong'] else 'Weak trend'})" if sig['adx'] else "")
    print(f"  Supertrend:     {sig['supertrend_dir']}")

    print(f"\n  {BOLD}{WHITE}---- CONFIRMATIONS ----{RESET}")
    print(f"  1H Trend:       {sig['htf_trend']}")
    print(f"  Market Structure:{sig['structure']}")
    print(f"  200 SMA:        {'ABOVE (bullish)' if sig['filters_passed']['200_sma'] else 'BELOW (bearish)'}")
    print(f"  Support:        ${sig['support']}")
    print(f"  Resistance:     ${sig['resistance']}")

    print(f"\n  {BOLD}{WHITE}---- WHY THIS TRADE ----{RESET}")
    for i, r in enumerate(sig["reasons"], 1):
        print(f"    {i}. {r}")

    print(f"\n  {BOLD}{WHITE}---- HOW TO TRADE ----{RESET}")
    if action == "BUY":
        print(f"    1. BUY at ${sig['price']} or market price")
        print(f"    2. Set STOP LOSS at ${sig['stop_loss']}")
        print(f"    3. Book 50% at Target 1 (${sig['target_1']})")
        print(f"    4. Move SL to entry price (breakeven)")
        print(f"    5. Let remaining 50% ride to Target 2 (${sig['target_2']})")
        print(f"    6. MAX LOSS: ${sig['risk_per_unit']}/unit  |  MAX GAIN: ${sig['reward_t2']}/unit")
    else:
        print(f"    1. SELL/SHORT at ${sig['price']} or market price")
        print(f"    2. Set STOP LOSS at ${sig['stop_loss']}")
        print(f"    3. Book 50% at Target 1 (${sig['target_1']})")
        print(f"    4. Move SL to entry price (breakeven)")
        print(f"    5. Let remaining 50% ride to Target 2 (${sig['target_2']})")
        print(f"    6. MAX LOSS: ${sig['risk_per_unit']}/unit  |  MAX GAIN: ${sig['reward_t2']}/unit")

    print(f"\n{color}{'='*60}{RESET}\n")


def print_overview(s):
    print(f"  {BOLD}Gold Futures (XAU/USD){RESET}")
    print(f"  {'─'*50}")
    print(f"  Price:          ${s['price']}")
    print(f"  RSI:            {s['rsi']}")
    ema_c = GREEN if s["ema_trend"] == "BULLISH" else RED
    st_c = GREEN if s["supertrend"] == "BULLISH" else RED
    htf_c = GREEN if s["htf_trend"] == "BULLISH" else RED if s["htf_trend"] == "BEARISH" else YELLOW
    print(f"  EMA Trend:      {ema_c}{s['ema_trend']}{RESET}")
    print(f"  Supertrend:     {st_c}{s['supertrend']}{RESET}")
    print(f"  MACD Hist:      {s['macd_hist']}")
    print(f"  ADX:            {s['adx']}")
    print(f"  Structure:      {s['structure']}")
    print(f"  1H Trend:       {htf_c}{s['htf_trend']}{RESET}")
    print(f"  200 SMA:        {'ABOVE' if s['above_200sma'] else 'BELOW'}")
    print(f"  Support:        ${s['support']}")
    print(f"  Resistance:     ${s['resistance']}")
    print(f"  BB Position:    {s['bb_position']}")
    print()
    print(f"  {BOLD}Scores:{RESET}")
    print(f"  Buy Score:      {GREEN}{s['buy_score']:.1f}/12{RESET}  (need 4.0 for signal)")
    print(f"  Sell Score:     {RED}{s['sell_score']:.1f}/12{RESET}  (need 4.0 for signal)")
    print()

    if s["buy_reasons"]:
        print(f"  {GREEN}Bullish factors:{RESET}")
        for r in s["buy_reasons"]:
            print(f"    + {r}")
    if s["sell_reasons"]:
        print(f"  {RED}Bearish factors:{RESET}")
        for r in s["sell_reasons"]:
            print(f"    - {r}")

    print()
    needed_buy = max(0, 4 - s["buy_score"])
    needed_sell = max(0, 4 - s["sell_score"])
    if needed_buy <= 1.5:
        print(f"  {BOLD}{GREEN}>>> BUY signal forming! Need {needed_buy:.1f} more points — WATCH CLOSELY{RESET}")
    elif needed_sell <= 1.5:
        print(f"  {BOLD}{RED}>>> SELL signal forming! Need {needed_sell:.1f} more points — WATCH CLOSELY{RESET}")
    else:
        closer = "BUY" if s["buy_score"] > s["sell_score"] else "SELL"
        needed = min(needed_buy, needed_sell)
        print(f"  {DIM}Nearest signal: {closer} (needs {needed:.1f} more points){RESET}")


def main():
    print(f"\n{BOLD}{YELLOW}")
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║   GOLD FUTURES MONITOR v2.0  (XAU/USD)          ║")
    print("  ║                                                  ║")
    print("  ║   12 indicators | Multi-timeframe | S/R levels   ║")
    print("  ║   Candle patterns | Supertrend | Structure       ║")
    print("  ║   BEEP alert + logged to gold_signals.log        ║")
    print("  ║                                                  ║")
    print("  ║   Press Ctrl+C to stop                           ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print(f"{RESET}")
    print(f"  Scan interval: {SCAN_INTERVAL}s | Threshold: 4/12\n")

    scan_count = 0
    try:
        while True:
            scan_count += 1
            clear_screen()
            print_header()
            print(f"  {DIM}Scan #{scan_count} — fetching 15m + 1H data...{RESET}")

            signals = []
            summaries = []
            for item in INSTRUMENTS:
                df = fetch_data(item["symbol"])
                if df is None:
                    continue
                result = generate_signal(df, item["name"], symbol=item["symbol"])
                if result is None:
                    continue
                if result["signal"]:
                    signals.append(result["signal"])
                summaries.append(result["summary"])

            clear_screen()
            print_header()

            if signals:
                beep_alert(signals[0]["action"])
                print(f"  {BOLD}{YELLOW}*** GOLD TRADE SIGNAL DETECTED ***{RESET}\n")
                for sig in signals:
                    print_signal(sig)
                    log_signal(sig)
            else:
                print(f"  {DIM}No trade signal. Monitoring...{RESET}\n")

            print(f"  {BOLD}Gold Analysis:{RESET}\n")
            if summaries:
                print_overview(summaries[0])
            else:
                print(f"  {DIM}No data (market may be closed){RESET}")

            print(f"\n  {DIM}Scan #{scan_count} | Next in {SCAN_INTERVAL}s | Ctrl+C to stop{RESET}\n")
            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n  {YELLOW}Stopped. Past signals in gold_signals.log{RESET}\n")


if __name__ == "__main__":
    main()
