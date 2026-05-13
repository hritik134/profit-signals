import time
import os
import sys
from datetime import datetime

from config import WATCHLIST, SCAN_INTERVAL
from analyzer import fetch_data, generate_signal


BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
MAGENTA = "\033[95m"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    print(f"\n{BOLD}{CYAN}{'='*70}")
    print(f"   TRADE SIGNAL MONITOR  |  Indian Market + XAU/USD")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}{RESET}\n")


def print_signal(sig):
    action = sig["action"]
    color = GREEN if action == "BUY" else RED
    strength_icon = ">>>" if sig["strength"] == "STRONG" else ">>"

    print(f"\n{BOLD}{color}{'='*60}")
    print(f"  {strength_icon} {sig['strength']} {action} SIGNAL: {sig['name']}")
    print(f"{'='*60}{RESET}")
    print(f"  {BOLD}Price:{RESET}      {sig['price']}")
    print(f"  {BOLD}Stop Loss:{RESET}  {sig['stop_loss']}")
    print(f"  {BOLD}Target 1:{RESET}   {sig['target_1']}")
    print(f"  {BOLD}Target 2:{RESET}   {sig['target_2']}")
    print(f"  {BOLD}Risk:Reward:{RESET} {sig['risk_reward']}")
    print(f"  {BOLD}Score:{RESET}      {sig['score']}/6")
    print(f"  {BOLD}RSI:{RESET}        {sig['rsi']}")
    if sig["adx"]:
        print(f"  {BOLD}ADX:{RESET}        {sig['adx']} ({'Strong trend' if sig['trend_strong'] else 'Weak trend'})")
    print(f"  {BOLD}Reasons:{RESET}")
    for r in sig["reasons"]:
        print(f"    - {r}")
    print(f"{color}{'-'*60}{RESET}\n")


def print_summary_table(summaries):
    print(f"  {BOLD}{DIM}{'Instrument':<20} {'Price':>10} {'RSI':>6} {'EMA':>8} {'MACD':>8} {'ADX':>6} {'Buy':>4} {'Sell':>4}{RESET}")
    print(f"  {DIM}{'-'*76}{RESET}")
    for s in summaries:
        rsi = s["rsi"] if s["rsi"] else "-"
        rsi_color = RED if (s["rsi"] and s["rsi"] > 70) else GREEN if (s["rsi"] and s["rsi"] < 30) else ""
        ema_color = GREEN if s["ema_trend"] == "BULLISH" else RED
        adx = s["adx"] if s["adx"] else "-"

        print(
            f"  {s['name']:<20} "
            f"{s['price']:>10} "
            f"{rsi_color}{str(rsi):>6}{RESET} "
            f"{ema_color}{s['ema_trend']:>8}{RESET} "
            f"{s['macd_hist'] if s['macd_hist'] else '-':>8} "
            f"{str(adx):>6} "
            f"{GREEN}{s['buy_score']:>4.1f}{RESET} "
            f"{RED}{s['sell_score']:>4.1f}{RESET}"
        )


def run_scan():
    signals = []
    summaries = []

    all_instruments = WATCHLIST["indian"] + WATCHLIST["gold"]

    for item in all_instruments:
        sym = item["symbol"]
        name = item["name"]
        sys.stdout.write(f"\r  Scanning {name:<25}")
        sys.stdout.flush()

        df = fetch_data(sym)
        if df is None:
            continue

        result = generate_signal(df, name)
        if result is None:
            continue

        if result["signal"]:
            signals.append(result["signal"])
        summaries.append(result["summary"])

    sys.stdout.write("\r" + " " * 40 + "\r")
    return signals, summaries


def main():
    print(f"\n{BOLD}{MAGENTA}")
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   KITE TRADE SIGNAL AGENT                   ║")
    print("  ║   Indian Market (NSE) + Gold (XAU/USD)      ║")
    print("  ║   Press Ctrl+C to stop                      ║")
    print("  ╚══════════════════════════════════════════════╝")
    print(f"{RESET}")
    print(f"  Data: Yahoo Finance (free, 15-min delayed for NSE)")
    print(f"  Scan interval: {SCAN_INTERVAL}s")
    print(f"  Strategy: RSI + MACD + EMA + Bollinger + StochRSI + Volume")
    print(f"  Signal threshold: 4/6 indicators must agree\n")

    scan_count = 0

    try:
        while True:
            scan_count += 1
            clear_screen()
            print_header()
            print(f"  {DIM}Scan #{scan_count} in progress...{RESET}")

            signals, summaries = run_scan()

            clear_screen()
            print_header()

            if signals:
                print(f"  {BOLD}{YELLOW}*** TRADE SIGNALS DETECTED ***{RESET}\n")
                for sig in signals:
                    print_signal(sig)
            else:
                print(f"  {DIM}No trade signals right now. Monitoring...{RESET}\n")

            print(f"\n  {BOLD}Market Overview:{RESET}\n")
            if summaries:
                print_summary_table(summaries)
            else:
                print(f"  {DIM}No data available (market may be closed){RESET}")

            print(f"\n  {DIM}Next scan in {SCAN_INTERVAL}s... (Ctrl+C to stop){RESET}")
            print(f"  {DIM}Tip: Edit config.py to add/remove instruments{RESET}\n")

            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Monitor stopped. Happy trading!{RESET}\n")


if __name__ == "__main__":
    main()
