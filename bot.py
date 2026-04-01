"""
ICT Scalp Bot
Exchange : Kraken Futures (krakenfutures via CCXT)
Pairs    : BTC/USD:USD  |  ETH/USD:USD
Trend    : 1H EMA50
Sweeps   : 15m liquidity sweeps
Entry    : 5m — 2e lower low/higher high, MSS (wick OF close), FVG, .618 Fibonacci
Risk     : 1% per trade  |  Min R:R 1:3
Loop     : elke 1 minuut
"""

import os, time, json, logging, schedule
from datetime import datetime, timezone
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import ccxt

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()])
log = logging.getLogger(__name__)

SYMBOLS=["BTC/USD:USD","ETH/USD:USD"]; RISK_PERCENT=1.0; MIN_RR=3.0; TP_RR=3.0
SL_BUFFER_PCT=0.1; SWING_LOOKBACK=10; H1_EMA_LEN=50; MSS_LOOKBACK=20
SWEEP_LOOKBACK=50; SECOND_LOOKBACK=30; LOOP_INTERVAL=1
TF_TREND="1h"; TF_SWEEP="15m"; TF_ENTRY="5m"

exchange = ccxt.krakenfutures({
    "apiKey": os.getenv("KRAKEN_API_KEY"),
    "secret": os.getenv("KRAKEN_PRIVATE_KEY"),
    "enableRateLimit": True,
})

def fetch_ohlcv(symbol, tf, limit=350):
    try:
        raw = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.sort_values("ts").reset_index(drop=True)
    except Exception as e:
        log.error(f"fetch_ohlcv ({symbol} {tf}): {e}"); return pd.DataFrame()

def calc_ema(s, n): return s.ewm(span=n, adjust=False).mean()

def pivot_high(s, n):
    res = pd.Series(np.nan, index=s.index)
    for i in range(n, len(s)-n):
        if s.iloc[i] == s.iloc[i-n:i+n+1].max(): res.iloc[i] = s.iloc[i]
    return res

def pivot_low(s, n):
    res = pd.Series(np.nan, index=s.index)
    for i in range(n, len(s)-n):
        if s.iloc[i] == s.iloc[i-n:i+n+1].min(): res.iloc[i] = s.iloc[i]
    return res

def round_to_tick(price, symbol):
    try:
        tick = exchange.market(symbol)["precision"]["price"]
        if not tick or tick <= 0: tick = 0.5
    except: tick = 0.5
    return round(round(price / tick) * tick, 10)

def check_setup(symbol):
    log.info(f"--- Check {symbol} ---")
    de = fetch_ohlcv(symbol, TF_ENTRY, 350)
    ds = fetch_ohlcv(symbol, TF_SWEEP, 100)
    dt = fetch_ohlcv(symbol, TF_TREND, 100)
    if de.empty or ds.empty or dt.empty: return None

    dt["ema50"] = calc_ema(dt["close"], H1_EMA_LEN)
    h1_bull = dt["close"].iloc[-1] > dt["ema50"].iloc[-1]
    h1_bear = dt["close"].iloc[-1] < dt["ema50"].iloc[-1]
    log.info(f"1H: {'BULL' if h1_bull else 'BEAR'} | Close={dt['close'].iloc[-1]:.2f} EMA50={dt['ema50'].iloc[-1]:.2f}")

    lb = min(SWEEP_LOOKBACK, len(ds)-1)
    sell_sweep = any(ds["low"].iloc[i] <= ds["low"].iloc[max(0,i-10):i].min() and
        ds["close"].iloc[i] > ds["low"].iloc[max(0,i-10):i].min()
        for i in range(max(1,len(ds)-lb), len(ds)) if i > 10)
    buy_sweep = any(ds["high"].iloc[i] >= ds["high"].iloc[max(0,i-10):i].max() and
        ds["close"].iloc[i] < ds["high"].iloc[max(0,i-10):i].max()
        for i in range(max(1,len(ds)-lb), len(ds)) if i > 10)
    log.info(f"15m sweeps: sellside={sell_sweep} buyside={buy_sweep}")

    de["ph"] = pivot_high(de["high"], SWING_LOOKBACK)
    de["pl"] = pivot_low(de["low"], SWING_LOOKBACK)
    pl_idx = de["pl"].dropna().index.tolist()
    ph_idx = de["ph"].dropna().index.tolist()
    if len(pl_idx) < 2 or len(ph_idx) < 2: return None

    last_pl=de.loc[pl_idx[-1],"pl"]; prev_pl=de.loc[pl_idx[-2],"pl"]
    last_ph=de.loc[ph_idx[-1],"ph"]; prev_ph=de.loc[ph_idx[-2],"ph"]
    last_pl_bar=pl_idx[-1]; last_ph_bar=ph_idx[-1]
    sec_low  = last_pl <= prev_pl*1.002 and (len(de)-1-last_pl_bar) < SECOND_LOOKBACK
    sec_high = last_ph >= prev_ph*0.998 and (len(de)-1-last_ph_bar) < SECOND_LOOKBACK

    # MSS detectie — wick OF candle close door de pivot is geldig
    bull_bar=bear_bar=None; bull_h=bull_l=bear_h=bear_l=None
    for i in range((last_pl_bar if sec_low else 0)+1, len(de)):
        # Long MSS: wick of close boven laatste pivot high
        if de["high"].iloc[i] > last_ph or de["close"].iloc[i] > last_ph:
            bull_bar=i; bull_h=de["high"].iloc[i]; bull_l=de["low"].iloc[i]; break

    for i in range((last_ph_bar if sec_high else 0)+1, len(de)):
        # Short MSS: wick of close onder laatste pivot low
        if de["low"].iloc[i] < last_pl or de["close"].iloc[i] < last_pl:
            bear_bar=i; bear_h=de["high"].iloc[i]; bear_l=de["low"].iloc[i]; break

    bull_mss = bull_bar is not None and (len(de)-1-bull_bar) < MSS_LOOKBACK
    bear_mss = bear_bar is not None and (len(de)-1-bear_bar) < MSS_LOOKBACK

    # FVG — elke tick geldig
    bull_fvg=bear_fvg=False
    if bull_bar and bull_bar >= 2:
        for i in range(bull_bar, min(bull_bar+5,len(de))):
            if de["low"].iloc[i] > de["high"].iloc[i-2]: bull_fvg=True; break
    if bear_bar and bear_bar >= 2:
        for i in range(bear_bar, min(bear_bar+5,len(de))):
            if de["low"].iloc[i-2] > de["high"].iloc[i]: bear_fvg=True; break

    long_fib  = (bull_h-(bull_h-last_pl)*0.618) if bull_mss and bull_h else None
    short_fib = (bear_l+(last_ph-bear_l)*0.618)  if bear_mss and bear_l else None
    cur_lo=de["low"].iloc[-1]; cur_hi=de["high"].iloc[-1]

    long_ok  = h1_bull and sell_sweep and sec_low  and bull_mss and bull_fvg and long_fib  is not None and cur_lo <= long_fib  <= cur_hi
    short_ok = h1_bear and buy_sweep  and sec_high and bear_mss and bear_fvg and short_fib is not None and cur_lo <= short_fib <= cur_hi
    log.info(f"MSS: bull={bull_mss} bear={bear_mss} | Long={long_ok} | Short={short_ok}")
    if not long_ok and not short_ok: return None

    if long_ok:
        entry=long_fib; sl=round_to_tick(last_pl*(1-SL_BUFFER_PCT/100),symbol)
        risk=abs(entry-sl); tp=entry+risk*TP_RR; rr=(tp-entry)/risk if risk>0 else 0; side="long"
    else:
        entry=short_fib; sl=round_to_tick(last_ph*(1+SL_BUFFER_PCT/100),symbol)
        risk=abs(entry-sl); tp=entry-risk*TP_RR; rr=(entry-tp)/risk if risk>0 else 0; side="short"

    if rr < MIN_RR: log.info(f"{symbol}: R:R={rr:.2f} < {MIN_RR} — overgeslagen"); return None
    log.info(f"SETUP {side.upper()}: entry={entry:.2f} sl={sl:.2f} tp={tp:.2f} R:R={rr:.2f}")
    return {"side":side,"entry":entry,"sl":sl,"tp":tp,"symbol":symbol,"rr":rr}

def get_balance():
    try:
        bal=exchange.fetch_balance()
        usd=float(bal.get("USD",{}).get("free",0) or 0)
        eur=float(bal.get("EUR",{}).get("free",0) or 0)
        flex=float(bal.get("FLEX",{}).get("free",0) or 0)
        total=usd+eur+flex
        log.info(f"Saldo: USD={usd:.2f} EUR={eur:.2f} FLEX={flex:.2f} | Totaal={total:.2f}")
        return total
    except Exception as e: log.error(f"Balance fout: {e}"); return 0.0

def has_position(symbol):
    try: return any(float(p.get("contracts") or 0)!=0 for p in exchange.fetch_positions([symbol]))
    except: return False

def place_trade(setup):
    sym=setup["symbol"]; side=setup["side"]
    entry=setup["entry"]; sl=setup["sl"]; tp=setup["tp"]; rr=setup["rr"]
    if has_position(sym): log.info(f"{sym}: al open positie"); return
    balance=get_balance()
    if balance <= 0: log.error("Geen saldo"); return
    try:
        risk=balance*(RISK_PERCENT/100); dist=abs(entry-sl)
        if dist==0: log.error("SL=0"); return
        qty=float(exchange.amount_to_precision(sym, risk/dist))
        if qty<=0: log.error("qty=0"); return
        eside="buy" if side=="long" else "sell"
        xside="sell" if side=="long" else "buy"
        try: exchange.cancel_all_orders(sym)
        except: pass
        o=exchange.create_order(sym,"market",eside,qty)
        log.info(f"Entry {eside} {qty} {sym} | R:R={rr:.2f} | id={o['id']}")
        time.sleep(1)
        exchange.create_order(sym,"stop",xside,qty,sl,params={"reduceOnly":True,"stopPrice":sl})
        log.info(f"SL={sl:.2f}")
        exchange.create_order(sym,"limit",xside,qty,tp,params={"reduceOnly":True})
        log.info(f"TP={tp:.2f}")
        with open("trades.json","a") as f:
            f.write(json.dumps({"time":datetime.now(timezone.utc).isoformat(),
                "sym":sym,"side":side,"entry":entry,"sl":sl,"tp":tp,"qty":qty,"rr":rr})+"\n")
        log.info("Trade geplaatst!")
    except ccxt.InsufficientFunds: log.error("Onvoldoende saldo")
    except ccxt.BaseError as e: log.error(f"Exchange fout: {e}")

def run_bot():
    log.info(f"=== Run {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===")
    get_balance()
    for sym in SYMBOLS:
        try:
            s=check_setup(sym)
            if s: place_trade(s)
            else: log.info(f"{sym}: geen setup")
        except Exception as e: log.error(f"{sym} fout: {e}", exc_info=True)
    log.info("=== Klaar ===\n")

if __name__ == "__main__":
    log.info("ICT Scalp Bot | 1H/15m/5m | MSS wick+close | Min R:R 1:3")
    run_bot()
    schedule.every(LOOP_INTERVAL).minutes.do(run_bot)
    while True:
        schedule.run_pending()
        time.sleep(10)
