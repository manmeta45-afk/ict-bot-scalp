"""
ICT Trading Bot — Scalp Edition
Exchange : Kraken Futures (krakenfutures via CCXT)
Pairs    : BTC/USD:USD  |  ETH/USD:USD
Timeframe: 1m candles  |  15m sweeps  |  1H trend
Strategy : 15m sweep -> 2nd 1m low/high -> MSS -> FVG -> .618 + VWAP + CVD divergence
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

SYMBOLS         = ["BTC/USD:USD", "ETH/USD:USD"]
RISK_PERCENT    = 1.0
MIN_RR          = 3.0
TP_RR           = 3.0
SL_BUFFER_PCT   = 0.1
FVG_MIN_PCT     = 0.05
SWING_LOOKBACK  = 10
H1_EMA_LEN      = 50
MSS_LOOKBACK    = 20
SWEEP_LOOKBACK  = 50
SECOND_LOOKBACK = 30
CVD_LOOKBACK    = 20
LOOP_INTERVAL   = 1

exchange = ccxt.krakenfutures({
    "apiKey":          os.getenv("KRAKEN_API_KEY"),
    "secret":          os.getenv("KRAKEN_PRIVATE_KEY"),
    "enableRateLimit": True,
})


def fetch_ohlcv(symbol, timeframe, limit=350):
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.sort_values("ts").reset_index(drop=True)
    except Exception as e:
        log.error(f"fetch_ohlcv ({symbol} {timeframe}): {e}")
        return pd.DataFrame()


def calc_ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def calc_vwap(df):
    df = df.copy()
    df["date"]   = df["ts"].dt.date
    df["hlc3"]   = (df["high"] + df["low"] + df["close"]) / 3
    df["cvol"]   = df.groupby("date")["volume"].cumsum()
    df["ctpvol"] = df.groupby("date").apply(
        lambda g: (g["hlc3"] * g["volume"]).cumsum()
    ).reset_index(level=0, drop=True)
    return df["ctpvol"] / df["cvol"]


def pivot_high(series, n):
    res = pd.Series(np.nan, index=series.index)
    for i in range(n, len(series) - n):
        if series.iloc[i] == series.iloc[i-n:i+n+1].max():
            res.iloc[i] = series.iloc[i]
    return res


def pivot_low(series, n):
    res = pd.Series(np.nan, index=series.index)
    for i in range(n, len(series) - n):
        if series.iloc[i] == series.iloc[i-n:i+n+1].min():
            res.iloc[i] = series.iloc[i]
    return res


def calc_cvd(df):
    delta = np.where(df["close"] >= df["open"], df["volume"], -df["volume"])
    return pd.Series(delta, index=df.index).cumsum()


def check_setup(symbol):
    log.info(f"--- Check {symbol} ---")
    df1m  = fetch_ohlcv(symbol, "1m",  350)
    df15m = fetch_ohlcv(symbol, "15m", 100)
    df1h  = fetch_ohlcv(symbol, "1h",  100)
    if df1m.empty or df15m.empty or df1h.empty:
        return None

    df1h["ema50"] = calc_ema(df1h["close"], H1_EMA_LEN)
    h1_bull = df1h["close"].iloc[-1] > df1h["ema50"].iloc[-1]
    h1_bear = df1h["close"].iloc[-1] < df1h["ema50"].iloc[-1]
    log.info(f"1H: {'BULL' if h1_bull else 'BEAR'} | Close={df1h['close'].iloc[-1]:.2f} EMA50={df1h['ema50'].iloc[-1]:.2f}")

    lookback = min(SWEEP_LOOKBACK, len(df15m) - 1)
    sellside_seen = any(
        df15m["low"].iloc[i]  <= df15m["low"].iloc[max(0,i-10):i].min() and
        df15m["close"].iloc[i] > df15m["low"].iloc[max(0,i-10):i].min()
        for i in range(max(1, len(df15m) - lookback), len(df15m)) if i > 10
    )
    buyside_seen = any(
        df15m["high"].iloc[i]  >= df15m["high"].iloc[max(0,i-10):i].max() and
        df15m["close"].iloc[i]  < df15m["high"].iloc[max(0,i-10):i].max()
        for i in range(max(1, len(df15m) - lookback), len(df15m)) if i > 10
    )
    log.info(f"15m sweeps: sellside={sellside_seen} buyside={buyside_seen}")

    df1m["ph"]   = pivot_high(df1m["high"], SWING_LOOKBACK)
    df1m["pl"]   = pivot_low (df1m["low"],  SWING_LOOKBACK)
    df1m["vwap"] = calc_vwap(df1m)
    df1m["cvd"]  = calc_cvd(df1m)

    pl_idx = df1m["pl"].dropna().index.tolist()
    ph_idx = df1m["ph"].dropna().index.tolist()
    if len(pl_idx) < 2 or len(ph_idx) < 2:
        return None

    last_pl = df1m.loc[pl_idx[-1], "pl"]; prev_pl = df1m.loc[pl_idx[-2], "pl"]
    last_ph = df1m.loc[ph_idx[-1], "ph"]; prev_ph = df1m.loc[ph_idx[-2], "ph"]
    last_pl_bar = pl_idx[-1]; last_ph_bar = ph_idx[-1]

    second_low_seen  = last_pl <= prev_pl * 1.002 and (len(df1m)-1 - last_pl_bar) < SECOND_LOOKBACK
    second_high_seen = last_ph >= prev_ph * 0.998 and (len(df1m)-1 - last_ph_bar) < SECOND_LOOKBACK

    bull_mss_bar = bear_mss_bar = None
    bull_mss_high = bull_mss_low = bear_mss_high = bear_mss_low = None

    for i in range((last_pl_bar if second_low_seen else 0)+1, len(df1m)):
        if df1m["close"].iloc[i] > last_ph and df1m["close"].iloc[i-1] <= last_ph:
            bull_mss_bar = i; bull_mss_high = df1m["high"].iloc[i]; bull_mss_low = df1m["low"].iloc[i]

    for i in range((last_ph_bar if second_high_seen else 0)+1, len(df1m)):
        if df1m["close"].iloc[i] < last_pl and df1m["close"].iloc[i-1] >= last_pl:
            bear_mss_bar = i; bear_mss_high = df1m["high"].iloc[i]; bear_mss_low = df1m["low"].iloc[i]

    bull_mss_seen = bull_mss_bar is not None and (len(df1m)-1 - bull_mss_bar) < MSS_LOOKBACK
    bear_mss_seen = bear_mss_bar is not None and (len(df1m)-1 - bear_mss_bar) < MSS_LOOKBACK

    bull_fvg = bear_fvg = False
    bull_fvg_top = bull_fvg_bot = bear_fvg_top = bear_fvg_bot = None

    if bull_mss_bar and bull_mss_bar >= 2:
        for i in range(bull_mss_bar, min(bull_mss_bar+5, len(df1m))):
            bot = df1m["high"].iloc[i-2]; top = df1m["low"].iloc[i]
            if top > bot and (top-bot)/bot*100 >= FVG_MIN_PCT:
                bull_fvg=True; bull_fvg_top=top; bull_fvg_bot=bot; break

    if bear_mss_bar and bear_mss_bar >= 2:
        for i in range(bear_mss_bar, min(bear_mss_bar+5, len(df1m))):
            top = df1m["low"].iloc[i-2]; bot = df1m["high"].iloc[i]
            if top > bot and (top-bot)/top*100 >= FVG_MIN_PCT:
                bear_fvg=True; bear_fvg_top=top; bear_fvg_bot=bot; break

    long_fib  = (bull_mss_high-(bull_mss_high-last_pl)*0.618) if bull_mss_seen and bull_mss_high else None
    short_fib = (bear_mss_low+(last_ph-bear_mss_low)*0.618)   if bear_mss_seen and bear_mss_low  else None

    cur_low  = df1m["low"].iloc[-1]
    cur_high = df1m["high"].iloc[-1]
    cur_vwap = df1m["vwap"].iloc[-1]

    cvd_bull_seen = any(
        df1m["low"].iloc[i] <= df1m["low"].iloc[max(0,i-CVD_LOOKBACK):i].min()*1.001 and
        df1m["cvd"].iloc[i]  > df1m["cvd"].iloc[max(0,i-CVD_LOOKBACK):i].min()*1.001 and
        df1m["cvd"].iloc[i]  < 0
        for i in range(max(1, len(df1m)-MSS_LOOKBACK), len(df1m))
    )
    cvd_bear_seen = any(
        df1m["high"].iloc[i] >= df1m["high"].iloc[max(0,i-CVD_LOOKBACK):i].max()*0.999 and
        df1m["cvd"].iloc[i]   < df1m["cvd"].iloc[max(0,i-CVD_LOOKBACK):i].max()*0.999 and
        df1m["cvd"].iloc[i]   > 0
        for i in range(max(1, len(df1m)-MSS_LOOKBACK), len(df1m))
    )
    log.info(f"MSS: bull={bull_mss_seen} bear={bear_mss_seen} | CVD: bull={cvd_bull_seen} bear={cvd_bear_seen}")

    vwap_long_ok  = long_fib  is not None and last_pl <= cur_vwap <= long_fib
    vwap_short_ok = short_fib is not None and short_fib <= cur_vwap <= last_ph

    long_ok = (h1_bull and sellside_seen and second_low_seen and bull_mss_seen and
               bull_fvg and long_fib is not None and
               cur_low <= long_fib <= cur_high and vwap_long_ok and cvd_bull_seen)

    short_ok = (h1_bear and buyside_seen and second_high_seen and bear_mss_seen and
                bear_fvg and short_fib is not None and
                cur_low <= short_fib <= cur_high and vwap_short_ok and cvd_bear_seen)

    log.info(f"Long={long_ok} | Short={short_ok}")
    if not long_ok and not short_ok:
        return None

    if long_ok:
        entry=long_fib; sl=last_pl*(1-SL_BUFFER_PCT/100)
        risk=abs(entry-sl); tp=entry+risk*TP_RR
        rr=(tp-entry)/risk if risk>0 else 0; side="long"
    else:
        entry=short_fib; sl=last_ph*(1+SL_BUFFER_PCT/100)
        risk=abs(entry-sl); tp=entry-risk*TP_RR
        rr=(entry-tp)/risk if risk>0 else 0; side="short"

    if rr < MIN_RR:
        log.info(f"{symbol}: R:R={rr:.2f} < {MIN_RR} — overgeslagen"); return None

    log.info(f"SETUP {side.upper()}: entry={entry:.2f} sl={sl:.2f} tp={tp:.2f} R:R={rr:.2f}")
    return {"side":side,"entry":entry,"sl":sl,"tp":tp,"symbol":symbol,"rr":rr}


def get_balance():
    try:
        bal   = exchange.fetch_balance()
        usd   = float(bal.get("USD",  {}).get("free", 0) or 0)
        eur   = float(bal.get("EUR",  {}).get("free", 0) or 0)
        flex  = float(bal.get("FLEX", {}).get("free", 0) or 0)
        total = usd + eur + flex
        log.info(f"Saldo: USD={usd:.2f} EUR={eur:.2f} FLEX={flex:.2f} | Totaal={total:.2f}")
        return total
    except Exception as e:
        log.error(f"Balance fout ({type(e).__name__}): {e}"); return 0.0


def has_position(symbol):
    try:
        return any(float(p.get("contracts") or 0)!=0 for p in exchange.fetch_positions([symbol]))
    except: return False


def place_trade(setup):
    sym=setup["symbol"]; side=setup["side"]
    entry=setup["entry"]; sl=setup["sl"]; tp=setup["tp"]; rr=setup["rr"]
    if has_position(sym):
        log.info(f"{sym}: al open positie"); return
    balance = get_balance()
    if balance <= 0: log.error("Geen saldo"); return
    try:
        risk=balance*(RISK_PERCENT/100)
        dist=abs(entry-sl)
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
        exchange.create_order(sym,"limit",xside,qty,tp,params={"reduceOnly":True})
        log.info(f"SL={sl:.2f} TP={tp:.2f}")
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
            s = check_setup(sym)
            if s: place_trade(s)
            else: log.info(f"{sym}: geen setup")
        except Exception as e:
            log.error(f"{sym} fout: {e}", exc_info=True)
    log.info("=== Klaar ===\n")


if __name__ == "__main__":
    log.info("ICT Bot SCALP — Kraken Futures | 1m/15m/1H | CVD | Min R:R 1:3")
    run_bot()
    schedule.every(LOOP_INTERVAL).minutes.do(run_bot)
    while True:
        schedule.run_pending()
        time.sleep(10)
bot.py
