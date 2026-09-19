
from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

APP_VERSION = "Alpha v1.0 — Autonomous 24/7 Paper Trading Engine"
KRAKEN_BASE = "https://api.kraken.com/0/public"

# Frozen before any real-data run in this release.
PREREGISTRATION = {
    "version": "v1.0-fast-batch-a",
    "mode": "paper_only",
    "data_source": "Kraken public REST",
    "markets_default": ["XBTUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "LINKUSD", "LTCUSD"],
    "timeframes_minutes_default": [5, 15, 60],
    "strategies": {
        "fast_breakout": {
            "entry": "EMA20>EMA50 and close breaks prior 20-bar high and volume>prior 20-bar median",
            "stop_atr": 1.50,
            "target_atr": 2.50,
        },
        "rsi_reclaim_fast": {
            "entry": "EMA50>EMA200 and RSI14 crosses upward through 50 and close>EMA20",
            "stop_atr": 1.30,
            "target_atr": 2.30,
        },
        "bollinger_reentry_fast": {
            "entry": "EMA50>EMA200 and previous close<previous lower Bollinger band and close re-enters above lower band",
            "stop_atr": 1.20,
            "target_atr": 2.00,
        },
    },
    "risk_defaults": {
        "risk_per_trade_pct": 0.25,
        "max_daily_loss_pct": 1.50,
        "max_portfolio_heat_pct": 1.25,
        "max_open_positions": 5,
        "max_gross_exposure_pct": 80.0,
        "max_position_notional_pct": 20.0,
        "fee_bps_each_side": 40.0,
        "slippage_bps_each_side": 5.0,
        "long_only": True,
        "leverage": 1.0,
    },
}
PREREGISTRATION_HASH = hashlib_sha = __import__("hashlib").sha256(
    json.dumps(PREREGISTRATION, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:16]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


@dataclass(frozen=True)
class Config:
    db_path: str
    markets: Tuple[str, ...]
    timeframes: Tuple[int, ...]
    scan_seconds: int
    starting_cash: float
    risk_per_trade: float
    max_daily_loss: float
    max_heat: float
    max_positions: int
    max_gross_exposure: float
    max_position_notional: float
    fee_rate: float
    slippage_rate: float
    min_notional: float

    @classmethod
    def from_env(cls) -> "Config":
        markets = tuple(
            x.strip().upper()
            for x in os.getenv(
                "ALPHA_MARKETS",
                ",".join(PREREGISTRATION["markets_default"]),
            ).split(",")
            if x.strip()
        )
        tfs = tuple(
            int(x.strip())
            for x in os.getenv("ALPHA_TIMEFRAMES", "5,15,60").split(",")
            if x.strip()
        )
        return cls(
            db_path=os.getenv("ALPHA_DB_PATH", "/data/alpha_v1.db"),
            markets=markets,
            timeframes=tfs,
            scan_seconds=max(30, env_int("ALPHA_SCAN_SECONDS", 60)),
            starting_cash=env_float("ALPHA_STARTING_CASH", 10000.0),
            risk_per_trade=env_float("ALPHA_RISK_PER_TRADE_PCT", 0.25) / 100.0,
            max_daily_loss=env_float("ALPHA_MAX_DAILY_LOSS_PCT", 1.50) / 100.0,
            max_heat=env_float("ALPHA_MAX_HEAT_PCT", 1.25) / 100.0,
            max_positions=env_int("ALPHA_MAX_POSITIONS", 5),
            max_gross_exposure=env_float("ALPHA_MAX_GROSS_EXPOSURE_PCT", 80.0) / 100.0,
            max_position_notional=env_float("ALPHA_MAX_POSITION_NOTIONAL_PCT", 20.0) / 100.0,
            fee_rate=env_float("ALPHA_FEE_BPS", 40.0) / 10000.0,
            slippage_rate=env_float("ALPHA_SLIPPAGE_BPS", 5.0) / 10000.0,
            min_notional=env_float("ALPHA_MIN_NOTIONAL", 25.0),
        )


CFG = Config.from_env()


class DB:
    def __init__(self, path: str, starting_cash: float):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init()
        self._ensure_account(starting_cash)

    def _init(self):
        c = self.conn.cursor()
        c.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS account(
              id INTEGER PRIMARY KEY CHECK(id=1),
              cash REAL NOT NULL,
              starting_cash REAL NOT NULL,
              realized_pnl REAL NOT NULL DEFAULT 0,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS positions(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              symbol TEXT NOT NULL,
              timeframe INTEGER NOT NULL,
              strategy TEXT NOT NULL,
              entry_time TEXT NOT NULL,
              entry_price REAL NOT NULL,
              qty REAL NOT NULL,
              stop_price REAL NOT NULL,
              target_price REAL NOT NULL,
              initial_risk REAL NOT NULL,
              entry_fee REAL NOT NULL,
              last_price REAL NOT NULL,
              signal_bar_ts INTEGER NOT NULL,
              UNIQUE(symbol, timeframe, strategy)
            );
            CREATE TABLE IF NOT EXISTS trades(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              symbol TEXT NOT NULL,
              timeframe INTEGER NOT NULL,
              strategy TEXT NOT NULL,
              entry_time TEXT NOT NULL,
              exit_time TEXT NOT NULL,
              entry_price REAL NOT NULL,
              exit_price REAL NOT NULL,
              qty REAL NOT NULL,
              gross_pnl REAL NOT NULL,
              fees REAL NOT NULL,
              net_pnl REAL NOT NULL,
              r_multiple REAL,
              exit_reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS signals(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT NOT NULL,
              symbol TEXT NOT NULL,
              timeframe INTEGER NOT NULL,
              strategy TEXT NOT NULL,
              signal_bar_ts INTEGER NOT NULL,
              status TEXT NOT NULL,
              detail TEXT,
              UNIQUE(symbol, timeframe, strategy, signal_bar_ts)
            );
            CREATE TABLE IF NOT EXISTS kv(
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def _ensure_account(self, starting_cash: float):
        row = self.conn.execute("SELECT * FROM account WHERE id=1").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO account(id,cash,starting_cash,realized_pnl,updated_at) VALUES(1,?,?,0,?)",
                (starting_cash, starting_cash, utcnow_iso()),
            )
            self.conn.commit()

    def account(self):
        return dict(self.conn.execute("SELECT * FROM account WHERE id=1").fetchone())

    def positions(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM positions ORDER BY id").fetchall()]

    def recent_trades(self, limit=100):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()]

    def recent_signals(self, limit=200):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()]

    def set_kv(self, key, value):
        self.conn.execute(
            "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_kv(self, key, default=None):
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return default

    def signal_exists(self, symbol, timeframe, strategy, signal_bar_ts):
        row = self.conn.execute(
            """SELECT 1 FROM signals WHERE symbol=? AND timeframe=? AND strategy=? AND signal_bar_ts=?""",
            (symbol, timeframe, strategy, int(signal_bar_ts)),
        ).fetchone()
        return bool(row)

    def add_signal(self, symbol, timeframe, strategy, signal_bar_ts, status, detail):
        try:
            self.conn.execute(
                """INSERT INTO signals(ts,symbol,timeframe,strategy,signal_bar_ts,status,detail)
                   VALUES(?,?,?,?,?,?,?)""",
                (utcnow_iso(), symbol, timeframe, strategy, int(signal_bar_ts), status, detail),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass

    def open_position(self, symbol, timeframe, strategy, entry_price, qty, stop_price, target_price, risk, fee, signal_bar_ts):
        with self.conn:
            acc = self.account()
            notional = entry_price * qty
            cash_after = acc["cash"] - notional - fee
            if cash_after < -1e-6:
                raise ValueError("insufficient cash")
            self.conn.execute(
                """INSERT INTO positions(symbol,timeframe,strategy,entry_time,entry_price,qty,stop_price,target_price,
                   initial_risk,entry_fee,last_price,signal_bar_ts)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (symbol,timeframe,strategy,utcnow_iso(),entry_price,qty,stop_price,target_price,
                 risk,fee,entry_price,int(signal_bar_ts)),
            )
            self.conn.execute(
                "UPDATE account SET cash=?, updated_at=? WHERE id=1",
                (cash_after, utcnow_iso()),
            )

    def update_last_price(self, pos_id, price):
        with self.conn:
            self.conn.execute("UPDATE positions SET last_price=? WHERE id=?", (price, pos_id))

    def close_position(self, pos_id, exit_price, exit_reason, fee_rate):
        with self.conn:
            row = self.conn.execute("SELECT * FROM positions WHERE id=?", (pos_id,)).fetchone()
            if row is None:
                return
            p = dict(row)
            exit_fee = exit_price * p["qty"] * fee_rate
            proceeds = exit_price * p["qty"] - exit_fee
            gross = (exit_price - p["entry_price"]) * p["qty"]
            fees = p["entry_fee"] + exit_fee
            net = gross - fees
            rmult = net / p["initial_risk"] if p["initial_risk"] > 0 else None
            acc = self.account()
            self.conn.execute(
                "UPDATE account SET cash=?, realized_pnl=?, updated_at=? WHERE id=1",
                (acc["cash"] + proceeds, acc["realized_pnl"] + net, utcnow_iso()),
            )
            self.conn.execute(
                """INSERT INTO trades(symbol,timeframe,strategy,entry_time,exit_time,entry_price,exit_price,qty,
                   gross_pnl,fees,net_pnl,r_multiple,exit_reason)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (p["symbol"],p["timeframe"],p["strategy"],p["entry_time"],utcnow_iso(),p["entry_price"],
                 exit_price,p["qty"],gross,fees,net,rmult,exit_reason),
            )
            self.conn.execute("DELETE FROM positions WHERE id=?", (pos_id,))

DBX = DB(CFG.db_path, CFG.starting_cash)


class Kraken:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "Alpha-v1-paper-research/1.0"})

    def _get(self, path: str, params: dict, timeout=12):
        r = self.s.get(f"{KRAKEN_BASE}/{path}", params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            raise RuntimeError("; ".join(data["error"]))
        return data["result"]

    def ohlc(self, symbol: str, interval: int) -> pd.DataFrame:
        res = self._get("OHLC", {"pair": symbol, "interval": interval})
        key = next(k for k in res.keys() if k != "last")
        rows = res[key]
        if len(rows) < 220:
            raise RuntimeError(f"not enough OHLC rows for {symbol}/{interval}m: {len(rows)}")
        df = pd.DataFrame(rows, columns=[
            "time","open","high","low","close","vwap","volume","count"
        ])
        for c in ["open","high","low","close","vwap","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("int64")
        df = df.dropna().sort_values("time").reset_index(drop=True)
        # Kraken's last row is the current, not-yet-committed candle.
        if len(df) > 1:
            df = df.iloc[:-1].copy()
        return df

    def last_price(self, symbol: str) -> float:
        res = self._get("Ticker", {"pair": symbol})
        key = next(iter(res))
        return float(res[key]["c"][0])

KRAKEN = Kraken()


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c = x["close"]
    h = x["high"]
    l = x["low"]

    x["ema20"] = c.ewm(span=20, adjust=False).mean()
    x["ema50"] = c.ewm(span=50, adjust=False).mean()
    x["ema200"] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    al = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = ag / al.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((al == 0) & (ag > 0)), 100.0)
    rsi = rsi.where(~((al == 0) & (ag == 0)), 50.0)
    x["rsi14"] = rsi

    prev_close = c.shift(1)
    tr = pd.concat(
        [(h-l).abs(), (h-prev_close).abs(), (l-prev_close).abs()], axis=1
    ).max(axis=1)
    x["atr14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std(ddof=0)
    x["bb_lower"] = ma20 - 2 * sd20
    x["prior_high20"] = h.shift(1).rolling(20).max()
    x["prior_volmed20"] = x["volume"].shift(1).rolling(20).median()
    return x


def frozen_signals(df: pd.DataFrame) -> List[Tuple[str,float,float,float]]:
    """Return (strategy, atr, stop_mult, target_mult) for latest completed bar."""
    x = indicators(df)
    if len(x) < 220:
        return []
    r = x.iloc[-1]
    p = x.iloc[-2]
    if not all(np.isfinite(r.get(k, np.nan)) for k in ["close","atr14","ema20","ema50","ema200","rsi14"]):
        return []

    out = []
    if (
        r["ema20"] > r["ema50"]
        and r["close"] > r["prior_high20"]
        and r["volume"] > r["prior_volmed20"]
    ):
        out.append(("fast_breakout", float(r["atr14"]), 1.50, 2.50))

    if (
        r["ema50"] > r["ema200"]
        and p["rsi14"] <= 50 < r["rsi14"]
        and r["close"] > r["ema20"]
    ):
        out.append(("rsi_reclaim_fast", float(r["atr14"]), 1.30, 2.30))

    if (
        r["ema50"] > r["ema200"]
        and np.isfinite(p["bb_lower"])
        and np.isfinite(r["bb_lower"])
        and p["close"] < p["bb_lower"]
        and r["close"] > r["bb_lower"]
    ):
        out.append(("bollinger_reentry_fast", float(r["atr14"]), 1.20, 2.00))
    return out


def marked_equity(prices: Optional[Dict[str,float]]=None) -> Tuple[float,float,float]:
    acc = DBX.account()
    cash = float(acc["cash"])
    gross = 0.0
    mtm = 0.0
    for p in DBX.positions():
        price = (prices or {}).get(p["symbol"], p["last_price"])
        gross += price * p["qty"]
        mtm += price * p["qty"]
    return cash + mtm, cash, gross


def current_heat(equity: float) -> float:
    if equity <= 0:
        return 1.0
    return sum(float(p["initial_risk"]) for p in DBX.positions()) / equity


def daily_realized_pnl() -> float:
    today = date.today().isoformat()
    row = DBX.conn.execute(
        "SELECT COALESCE(SUM(net_pnl),0) x FROM trades WHERE substr(exit_time,1,10)=?",
        (today,),
    ).fetchone()
    return float(row["x"])


def risk_gate(symbol: str, entry: float, stop: float) -> Tuple[bool,str,Optional[float],Optional[float]]:
    positions = DBX.positions()
    if len(positions) >= CFG.max_positions:
        return False, "max_positions", None, None
    if any(p["symbol"] == symbol for p in positions):
        return False, "symbol_already_open", None, None

    equity, cash, gross = marked_equity()
    if equity <= 0:
        return False, "nonpositive_equity", None, None
    if daily_realized_pnl() <= -CFG.max_daily_loss * equity:
        return False, "daily_loss_kill", None, None

    distance = entry - stop
    if distance <= 0:
        return False, "invalid_stop", None, None

    risk_budget = equity * CFG.risk_per_trade
    qty_risk = risk_budget / distance

    max_position_notional = equity * CFG.max_position_notional
    qty_position_cap = max_position_notional / entry

    gross_room = max(0.0, equity * CFG.max_gross_exposure - gross)
    qty_gross_cap = gross_room / entry

    cash_room = max(0.0, cash / (1.0 + CFG.fee_rate))
    qty_cash_cap = cash_room / entry

    qty = min(qty_risk, qty_position_cap, qty_gross_cap, qty_cash_cap)
    if qty <= 0:
        return False, "no_capacity", None, None
    notional = qty * entry
    if notional < CFG.min_notional:
        return False, "below_min_notional", None, None

    actual_risk = qty * distance
    if current_heat(equity) + actual_risk / equity > CFG.max_heat + 1e-12:
        return False, "portfolio_heat", None, None
    return True, "accepted", qty, actual_risk


async def manage_positions(price_cache: Dict[str,float]):
    for p in DBX.positions():
        try:
            px = price_cache.get(p["symbol"])
            if px is None:
                px = await asyncio.to_thread(KRAKEN.last_price, p["symbol"])
                price_cache[p["symbol"]] = px
            DBX.update_last_price(p["id"], px)
            if px <= p["stop_price"]:
                exit_px = px * (1 - CFG.slippage_rate)
                DBX.close_position(p["id"], exit_px, "STOP", CFG.fee_rate)
            elif px >= p["target_price"]:
                exit_px = px * (1 - CFG.slippage_rate)
                DBX.close_position(p["id"], exit_px, "TARGET", CFG.fee_rate)
        except Exception as e:
            DBX.set_kv("last_position_error", {"at": utcnow_iso(), "error": repr(e), "position_id": p["id"]})


async def scan_once():
    started = time.time()
    DBX.set_kv("engine_status", {"state":"running","started":utcnow_iso()})
    price_cache: Dict[str,float] = {}
    errors = []
    scanned = 0
    opportunities = 0
    accepted = 0
    rejected = 0

    await manage_positions(price_cache)

    for symbol in CFG.markets:
        for tf in CFG.timeframes:
            scanned += 1
            try:
                df = await asyncio.to_thread(KRAKEN.ohlc, symbol, tf)
                signal_bar_ts = int(df.iloc[-1]["time"])
                sigs = frozen_signals(df)
                for strategy, atr, stop_mult, target_mult in sigs:
                    opportunities += 1
                    if DBX.signal_exists(symbol, tf, strategy, signal_bar_ts):
                        continue
                    px = price_cache.get(symbol)
                    if px is None:
                        px = await asyncio.to_thread(KRAKEN.last_price, symbol)
                        price_cache[symbol] = px

                    entry = px * (1 + CFG.slippage_rate)
                    stop = entry - stop_mult * atr
                    target = entry + target_mult * atr

                    ok, reason, qty, risk = risk_gate(symbol, entry, stop)
                    if not ok:
                        rejected += 1
                        DBX.add_signal(symbol, tf, strategy, signal_bar_ts, "REJECTED", reason)
                        continue

                    fee = entry * qty * CFG.fee_rate
                    try:
                        DBX.open_position(
                            symbol, tf, strategy, entry, qty, stop, target, risk, fee, signal_bar_ts
                        )
                        accepted += 1
                        DBX.add_signal(
                            symbol, tf, strategy, signal_bar_ts, "OPENED",
                            json.dumps({
                                "entry": entry, "qty": qty, "stop": stop,
                                "target": target, "initial_risk": risk,
                            }, separators=(",", ":"))
                        )
                    except Exception as e:
                        rejected += 1
                        DBX.add_signal(symbol, tf, strategy, signal_bar_ts, "REJECTED", f"open_error:{e}")
            except Exception as e:
                errors.append(f"{symbol}/{tf}m: {e}")

    equity, cash, gross = marked_equity(price_cache)
    snapshot = {
        "state": "idle",
        "finished": utcnow_iso(),
        "seconds": round(time.time() - started, 2),
        "markets": len(CFG.markets),
        "timeframes": list(CFG.timeframes),
        "market_timeframe_scans": scanned,
        "setups_detected": opportunities,
        "opened": accepted,
        "rejected": rejected,
        "open_positions": len(DBX.positions()),
        "equity": equity,
        "cash": cash,
        "gross_exposure": gross,
        "portfolio_heat_pct": round(current_heat(equity)*100, 4),
        "daily_realized_pnl": daily_realized_pnl(),
        "errors": errors[-10:],
    }
    DBX.set_kv("engine_status", snapshot)
    DBX.set_kv("last_scan", snapshot)
    return snapshot


STOP_EVENT = asyncio.Event()


async def engine_loop():
    # Do not run a second engine after accidental duplicate startup in same process.
    if DBX.get_kv("loop_started_in_process"):
        return
    DBX.set_kv("loop_started_in_process", True)
    while not STOP_EVENT.is_set():
        try:
            await scan_once()
        except Exception as e:
            DBX.set_kv("engine_status", {"state":"error","at":utcnow_iso(),"error":repr(e)})
        try:
            await asyncio.wait_for(STOP_EVENT.wait(), timeout=CFG.scan_seconds)
        except asyncio.TimeoutError:
            pass


app = FastAPI(title=APP_VERSION)

@app.on_event("startup")
async def startup():
    # This release is deliberately designed for one Uvicorn worker/process.
    asyncio.create_task(engine_loop())

@app.on_event("shutdown")
async def shutdown():
    STOP_EVENT.set()


def status_payload():
    equity, cash, gross = marked_equity()
    acc = DBX.account()
    last = DBX.get_kv("last_scan", {})
    return {
        "version": APP_VERSION,
        "mode": "PAPER ONLY — no live order routing exists",
        "preregistration_hash": PREREGISTRATION_HASH,
        "markets": CFG.markets,
        "timeframes_minutes": CFG.timeframes,
        "scan_seconds": CFG.scan_seconds,
        "equity": equity,
        "cash": cash,
        "gross_exposure": gross,
        "realized_pnl_all_time": acc["realized_pnl"],
        "daily_realized_pnl": daily_realized_pnl(),
        "portfolio_heat_pct": current_heat(equity)*100,
        "open_positions": len(DBX.positions()),
        "last_scan": last,
        "risk": {
            "risk_per_trade_pct": CFG.risk_per_trade*100,
            "max_daily_loss_pct": CFG.max_daily_loss*100,
            "max_heat_pct": CFG.max_heat*100,
            "max_open_positions": CFG.max_positions,
            "max_gross_exposure_pct": CFG.max_gross_exposure*100,
            "max_position_notional_pct": CFG.max_position_notional*100,
            "fee_bps_each_side": CFG.fee_rate*10000,
            "slippage_bps_each_side": CFG.slippage_rate*10000,
            "leverage": 1.0,
            "shorts": False,
        },
    }


@app.get("/health")
def health():
    return {"ok": True, "version": APP_VERSION}


@app.get("/api/status")
def api_status():
    return JSONResponse(status_payload())


@app.get("/api/positions")
def api_positions():
    return JSONResponse(DBX.positions())


@app.get("/api/trades")
def api_trades(limit: int = 100):
    return JSONResponse(DBX.recent_trades(min(max(limit,1),500)))


@app.get("/api/signals")
def api_signals(limit: int = 200):
    return JSONResponse(DBX.recent_signals(min(max(limit,1),1000)))


DASHBOARD = r"""
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alpha v1.0</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;margin:24px;max-width:1100px;background:#0d1117;color:#e6edf3}
h1{font-size:34px;margin-bottom:4px}.muted{color:#8b949e}.warn{background:#2d2205;padding:14px;border-radius:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:20px 0}
.card{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:16px}
.big{font-size:28px;font-weight:700}.ok{color:#3fb950}.bad{color:#f85149}
table{width:100%;border-collapse:collapse;background:#161b22;border-radius:12px;overflow:hidden;margin-top:12px}
th,td{padding:9px;border-bottom:1px solid #30363d;font-size:13px;text-align:left}
code{color:#79c0ff}
</style>
</head>
<body>
<h1>Alpha v1.0 — 24/7 Paper Engine</h1>
<div class="muted">Autonomous scanner + deterministic risk governor + persistent paper broker.</div>
<p class="warn"><b>Paper only.</b> This build cannot send real orders and cannot use leverage.</p>
<div id="cards" class="grid"></div>
<h2>Open positions</h2><div id="positions"></div>
<h2>Recent trades</h2><div id="trades"></div>
<h2>Recent signals</h2><div id="signals"></div>
<script>
function n(x,d=2){return Number(x||0).toFixed(d)}
function table(rows, cols){
  if(!rows.length) return '<div class="muted">None</div>';
  return '<table><thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>'+
    rows.map(r=>'<tr>'+cols.map(c=>'<td>'+String(r[c]??'')+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
}
async function refresh(){
  const [s,p,t,g]=await Promise.all([
    fetch('/api/status').then(r=>r.json()),
    fetch('/api/positions').then(r=>r.json()),
    fetch('/api/trades?limit=20').then(r=>r.json()),
    fetch('/api/signals?limit=30').then(r=>r.json())
  ]);
  const ls=s.last_scan||{};
  document.getElementById('cards').innerHTML=[
    ['Equity','$'+n(s.equity)],
    ['P&L all time','$'+n(s.realized_pnl_all_time)],
    ['Open positions',s.open_positions],
    ['Heat',n(s.portfolio_heat_pct,3)+'%'],
    ['Scans / cycle',ls.market_timeframe_scans||0],
    ['Setups last cycle',ls.setups_detected||0],
    ['Opened last cycle',ls.opened||0],
    ['Rejected last cycle',ls.rejected||0],
  ].map(x=>'<div class="card"><div class="muted">'+x[0]+'</div><div class="big">'+x[1]+'</div></div>').join('');
  document.getElementById('positions').innerHTML=table(p,['symbol','timeframe','strategy','entry_price','qty','stop_price','target_price','last_price']);
  document.getElementById('trades').innerHTML=table(t,['exit_time','symbol','timeframe','strategy','net_pnl','r_multiple','exit_reason']);
  document.getElementById('signals').innerHTML=table(g,['ts','symbol','timeframe','strategy','status','detail']);
}
refresh(); setInterval(refresh,10000);
</script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD)
