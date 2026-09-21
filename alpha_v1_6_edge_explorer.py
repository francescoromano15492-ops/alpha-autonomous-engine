from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

APP_VERSION = "Alpha v1.6 — Regime + Relative Strength Edge Explorer"
KRAKEN_BASE = "https://api.kraken.com/0/public"
BITSTAMP_BASE = "https://www.bitstamp.net/api/v2"
DEFAULT_MARKETS = [
    "XBTUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD", "LINKUSD", "LTCUSD",
    "AVAXUSD", "DOTUSD", "BCHUSD", "ATOMUSD", "XLMUSD", "UNIUSD", "AAVEUSD", "ETCUSD",
    "ALGOUSD", "NEARUSD", "FILUSD", "ICPUSD", "INJUSD", "SUIUSD", "ARBUSD", "OPUSD", "TRXUSD",
]
RESEARCH_MARKETS = DEFAULT_MARKETS[:12]
RESEARCH_TIMEFRAMES = (5, 15, 60)
RESEARCH_LOOKBACK_DAYS = max(30, min(180, int(os.getenv("ALPHA_RESEARCH_LOOKBACK_DAYS", "90"))))
RESEARCH_REFRESH_HOURS = max(6, min(24, int(os.getenv("ALPHA_RESEARCH_REFRESH_HOURS", "12"))))
MAX_HOLD_BARS = {5: 72, 15: 48, 60: 24}

# Research Lab is deliberately report-only: it never changes live rules or sends orders.
# Cost profiles are stress scenarios, not claims about any specific venue.
LAB_COST_PROFILES = {
    "current_40fee_5slip": {"fee_bps": 40.0, "slippage_bps": 5.0, "funding_bps_8h": 1.0},
    "mid_20fee_4slip": {"fee_bps": 20.0, "slippage_bps": 4.0, "funding_bps_8h": 1.0},
    "lean_10fee_3slip": {"fee_bps": 10.0, "slippage_bps": 3.0, "funding_bps_8h": 0.5},
}
LAB_MIN_TRAIN_TRADES = max(12, int(os.getenv("ALPHA_LAB_MIN_TRAIN_TRADES", "20")))
LAB_TOP_FULL_SAMPLE = max(5, min(30, int(os.getenv("ALPHA_LAB_TOP_FULL_SAMPLE", "15"))))

PREREGISTRATION = {
    "version": "v1.6-regime-relative-strength-edge-explorer-paper",
    "mode": "paper_only",
    "data_source": "Kraken public REST live + Bitstamp public OHLC history with Kraken fallback; parameter lab is report-only",
    "markets_default": DEFAULT_MARKETS,
    "timeframes": [5, 15, 60],
    "design": {
        "candidate_ranking": True,
        "correlation_guard": True,
        "mandatory_stop": True,
        "breakeven_trailing": True,
        "stale_trade_exit": True,
        "deep_historical_research": True,
        "walk_forward_stability": True,
        "research_report_only": True,
        "parameter_search_lab": True,
        "cost_stress": True,
        "walk_forward_parameter_selection": True,
        "edge_explorer": True,
        "research_long_short": True,
        "regime_detection": True,
        "relative_strength": True,
        "cross_sectional_momentum": True,
        "multi_timeframe_research": True,
    },
    "risk": {
        "risk_per_trade_pct": 0.20,
        "max_daily_loss_pct": 1.50,
        "max_heat_pct": 1.25,
        "max_positions": 10,
        "max_gross_leverage": 2.0,
        "max_position_notional_pct": 35.0,
        "fee_bps_each_side": 40.0,
        "slippage_bps_each_side": 5.0,
        "funding_stress_bps_per_8h": 1.0,
        "min_net_rr": 1.25,
        "max_correlated_positions": 3,
        "correlation_threshold": 0.85,
    },
}
PREREGISTRATION_HASH = hashlib.sha256(
    json.dumps(PREREGISTRATION, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:16]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso() -> str:
    return utcnow().isoformat(timespec="seconds")


def ef(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def ei(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
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
    max_gross_leverage: float
    max_position_notional: float
    fee_rate: float
    slippage_rate: float
    funding_rate_8h: float
    min_notional: float
    min_net_rr: float
    min_net_target_bps: float
    corr_threshold: float
    max_correlated_positions: int

    @classmethod
    def from_env(cls):
        # V1.5 keeps the v1.3/v1.4 live-risk settings. Research Lab changes are report-only.
        # Old v1.2 ALPHA_MAX_POSITIONS=5 cannot silently keep the engine at five slots.
        return cls(
            db_path=os.getenv("ALPHA_DB_PATH", "/data/alpha_v1.db"),
            markets=tuple(
                x.strip().upper()
                for x in os.getenv("ALPHA_MARKETS", ",".join(DEFAULT_MARKETS)).split(",")
                if x.strip()
            ),
            timeframes=tuple(int(x) for x in os.getenv("ALPHA_TIMEFRAMES", "5,15,60").split(",") if x.strip()),
            scan_seconds=max(30, ei("ALPHA_SCAN_SECONDS", 60)),
            starting_cash=ef("ALPHA_STARTING_CASH", 10000.0),
            risk_per_trade=ef("ALPHA_RISK_PER_TRADE_PCT", 0.20) / 100.0,
            max_daily_loss=ef("ALPHA_MAX_DAILY_LOSS_PCT", 1.50) / 100.0,
            max_heat=ef("ALPHA_MAX_HEAT_PCT", 1.25) / 100.0,
            max_positions=max(0, ei("ALPHA_V13_MAX_POSITIONS", 10)),
            max_gross_leverage=max(1.0, min(2.0, ef("ALPHA_V13_MAX_GROSS_LEVERAGE", 2.0))),
            max_position_notional=ef("ALPHA_V13_MAX_POSITION_NOTIONAL_PCT", 35.0) / 100.0,
            fee_rate=ef("ALPHA_FEE_BPS", 40.0) / 10000.0,
            slippage_rate=ef("ALPHA_SLIPPAGE_BPS", 5.0) / 10000.0,
            funding_rate_8h=max(0.0, ef("ALPHA_FUNDING_BPS_8H", 1.0)) / 10000.0,
            min_notional=ef("ALPHA_MIN_NOTIONAL", 25.0),
            min_net_rr=max(0.0, ef("ALPHA_MIN_NET_RR", 1.25)),
            min_net_target_bps=max(0.0, ef("ALPHA_MIN_NET_TARGET_BPS", 10.0)),
            corr_threshold=min(0.99, max(0.0, ef("ALPHA_CORR_THRESHOLD", 0.85))),
            max_correlated_positions=max(1, ei("ALPHA_MAX_CORRELATED_POSITIONS", 3)),
        )


CFG = Config.from_env()
BOOT_ID = uuid.uuid4().hex[:12]


class DB:
    def __init__(self, path: str, starting_cash: float):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._migrate_schema()
        self._ensure_account(starting_cash)
        self._ensure_identity()

    def _init_schema(self):
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS account(
              id INTEGER PRIMARY KEY CHECK(id=1), cash REAL NOT NULL, starting_cash REAL NOT NULL,
              realized_pnl REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS positions(
              id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, timeframe INTEGER NOT NULL,
              strategy TEXT NOT NULL, entry_time TEXT NOT NULL, entry_price REAL NOT NULL, qty REAL NOT NULL,
              stop_price REAL NOT NULL, target_price REAL NOT NULL, initial_risk REAL NOT NULL,
              entry_fee REAL NOT NULL, last_price REAL NOT NULL, signal_bar_ts INTEGER NOT NULL,
              UNIQUE(symbol,timeframe,strategy));
            CREATE TABLE IF NOT EXISTS trades(
              id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, timeframe INTEGER NOT NULL,
              strategy TEXT NOT NULL, entry_time TEXT NOT NULL, exit_time TEXT NOT NULL,
              entry_price REAL NOT NULL, exit_price REAL NOT NULL, qty REAL NOT NULL,
              gross_pnl REAL NOT NULL, fees REAL NOT NULL, net_pnl REAL NOT NULL,
              r_multiple REAL, exit_reason TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS signals(
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, symbol TEXT NOT NULL,
              timeframe INTEGER NOT NULL, strategy TEXT NOT NULL, signal_bar_ts INTEGER NOT NULL,
              status TEXT NOT NULL, detail TEXT,
              UNIQUE(symbol,timeframe,strategy,signal_bar_ts));
            CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            """
        )
        self.conn.commit()

    def _columns(self, table: str) -> set[str]:
        return {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def _migrate_schema(self):
        pcols = self._columns("positions")
        if "initial_stop_price" not in pcols:
            self.conn.execute("ALTER TABLE positions ADD COLUMN initial_stop_price REAL")
            self.conn.execute("UPDATE positions SET initial_stop_price=stop_price WHERE initial_stop_price IS NULL")
        if "best_price" not in pcols:
            self.conn.execute("ALTER TABLE positions ADD COLUMN best_price REAL")
            self.conn.execute("UPDATE positions SET best_price=last_price WHERE best_price IS NULL")
        tcols = self._columns("trades")
        if "funding" not in tcols:
            self.conn.execute("ALTER TABLE trades ADD COLUMN funding REAL NOT NULL DEFAULT 0")
        self.conn.commit()

    def _ensure_account(self, starting_cash: float):
        if self.conn.execute("SELECT 1 FROM account WHERE id=1").fetchone() is None:
            self.conn.execute(
                "INSERT INTO account(id,cash,starting_cash,realized_pnl,updated_at) VALUES(1,?,?,0,?)",
                (starting_cash, starting_cash, iso()),
            )
            self.conn.commit()

    def _ensure_identity(self):
        if self.get("persistent_state_id") is None:
            self.set("persistent_state_id", uuid.uuid4().hex[:16])
            self.set("persistent_state_created_at", iso())

    def account(self):
        return dict(self.conn.execute("SELECT * FROM account WHERE id=1").fetchone())

    def positions(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM positions ORDER BY id").fetchall()]

    def trades(self, n=100):
        return [dict(r) for r in self.conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (n,)).fetchall()]

    def signals(self, n=200):
        return [dict(r) for r in self.conn.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (n,)).fetchall()]

    def count(self, table: str) -> int:
        if table not in {"positions", "trades", "signals"}:
            raise ValueError("bad table")
        return int(self.conn.execute(f"SELECT COUNT(*) n FROM {table}").fetchone()["n"])

    def set(self, key: str, value):
        self.conn.execute(
            "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return default

    def exists(self, symbol: str, timeframe: int, strategy: str, signal_bar_ts: int) -> bool:
        return bool(
            self.conn.execute(
                "SELECT 1 FROM signals WHERE symbol=? AND timeframe=? AND strategy=? AND signal_bar_ts=?",
                (symbol, timeframe, strategy, int(signal_bar_ts)),
            ).fetchone()
        )

    def signal(self, symbol, timeframe, strategy, bar, status, detail):
        try:
            self.conn.execute(
                "INSERT INTO signals(ts,symbol,timeframe,strategy,signal_bar_ts,status,detail) VALUES(?,?,?,?,?,?,?)",
                (iso(), symbol, timeframe, strategy, int(bar), status, detail),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass

    def open(self, symbol, timeframe, strategy, entry, qty, stop, target, risk, fee, bar):
        # V1.3 permits negative cash as simulated borrowed capital. Gross exposure is capped separately at 2x.
        with self.conn:
            a = self.account()
            cash_after = a["cash"] - entry * qty - fee
            self.conn.execute(
                """INSERT INTO positions(symbol,timeframe,strategy,entry_time,entry_price,qty,stop_price,target_price,
                   initial_risk,entry_fee,last_price,signal_bar_ts,initial_stop_price,best_price)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (symbol, timeframe, strategy, iso(), entry, qty, stop, target, risk, fee, entry, int(bar), stop, entry),
            )
            self.conn.execute("UPDATE account SET cash=?,updated_at=? WHERE id=1", (cash_after, iso()))

    def mark(self, pos_id: int, price: float):
        with self.conn:
            self.conn.execute(
                "UPDATE positions SET last_price=?,best_price=MAX(COALESCE(best_price,?),?) WHERE id=?",
                (price, price, price, pos_id),
            )

    def raise_stop(self, pos_id: int, new_stop: float):
        with self.conn:
            self.conn.execute("UPDATE positions SET stop_price=MAX(stop_price,?) WHERE id=?", (new_stop, pos_id))

    def close(self, pos_id: int, exit_price: float, reason: str, fee_rate: float, funding_rate_8h: float):
        with self.conn:
            row = self.conn.execute("SELECT * FROM positions WHERE id=?", (pos_id,)).fetchone()
            if row is None:
                return
            p = dict(row)
            exit_fee = exit_price * p["qty"] * fee_rate
            try:
                held_hours = max(0.0, (utcnow() - datetime.fromisoformat(p["entry_time"])).total_seconds() / 3600.0)
            except Exception:
                held_hours = 0.0
            funding = abs(p["entry_price"] * p["qty"]) * funding_rate_8h * (held_hours / 8.0)
            proceeds = exit_price * p["qty"] - exit_fee - funding
            gross = (exit_price - p["entry_price"]) * p["qty"]
            fees = p["entry_fee"] + exit_fee
            net = gross - fees - funding
            r_mult = net / p["initial_risk"] if p["initial_risk"] > 0 else None
            a = self.account()
            self.conn.execute(
                "UPDATE account SET cash=?,realized_pnl=?,updated_at=? WHERE id=1",
                (a["cash"] + proceeds, a["realized_pnl"] + net, iso()),
            )
            self.conn.execute(
                """INSERT INTO trades(symbol,timeframe,strategy,entry_time,exit_time,entry_price,exit_price,qty,
                   gross_pnl,fees,net_pnl,r_multiple,exit_reason,funding) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (p["symbol"], p["timeframe"], p["strategy"], p["entry_time"], iso(), p["entry_price"], exit_price,
                 p["qty"], gross, fees, net, r_mult, reason, funding),
            )
            self.conn.execute("DELETE FROM positions WHERE id=?", (pos_id,))


DBX = DB(CFG.db_path, CFG.starting_cash)


class Kraken:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "Alpha-v1.4-paper-research/1.0"})

    def get(self, path: str, params: dict):
        r = self.s.get(f"{KRAKEN_BASE}/{path}", params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            raise RuntimeError("; ".join(data["error"]))
        return data["result"]

    def ohlc(self, symbol: str, timeframe: int) -> pd.DataFrame:
        data = self.get("OHLC", {"pair": symbol, "interval": timeframe})
        key = next(k for k in data if k != "last")
        x = pd.DataFrame(data[key], columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
        if len(x) < 220:
            raise RuntimeError(f"not enough bars: {len(x)}")
        for c in ["open", "high", "low", "close", "vwap", "volume"]:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        x["time"] = pd.to_numeric(x["time"], errors="coerce").astype("int64")
        x = x.dropna().sort_values("time").reset_index(drop=True)
        return x.iloc[:-1].copy() if len(x) > 1 else x

    def price(self, symbol: str) -> float:
        data = self.get("Ticker", {"pair": symbol})
        return float(data[next(iter(data))]["c"][0])


K = Kraken()


class BitstampResearch:
    """Public Bitstamp OHLC history used only for research/backtesting. No order route exists."""
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "Alpha-v1.5-research-lab/1.0"})

    @staticmethod
    def symbol(kraken_symbol: str) -> str:
        if kraken_symbol == "XBTUSD":
            return "btcusd"
        if kraken_symbol.endswith("USD"):
            return kraken_symbol.lower()
        raise ValueError(f"no Bitstamp mapping for {kraken_symbol}")

    def ohlc(self, kraken_symbol: str, timeframe: int, days: int) -> pd.DataFrame:
        pair = self.symbol(kraken_symbol)
        step = int(timeframe) * 60
        if step not in {300, 900, 3600}:
            raise ValueError(f"unsupported Bitstamp timeframe: {timeframe}m")

        now_s = int(time.time())
        end_s = now_s - step  # completed candles only
        start_s = end_s - int(days) * 86_400
        cursor = start_s
        rows = []
        requests_used = 0
        expected = max(1, int((end_s - start_s) / step))
        max_requests = max(2, math.ceil(expected / 1000) + 4)

        while cursor < end_s and requests_used < max_requests:
            r = self.s.get(
                f"{BITSTAMP_BASE}/ohlc/{pair}/",
                params={
                    "step": step,
                    "limit": 1000,
                    "start": cursor,
                    "exclude_current_candle": "true",
                },
                timeout=18,
            )
            r.raise_for_status()
            payload = r.json()
            data = ((payload or {}).get("data") or {}).get("ohlc")
            if not isinstance(data, list):
                raise RuntimeError(f"unexpected Bitstamp response: {str(payload)[:180]}")
            if not data:
                break

            rows.extend(data)
            requests_used += 1
            last_ts = max(int(z.get("timestamp", 0)) for z in data)
            nxt = last_ts + step
            if nxt <= cursor:
                break
            cursor = nxt
            if last_ts >= end_s - step:
                break
            time.sleep(0.035)

        if not rows:
            raise RuntimeError("no Bitstamp candles returned")
        x = pd.DataFrame(rows)
        need = {"timestamp", "open", "high", "low", "close", "volume"}
        if not need.issubset(set(x.columns)):
            raise RuntimeError(f"Bitstamp OHLC fields missing: {sorted(need - set(x.columns))}")
        for c in ["open", "high", "low", "close", "volume"]:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        x["time"] = pd.to_numeric(x["timestamp"], errors="coerce")
        x = x[["time", "open", "high", "low", "close", "volume"]]
        x = x.dropna().drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        x = x[(x["time"] >= start_s) & (x["time"] <= end_s)]

        # Do not label a shallow/partial response as deep coverage.
        minimum = max(220, int(expected * 0.70))
        if len(x) < minimum:
            raise RuntimeError(f"partial Bitstamp history: {len(x)}/{expected} bars")
        return x


BR = BitstampResearch()


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c, h, l = x["close"], x["high"], x["low"]
    for n in [9, 20, 50, 200]:
        x[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    delta = c.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = ag / al.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((al == 0) & (ag > 0)), 100.0)
    rsi = rsi.where(~((al == 0) & (ag == 0)), 50.0)
    x["rsi14"] = rsi
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    x["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    m = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    x["bb_lower"] = m - 2 * sd
    x["bb_upper"] = m + 2 * sd
    x["prior_high10"] = h.shift(1).rolling(10).max()
    x["prior_high20"] = h.shift(1).rolling(20).max()
    x["prior_high40"] = h.shift(1).rolling(40).max()
    x["prior_low10"] = l.shift(1).rolling(10).min()
    x["prior_low20"] = l.shift(1).rolling(20).min()
    x["prior_low40"] = l.shift(1).rolling(40).min()
    x["prior_volmed20"] = x["volume"].shift(1).rolling(20).median()
    return x


def signals_at(x: pd.DataFrame, i: int):
    if i < 220 or i >= len(x):
        return []
    r, p = x.iloc[i], x.iloc[i - 1]
    need = ["close", "atr14", "ema9", "ema20", "ema50", "ema200", "rsi14", "prior_volmed20"]
    if not all(np.isfinite(r.get(k, np.nan)) for k in need):
        return []
    out = []
    trend = r["ema20"] > r["ema50"]
    strong = r["ema20"] > r["ema50"] > r["ema200"]
    liquid = r["volume"] >= 0.80 * r["prior_volmed20"]
    if trend and np.isfinite(r["prior_high20"]) and r["close"] > r["prior_high20"] and r["volume"] > r["prior_volmed20"]:
        out.append(("fast_breakout", float(r["atr14"]), 1.40, 3.00))
    if trend and np.isfinite(r["prior_high10"]) and r["close"] > r["prior_high10"] and liquid:
        out.append(("donchian_10_breakout", float(r["atr14"]), 1.30, 2.80))
    if strong and p["rsi14"] <= 50 < r["rsi14"] and r["close"] > r["ema20"]:
        out.append(("rsi_reclaim_fast", float(r["atr14"]), 1.20, 2.60))
    if strong and p["close"] <= p["ema20"] and r["close"] > r["ema20"] and 45 <= r["rsi14"] <= 68:
        out.append(("trend_pullback", float(r["atr14"]), 1.20, 2.60))
    if trend and p["ema9"] <= p["ema20"] and r["ema9"] > r["ema20"] and 50 <= r["rsi14"] <= 72 and liquid:
        out.append(("ema9_momentum", float(r["atr14"]), 1.20, 2.50))
    if strong and np.isfinite(p["bb_lower"]) and np.isfinite(r["bb_lower"]) and p["close"] < p["bb_lower"] and r["close"] > r["bb_lower"] and r["rsi14"] < 60:
        out.append(("bollinger_reentry_fast", float(r["atr14"]), 1.10, 2.40))
    return out


def latest_signals(df: pd.DataFrame):
    x = indicators(df)
    return signals_at(x, len(x) - 1)


def marked_equity(prices=None):
    a = DBX.account()
    cash = float(a["cash"])
    gross = 0.0
    for p in DBX.positions():
        gross += (prices or {}).get(p["symbol"], p["last_price"]) * p["qty"]
    return cash + gross, cash, gross


def current_heat(eq: float) -> float:
    return 1.0 if eq <= 0 else sum(float(p["initial_risk"]) for p in DBX.positions()) / eq


def daily_realized() -> float:
    d = utcnow().date().isoformat()
    return float(
        DBX.conn.execute(
            "SELECT COALESCE(SUM(net_pnl),0) x FROM trades WHERE substr(exit_time,1,10)=?", (d,)
        ).fetchone()["x"]
    )


def performance_payload():
    rows = [dict(r) for r in DBX.conn.execute("SELECT * FROM trades ORDER BY id").fetchall()]
    if not rows:
        return {
            "closed_trades": 0, "win_rate_pct": 0.0, "net_pnl": 0.0, "fees_paid": 0.0,
            "funding_paid": 0.0, "profit_factor": None, "avg_r": None, "trades_today": 0, "by_strategy": []
        }
    net = [float(r["net_pnl"]) for r in rows]
    wins = [x for x in net if x > 0]
    losses = [x for x in net if x < 0]
    rs = [float(r["r_multiple"]) for r in rows if r["r_multiple"] is not None and np.isfinite(r["r_multiple"])]
    today = utcnow().date().isoformat()
    by = []
    for strategy in sorted({r["strategy"] for r in rows}):
        sr = [r for r in rows if r["strategy"] == strategy]
        sn = [float(r["net_pnl"]) for r in sr]
        sw = sum(1 for x in sn if x > 0)
        gp = sum(x for x in sn if x > 0)
        gl = abs(sum(x for x in sn if x < 0))
        rr = [float(r["r_multiple"]) for r in sr if r["r_multiple"] is not None and np.isfinite(r["r_multiple"])]
        by.append({
            "strategy": strategy,
            "trades": len(sr),
            "net_pnl": round(sum(sn), 2),
            "win_rate_pct": round(100 * sw / len(sr), 1),
            "profit_factor": round(gp / gl, 2) if gl > 0 else None,
            "avg_r": round(sum(rr) / len(rr), 3) if rr else None,
        })
    gp, gl = sum(wins), abs(sum(losses))
    return {
        "closed_trades": len(rows),
        "win_rate_pct": 100 * len(wins) / len(rows),
        "net_pnl": sum(net),
        "fees_paid": sum(float(r["fees"]) for r in rows),
        "funding_paid": sum(float(r.get("funding") or 0.0) for r in rows),
        "profit_factor": gp / gl if gl > 0 else None,
        "avg_r": sum(rs) / len(rs) if rs else None,
        "trades_today": sum(str(r["exit_time"]).startswith(today) for r in rows),
        "by_strategy": by,
    }


def cost_model(entry: float, stop: float, target: float):
    stop_exec = max(0.0, stop * (1 - CFG.slippage_rate))
    target_exec = max(0.0, target * (1 - CFG.slippage_rate))
    entry_fee = entry * CFG.fee_rate
    stop_fee = stop_exec * CFG.fee_rate
    target_fee = target_exec * CFG.fee_rate
    stop_loss = max(0.0, (entry - stop_exec) + entry_fee + stop_fee)
    target_profit = (target_exec - entry) - entry_fee - target_fee
    return {
        "loss": stop_loss,
        "win": target_profit,
        "rr": target_profit / stop_loss if stop_loss > 0 else -math.inf,
        "bps": (target_profit / entry) * 10000 if entry > 0 else -math.inf,
    }


def economic_precheck(entry: float, stop: float, target: float):
    econ = cost_model(entry, stop, target)
    if stop <= 0 or stop >= entry or target <= entry:
        return False, "invalid_levels", econ
    if econ["win"] <= 0:
        return False, "negative_net_target", econ
    if econ["bps"] < CFG.min_net_target_bps:
        return False, "net_target_too_small", econ
    if econ["rr"] < CFG.min_net_rr:
        return False, "net_rr_too_low", econ
    return True, "economic_pass", econ


def correlation_count(symbol: str, chosen_symbols: list[str], corr_series: dict[str, pd.Series]) -> int:
    a = corr_series.get(symbol)
    if a is None or len(a) < 30:
        return 0
    count = 0
    for other in chosen_symbols:
        b = corr_series.get(other)
        if b is None:
            continue
        pair = pd.concat([a, b], axis=1).dropna().tail(120)
        if len(pair) < 30:
            continue
        corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        if np.isfinite(corr) and corr >= CFG.corr_threshold:
            count += 1
    return count


def capacity_gate(symbol: str, entry: float, econ: dict, corr_series: dict[str, pd.Series]):
    ps = DBX.positions()
    if CFG.max_positions <= 0:
        return False, "entries_paused", None, None
    if len(ps) >= CFG.max_positions:
        return False, "max_positions", None, None
    if any(p["symbol"] == symbol for p in ps):
        return False, "symbol_already_open", None, None
    eq, _, gross = marked_equity()
    if eq <= 0:
        return False, "nonpositive_equity", None, None
    if daily_realized() <= -CFG.max_daily_loss * eq:
        return False, "daily_loss_kill", None, None
    chosen_symbols = [p["symbol"] for p in ps]
    if correlation_count(symbol, chosen_symbols, corr_series) >= CFG.max_correlated_positions:
        return False, "correlation_cluster", None, None
    per_unit_risk = econ["loss"]
    if per_unit_risk <= 0:
        return False, "invalid_net_risk", None, None
    max_gross = eq * CFG.max_gross_leverage
    room = max(0.0, max_gross - gross)
    qty = min(
        eq * CFG.risk_per_trade / per_unit_risk,
        eq * CFG.max_position_notional / entry,
        room / entry,
    )
    if qty <= 0 or qty * entry < CFG.min_notional:
        return False, "no_capacity", None, None
    actual_risk = qty * per_unit_risk
    if current_heat(eq) + actual_risk / eq > CFG.max_heat + 1e-12:
        return False, "portfolio_heat", None, None
    return True, "accepted", qty, actual_risk


async def manage_positions(price_cache: dict[str, float]):
    for p in DBX.positions():
        try:
            px = price_cache.get(p["symbol"])
            if px is None:
                px = await asyncio.to_thread(K.price, p["symbol"])
                price_cache[p["symbol"]] = px
            DBX.mark(p["id"], px)

            # Mandatory hard stop / target first.
            if px <= p["stop_price"]:
                DBX.close(p["id"], px * (1 - CFG.slippage_rate), "STOP", CFG.fee_rate, CFG.funding_rate_8h)
                continue
            if px >= p["target_price"]:
                DBX.close(p["id"], px * (1 - CFG.slippage_rate), "TARGET", CFG.fee_rate, CFG.funding_rate_8h)
                continue

            # Profit protection: once price reaches +1R (price-risk basis), move stop above fee break-even.
            initial_stop = float(p.get("initial_stop_price") or p["stop_price"])
            price_risk = max(1e-12, p["entry_price"] - initial_stop)
            progress_r = (px - p["entry_price"]) / price_risk
            if progress_r >= 1.0:
                fee_buffer = p["entry_price"] * (2 * CFG.fee_rate + CFG.slippage_rate)
                DBX.raise_stop(p["id"], p["entry_price"] + fee_buffer)
            if progress_r >= 1.5:
                DBX.raise_stop(p["id"], p["entry_price"] + 0.50 * price_risk)

            # Stale trades do not occupy capital indefinitely. Time exit only if they have not developed.
            try:
                held_min = max(0.0, (utcnow() - datetime.fromisoformat(p["entry_time"])).total_seconds() / 60.0)
            except Exception:
                held_min = 0.0
            max_bars = MAX_HOLD_BARS.get(int(p["timeframe"]), 24)
            if held_min >= max_bars * int(p["timeframe"]) and progress_r < 0.25:
                DBX.close(p["id"], px * (1 - CFG.slippage_rate), "TIME_STALE", CFG.fee_rate, CFG.funding_rate_8h)
        except Exception as e:
            DBX.set("last_position_error", {"at": iso(), "position_id": p.get("id"), "error": repr(e)})


async def scan_once():
    started = time.time()
    price_cache: dict[str, float] = {}
    errors = []
    scanned = raw = new = duplicates = opened = rejected = 0
    await manage_positions(price_cache)

    candidates = []
    corr_series: dict[str, pd.Series] = {}

    # Phase 1: scan the whole universe and economically qualify every new signal BEFORE slot limits.
    for symbol in CFG.markets:
        for tf in CFG.timeframes:
            scanned += 1
            try:
                df = await asyncio.to_thread(K.ohlc, symbol, tf)
                if tf == 60:
                    s = df.set_index("time")["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                    corr_series[symbol] = s
                bar = int(df.iloc[-1]["time"])
                for strategy, atr, stop_mult, target_mult in latest_signals(df):
                    raw += 1
                    if DBX.exists(symbol, tf, strategy, bar):
                        duplicates += 1
                        continue
                    new += 1
                    px = price_cache.get(symbol)
                    if px is None:
                        px = await asyncio.to_thread(K.price, symbol)
                        price_cache[symbol] = px
                    entry = px * (1 + CFG.slippage_rate)
                    stop = entry - stop_mult * atr
                    target = entry + target_mult * atr
                    ok, reason, econ = economic_precheck(entry, stop, target)
                    if not ok:
                        rejected += 1
                        DBX.signal(symbol, tf, strategy, bar, "REJECTED", json.dumps(
                            {"reason": reason, "net_rr": econ["rr"], "target_net_bps": econ["bps"]}, separators=(",", ":")
                        ))
                        continue
                    score = float(econ["rr"]) + min(max(float(econ["bps"]), 0.0), 600.0) / 600.0
                    # Mild preference for faster setups when quality is otherwise similar.
                    score += {5: 0.12, 15: 0.06, 60: 0.0}.get(tf, 0.0)
                    candidates.append({
                        "symbol": symbol, "tf": tf, "strategy": strategy, "bar": bar,
                        "entry": entry, "stop": stop, "target": target, "econ": econ, "score": score,
                    })
            except Exception as e:
                errors.append(f"{symbol}/{tf}m: {e}")

    # Phase 2: best opportunities first. Slot/risk/correlation limits are applied only now.
    candidates.sort(key=lambda c: c["score"], reverse=True)
    for c in candidates:
        ok, reason, qty, risk = capacity_gate(c["symbol"], c["entry"], c["econ"], corr_series)
        if not ok:
            rejected += 1
            DBX.signal(c["symbol"], c["tf"], c["strategy"], c["bar"], "REJECTED", json.dumps(
                {"reason": reason, "score": c["score"], "net_rr": c["econ"]["rr"], "target_net_bps": c["econ"]["bps"]},
                separators=(",", ":")
            ))
            continue
        try:
            fee = c["entry"] * qty * CFG.fee_rate
            DBX.open(c["symbol"], c["tf"], c["strategy"], c["entry"], qty, c["stop"], c["target"], risk, fee, c["bar"])
            opened += 1
            DBX.signal(c["symbol"], c["tf"], c["strategy"], c["bar"], "OPENED", json.dumps(
                {"entry": c["entry"], "qty": qty, "stop": c["stop"], "target": c["target"],
                 "score": c["score"], "net_rr": c["econ"]["rr"], "target_net_bps": c["econ"]["bps"]},
                separators=(",", ":")
            ))
        except Exception as e:
            rejected += 1
            DBX.signal(c["symbol"], c["tf"], c["strategy"], c["bar"], "REJECTED", f"open_error:{e}")

    eq, cash, gross = marked_equity(price_cache)
    snap = {
        "state": "idle",
        "finished": iso(),
        "boot_id": BOOT_ID,
        "seconds": round(time.time() - started, 2),
        "markets": len(CFG.markets),
        "scans": scanned,
        "setups_detected": raw,
        "new_setups": new,
        "duplicates": duplicates,
        "economically_qualified": len(candidates),
        "opened": opened,
        "rejected": rejected,
        "open_positions": len(DBX.positions()),
        "equity": eq,
        "gross_exposure": gross,
        "gross_leverage": gross / eq if eq > 0 else None,
        "portfolio_heat_pct": current_heat(eq) * 100,
        "errors": errors[-10:],
    }
    DBX.set("last_scan", snap)
    return snap


def historical_exit(x: pd.DataFrame, entry_i: int, entry: float, stop: float, target: float, tf: int):
    # Conservative intrabar assumptions: if stop and target are both touched, stop wins.
    stale_bars = MAX_HOLD_BARS.get(tf, 24)
    hard_end = min(len(x) - 1, entry_i + stale_bars * 2)
    initial_stop = stop
    price_risk = max(1e-12, entry - initial_stop)
    active_stop = stop
    for j in range(entry_i, hard_end + 1):
        bar = x.iloc[j]
        hit_stop = float(bar["low"]) <= active_stop
        hit_target = float(bar["high"]) >= target
        if hit_stop and hit_target:
            return j, active_stop * (1 - CFG.slippage_rate), "STOP_BOTH"
        if hit_stop:
            return j, active_stop * (1 - CFG.slippage_rate), "STOP"
        if hit_target:
            return j, target * (1 - CFG.slippage_rate), "TARGET"

        # Mirror live v1.3/v1.4 protection on the next bar after the threshold is observed.
        high_r = (float(bar["high"]) - entry) / price_risk
        if high_r >= 1.0:
            fee_buffer = entry * (2 * CFG.fee_rate + CFG.slippage_rate)
            active_stop = max(active_stop, entry + fee_buffer)
        if high_r >= 1.5:
            active_stop = max(active_stop, entry + 0.50 * price_risk)

        bars_held = j - entry_i + 1
        if bars_held >= stale_bars:
            close_r = (float(bar["close"]) - entry) / price_risk
            if close_r < 0.25:
                return j, float(bar["close"]) * (1 - CFG.slippage_rate), "TIME_STALE"

    return hard_end, float(x.iloc[hard_end]["close"]) * (1 - CFG.slippage_rate), "TIME_HARD"


def backtest_all_strategies(df: pd.DataFrame, symbol: str, tf: int):
    """Evaluate each strategy independently; one concurrent position per strategy/symbol/timeframe."""
    x = indicators(df).reset_index(drop=True)
    trades = []
    busy_until: dict[str, int] = {}
    for i in range(220, len(x) - 2):
        sigs = signals_at(x, i)
        if not sigs:
            continue
        entry_i = i + 1
        raw_entry = float(x.iloc[entry_i]["open"])
        if not np.isfinite(raw_entry) or raw_entry <= 0:
            continue
        for strategy, atr, sm, tm in sigs:
            if i < busy_until.get(strategy, -1):
                continue
            entry = raw_entry * (1 + CFG.slippage_rate)
            stop = entry - sm * atr
            target = entry + tm * atr
            ok, _, econ = economic_precheck(entry, stop, target)
            if not ok:
                continue
            exit_i, exit_px, reason = historical_exit(x, entry_i, entry, stop, target, tf)
            held_hours = max(0.0, (exit_i - entry_i + 1) * tf / 60.0)
            funding_per_unit = entry * CFG.funding_rate_8h * (held_hours / 8.0)
            net_per_unit = (exit_px - entry) - entry * CFG.fee_rate - exit_px * CFG.fee_rate - funding_per_unit
            r_mult = net_per_unit / econ["loss"] if econ["loss"] > 0 else None
            trades.append({
                "symbol": symbol,
                "timeframe": tf,
                "strategy": strategy,
                "entry_ts": int(x.iloc[entry_i]["time"]),
                "exit_ts": int(x.iloc[exit_i]["time"]),
                "r": r_mult,
                "reason": reason,
            })
            busy_until[strategy] = exit_i + 1
    return trades


def summarize_research(rows: list[dict]):
    rows = [r for r in rows if r.get("r") is not None and np.isfinite(r["r"])]
    if not rows:
        return {"trades": 0, "avg_r": None, "win_rate_pct": 0.0, "profit_factor_r": None, "total_r": 0.0, "max_drawdown_r": None}
    rows = sorted(rows, key=lambda z: (z.get("entry_ts", 0), z.get("symbol", ""), z.get("strategy", "")))
    rs = [float(r["r"]) for r in rows]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    gp, gl = sum(wins), abs(sum(losses))
    curve = np.cumsum(rs)
    peaks = np.maximum.accumulate(np.insert(curve, 0, 0.0))[1:]
    dd = curve - peaks
    return {
        "trades": len(rs),
        "avg_r": round(float(np.mean(rs)), 4),
        "win_rate_pct": round(100 * len(wins) / len(rs), 1),
        "profit_factor_r": round(gp / gl, 3) if gl > 0 else None,
        "total_r": round(float(sum(rs)), 2),
        "max_drawdown_r": round(float(abs(min(dd))) if len(dd) else 0.0, 2),
    }


def grouped_research(rows: list[dict], fields: tuple[str, ...], min_trades: int = 1):
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = tuple(r[f] for f in fields)
        groups.setdefault(key, []).append(r)
    out = []
    for key, rr in groups.items():
        sm = summarize_research(rr)
        if sm["trades"] < min_trades:
            continue
        row = {f: key[i] for i, f in enumerate(fields)}
        row.update(sm)
        out.append(row)
    out.sort(key=lambda z: (z.get("avg_r") if z.get("avg_r") is not None else -999, z.get("trades", 0)), reverse=True)
    return out


def walk_forward_report(rows: list[dict]):
    valid = [r for r in rows if r.get("r") is not None and np.isfinite(r["r"])]
    if len(valid) < 80:
        return {"state": "insufficient_sample", "folds": [], "test_trades": 0, "avg_r": None, "total_r": 0.0, "profit_factor_r": None, "profitable_folds": 0}
    lo = min(r["entry_ts"] for r in valid)
    hi = max(r["entry_ts"] for r in valid)
    span = max(1, hi - lo)
    folds = []
    selected_all = []
    # Expanding train; three sequential 20% out-of-sample windows after the first 40%.
    for fold_idx, (a, b) in enumerate(((0.40, 0.60), (0.60, 0.80), (0.80, 1.001)), start=1):
        test_start = lo + int(span * a)
        test_end = lo + int(span * b)
        train = [r for r in valid if r["entry_ts"] < test_start]
        test = [r for r in valid if test_start <= r["entry_ts"] < test_end]
        train_groups = grouped_research(train, ("strategy", "timeframe"), min_trades=20)
        eligible = {
            (g["strategy"], g["timeframe"])
            for g in train_groups
            if (g.get("avg_r") or -999) > 0 and (g.get("profit_factor_r") or 0) > 1.05
        }
        selected = [r for r in test if (r["strategy"], r["timeframe"]) in eligible]
        selected_all.extend(selected)
        sm = summarize_research(selected)
        folds.append({
            "fold": fold_idx,
            "train_trades": len(train),
            "eligible_setups": len(eligible),
            "test_trades": sm["trades"],
            "avg_r": sm["avg_r"],
            "profit_factor_r": sm["profit_factor_r"],
            "total_r": sm["total_r"],
        })
    total = summarize_research(selected_all)
    profitable = sum(1 for f in folds if (f.get("total_r") or 0) > 0 and f.get("test_trades", 0) > 0)
    return {
        "state": "done",
        "folds": folds,
        "test_trades": total["trades"],
        "avg_r": total["avg_r"],
        "profit_factor_r": total["profit_factor_r"],
        "total_r": total["total_r"],
        "max_drawdown_r": total["max_drawdown_r"],
        "profitable_folds": profitable,
    }



def lab_cost_model(entry_raw: float, atr: float, stop_mult: float, target_mult: float, cost: dict):
    """Return executable entry/levels and per-unit net economics for one stress-cost scenario."""
    fee_rate = float(cost["fee_bps"]) / 10000.0
    slip_rate = float(cost["slippage_bps"]) / 10000.0
    entry = entry_raw * (1.0 + slip_rate)
    stop = entry - float(stop_mult) * float(atr)
    target = entry + float(target_mult) * float(atr)
    if entry <= 0 or stop <= 0 or stop >= entry or target <= entry:
        return None
    stop_exec = max(0.0, stop * (1.0 - slip_rate))
    target_exec = max(0.0, target * (1.0 - slip_rate))
    loss = max(0.0, (entry - stop_exec) + entry * fee_rate + stop_exec * fee_rate)
    win = (target_exec - entry) - entry * fee_rate - target_exec * fee_rate
    rr = win / loss if loss > 0 else -math.inf
    return {
        "entry": entry, "stop": stop, "target": target,
        "fee_rate": fee_rate, "slip_rate": slip_rate,
        "funding_rate_8h": float(cost["funding_bps_8h"]) / 10000.0,
        "risk_per_unit": loss, "net_rr": rr,
        "target_net_bps": (win / entry) * 10000.0 if entry > 0 else -math.inf,
    }


def historical_exit_lab(x: pd.DataFrame, entry_i: int, entry: float, stop: float, target: float, tf: int, slip_rate: float, fee_rate: float):
    """Same conservative stop/target ordering as live research, but with scenario-specific slippage."""
    stale_bars = MAX_HOLD_BARS.get(tf, 24)
    hard_end = min(len(x) - 1, entry_i + stale_bars * 2)
    initial_stop = stop
    price_risk = max(1e-12, entry - initial_stop)
    active_stop = stop
    for j in range(entry_i, hard_end + 1):
        bar = x.iloc[j]
        hit_stop = float(bar["low"]) <= active_stop
        hit_target = float(bar["high"]) >= target
        if hit_stop and hit_target:
            return j, active_stop * (1 - slip_rate), "STOP_BOTH"
        if hit_stop:
            return j, active_stop * (1 - slip_rate), "STOP"
        if hit_target:
            return j, target * (1 - slip_rate), "TARGET"

        high_r = (float(bar["high"]) - entry) / price_risk
        if high_r >= 1.0:
            fee_buffer = entry * (2 * fee_rate + slip_rate)
            active_stop = max(active_stop, entry + fee_buffer)
        if high_r >= 1.5:
            active_stop = max(active_stop, entry + 0.50 * price_risk)

        bars_held = j - entry_i + 1
        if bars_held >= stale_bars:
            close_r = (float(bar["close"]) - entry) / price_risk
            if close_r < 0.25:
                return j, float(bar["close"]) * (1 - slip_rate), "TIME_STALE"
    return hard_end, float(x.iloc[hard_end]["close"]) * (1 - slip_rate), "TIME_HARD"


def lab_signal_specs(x: pd.DataFrame, i: int):
    """Generate a compact preregistered parameter grid without touching live strategy settings."""
    if i < 220 or i >= len(x) - 2:
        return []
    r, p = x.iloc[i], x.iloc[i - 1]
    need = ["close", "atr14", "ema9", "ema20", "ema50", "ema200", "rsi14", "prior_volmed20"]
    if not all(np.isfinite(r.get(k, np.nan)) for k in need):
        return []
    atr = float(r["atr14"])
    if not np.isfinite(atr) or atr <= 0:
        return []

    out = []
    trend = r["ema20"] > r["ema50"]
    strong = r["ema20"] > r["ema50"] > r["ema200"]
    volmed = float(r["prior_volmed20"])
    vol_ratio = float(r["volume"]) / volmed if volmed > 0 else 0.0

    # Breakout family: lookback, trend strictness, volume filter, stop and target ATR multiples.
    for lookback in (10, 20, 40):
        ph = r.get(f"prior_high{lookback}", np.nan)
        if not np.isfinite(ph) or float(r["close"]) <= float(ph):
            continue
        for strong_only in (False, True):
            if strong_only and not strong:
                continue
            if not strong_only and not trend:
                continue
            for volume_mult in (0.8, 1.0):
                if vol_ratio < volume_mult:
                    continue
                for sm in (1.2, 1.6):
                    for tm in (2.4, 3.2):
                        vid = f"breakout_L{lookback}_{'strong' if strong_only else 'trend'}_V{volume_mult:.1f}_S{sm:.1f}_T{tm:.1f}"
                        out.append((vid, "breakout", atr, sm, tm, {
                            "lookback": lookback, "trend": "strong" if strong_only else "trend",
                            "volume_mult": volume_mult, "stop_atr": sm, "target_atr": tm,
                        }))

    # RSI reclaim family: threshold, strict trend, stop/target.
    if strong and float(r["close"]) > float(r["ema20"]):
        for reclaim in (45, 50, 55):
            if float(p["rsi14"]) <= reclaim < float(r["rsi14"]):
                for sm in (1.0, 1.3):
                    for tm in (2.2, 3.0):
                        vid = f"rsi_reclaim_R{reclaim}_S{sm:.1f}_T{tm:.1f}"
                        out.append((vid, "rsi_reclaim", atr, sm, tm, {
                            "reclaim": reclaim, "stop_atr": sm, "target_atr": tm,
                        }))

    # Trend pullback family: reclaim EMA20 with two RSI ceilings and two stop/target pairs.
    if strong and float(p["close"]) <= float(p["ema20"]) and float(r["close"]) > float(r["ema20"]):
        for rsi_max in (60, 68):
            if 45 <= float(r["rsi14"]) <= rsi_max:
                for sm in (1.0, 1.3):
                    for tm in (2.2, 3.0):
                        vid = f"trend_pullback_R{rsi_max}_S{sm:.1f}_T{tm:.1f}"
                        out.append((vid, "trend_pullback", atr, sm, tm, {
                            "rsi_max": rsi_max, "stop_atr": sm, "target_atr": tm,
                        }))
    return out


def backtest_lab_frame(df: pd.DataFrame, symbol: str, tf: int):
    x = indicators(df).reset_index(drop=True)
    trades = []
    busy_until: dict[tuple[str, str], int] = {}
    for i in range(220, len(x) - 2):
        specs = lab_signal_specs(x, i)
        if not specs:
            continue
        entry_i = i + 1
        raw_entry = float(x.iloc[entry_i]["open"])
        if not np.isfinite(raw_entry) or raw_entry <= 0:
            continue
        for variant_id, family, atr, sm, tm, params in specs:
            for cost_name, cost in LAB_COST_PROFILES.items():
                key = (variant_id, cost_name)
                if i < busy_until.get(key, -1):
                    continue
                econ = lab_cost_model(raw_entry, atr, sm, tm, cost)
                if (not econ or econ["risk_per_unit"] <= 0 or econ["net_rr"] < CFG.min_net_rr
                        or econ["target_net_bps"] < CFG.min_net_target_bps):
                    continue
                exit_i, exit_px, reason = historical_exit_lab(
                    x, entry_i, econ["entry"], econ["stop"], econ["target"], tf,
                    econ["slip_rate"], econ["fee_rate"]
                )
                held_hours = max(0.0, (exit_i - entry_i + 1) * tf / 60.0)
                funding = econ["entry"] * econ["funding_rate_8h"] * (held_hours / 8.0)
                exit_fee = exit_px * econ["fee_rate"]
                entry_fee = econ["entry"] * econ["fee_rate"]
                net = (exit_px - econ["entry"]) - entry_fee - exit_fee - funding
                r_mult = net / econ["risk_per_unit"] if econ["risk_per_unit"] > 0 else None
                trades.append({
                    "symbol": symbol, "timeframe": tf, "family": family,
                    "variant_id": variant_id, "cost_profile": cost_name,
                    "entry_ts": int(x.iloc[entry_i]["time"]), "exit_ts": int(x.iloc[exit_i]["time"]),
                    "r": r_mult, "reason": reason, "params": params,
                })
                busy_until[key] = exit_i + 1
    return trades


def lab_candidate_key(row: dict):
    return (row["variant_id"], int(row["timeframe"]), row["cost_profile"])


def summarize_lab_groups(rows: list[dict], min_trades: int = 1):
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("r") is None or not np.isfinite(r["r"]):
            continue
        groups.setdefault(lab_candidate_key(r), []).append(r)
    out = []
    for (variant_id, tf, cost_profile), rr in groups.items():
        sm = summarize_research(rr)
        if sm["trades"] < min_trades:
            continue
        first = rr[0]
        # Penalize tiny samples and deep drawdowns; this is only a research ranking.
        robust_score = (sm.get("avg_r") or -999) * math.sqrt(sm["trades"]) - 0.015 * (sm.get("max_drawdown_r") or 0)
        out.append({
            "variant_id": variant_id, "family": first["family"], "timeframe": tf,
            "cost_profile": cost_profile, "params": first["params"],
            **sm, "robust_score": round(float(robust_score), 4),
        })
    out.sort(key=lambda z: (z["robust_score"], z.get("avg_r") or -999, z["trades"]), reverse=True)
    return out


def lab_walk_forward(rows: list[dict]):
    valid = [r for r in rows if r.get("r") is not None and np.isfinite(r["r"])]
    if len(valid) < 100:
        return {"state": "insufficient_sample", "folds": [], "test_trades": 0, "avg_r": None,
                "profit_factor_r": None, "total_r": 0.0, "max_drawdown_r": None, "profitable_folds": 0}
    lo = min(r["entry_ts"] for r in valid)
    hi = max(r["entry_ts"] for r in valid)
    span = max(1, hi - lo)
    folds = []
    selected_oos = []
    for fold_idx, (a, b) in enumerate(((0.40, 0.60), (0.60, 0.80), (0.80, 1.001)), start=1):
        test_start = lo + int(span * a)
        test_end = lo + int(span * b)
        train = [r for r in valid if r["entry_ts"] < test_start]
        test = [r for r in valid if test_start <= r["entry_ts"] < test_end]
        ranked = summarize_lab_groups(train, min_trades=LAB_MIN_TRAIN_TRADES)
        ranked = [g for g in ranked if g["cost_profile"] == "current_40fee_5slip"]
        eligible = [g for g in ranked if (g.get("avg_r") or -999) > 0 and (g.get("profit_factor_r") or 0) > 1.05]
        chosen = eligible[0] if eligible else None
        if chosen:
            key = (chosen["variant_id"], int(chosen["timeframe"]), chosen["cost_profile"])
            selected = [r for r in test if lab_candidate_key(r) == key]
        else:
            selected = []
        selected_oos.extend(selected)
        sm = summarize_research(selected)
        folds.append({
            "fold": fold_idx,
            "train_trade_rows": len(train),
            "eligible_candidates": len(eligible),
            "selected": (f"{chosen['variant_id']} | {chosen['timeframe']}m | {chosen['cost_profile']}" if chosen else None),
            "selected_train_trades": chosen.get("trades", 0) if chosen else 0,
            "selected_train_avg_r": chosen.get("avg_r") if chosen else None,
            "test_trades": sm["trades"], "avg_r": sm["avg_r"],
            "profit_factor_r": sm["profit_factor_r"], "total_r": sm["total_r"],
        })
    total = summarize_research(selected_oos)
    profitable = sum(1 for f in folds if f.get("test_trades", 0) > 0 and (f.get("total_r") or 0) > 0)
    return {
        "state": "done", "folds": folds, "test_trades": total["trades"], "avg_r": total["avg_r"],
        "profit_factor_r": total["profit_factor_r"], "total_r": total["total_r"],
        "max_drawdown_r": total["max_drawdown_r"], "profitable_folds": profitable,
    }


def parameter_lab(frames: dict[tuple[str, int], pd.DataFrame]):
    started = time.time()
    DBX.set("lab", {
        "state": "running", "started": iso(), "completed_units": 0, "total_units": len(frames),
        "trade_rows_so_far": 0, "note": "Report-only parameter search; live settings are unchanged.",
    })
    rows: list[dict] = []
    errors = []
    for idx, ((symbol, tf), df) in enumerate(frames.items(), start=1):
        try:
            rows.extend(backtest_lab_frame(df, symbol, tf))
        except Exception as e:
            errors.append(f"{symbol}/{tf}m: {e}")
        if idx == 1 or idx % 3 == 0 or idx == len(frames):
            DBX.set("lab", {
                "state": "running", "started": iso(), "completed_units": idx, "total_units": len(frames),
                "current": f"{symbol}/{tf}m", "trade_rows_so_far": len(rows),
                "seconds": round(time.time() - started, 1), "errors": errors[-6:],
                "note": "Report-only parameter search; live settings are unchanged.",
            })

    ranked = summarize_lab_groups(rows, min_trades=20)
    top = ranked[:LAB_TOP_FULL_SAMPLE]
    best_by_cost = []
    for cost_name in LAB_COST_PROFILES:
        eligible = [g for g in ranked if g["cost_profile"] == cost_name]
        if eligible:
            best_by_cost.append(eligible[0])
    wfo = lab_walk_forward(rows)

    # A strict research gate: this does NOT enable live trading or modify live parameters.
    current_ranked = [g for g in ranked if g["cost_profile"] == "current_40fee_5slip"]
    best = current_ranked[0] if current_ranked else None
    checks = {
        "candidate_sample_at_least_40": bool(best and best.get("trades", 0) >= 40),
        "candidate_avg_r_positive": bool(best and (best.get("avg_r") or -999) > 0),
        "candidate_pf_gt_1_10": bool(best and (best.get("profit_factor_r") or 0) > 1.10),
        "wfo_test_at_least_30": wfo.get("test_trades", 0) >= 30,
        "wfo_avg_r_positive": (wfo.get("avg_r") or -999) > 0,
        "wfo_profitable_folds_at_least_2": wfo.get("profitable_folds", 0) >= 2,
    }
    gate = "RESEARCH_CANDIDATE" if all(checks.values()) else "NO_ROBUST_EDGE_YET"

    # Count distinct parameter candidates (variant x timeframe x cost), not duplicated trade rows.
    candidate_count = len({lab_candidate_key(r) for r in rows})
    result = {
        "state": "done", "finished": iso(), "seconds": round(time.time() - started, 2),
        "scope": f"{RESEARCH_LOOKBACK_DAYS}d cached deep history · parameter grid · 3 cost stresses · 3-fold expanding walk-forward · report-only",
        "frames": len(frames), "trade_rows": len(rows), "candidate_count": candidate_count,
        "cost_profiles": LAB_COST_PROFILES, "top_candidates": top,
        "current_cost_best": best, "best_by_cost": best_by_cost, "walk_forward": wfo,
        "evidence_gate": gate, "evidence_checks": checks,
        "multiple_testing_note": "Many parameter variants are tested. Full-sample winners are hypothesis generators; the evidence gate and walk-forward selection use the current conservative 40-bps fee + 5-bps slippage scenario.",
        "errors": errors[-12:],
    }
    DBX.set("lab", result)
    return result


# -------------------- v1.6 Edge Explorer (research-only) --------------------
# Purpose: test genuinely different sources of edge rather than re-tuning the
# same entry rule. Nothing in this section changes the live paper engine.

EDGE_CURRENT_COST = "current_40fee_5slip"
EDGE_MIN_TRAIN_TRADES = max(18, int(os.getenv("ALPHA_EDGE_MIN_TRAIN_TRADES", "25")))


def edge_regime_frame(btc60: pd.DataFrame) -> pd.DataFrame:
    x = indicators(btc60).reset_index(drop=True)
    x["ret24"] = x["close"].pct_change(24)
    x["trend_strength"] = (x["ema20"] - x["ema50"]).abs() / x["atr14"].replace(0, np.nan)
    reg = np.full(len(x), "mixed", dtype=object)
    bull = (x["ema20"] > x["ema50"]) & (x["ema50"] > x["ema200"]) & (x["ret24"] > 0)
    bear = (x["ema20"] < x["ema50"]) & (x["ema50"] < x["ema200"]) & (x["ret24"] < 0)
    side = (~bull) & (~bear) & (x["trend_strength"] < 1.25)
    reg[bull.fillna(False).to_numpy()] = "bull"
    reg[bear.fillna(False).to_numpy()] = "bear"
    reg[side.fillna(False).to_numpy()] = "sideways"
    x["regime"] = reg
    return x


def _asof_index(times: np.ndarray, ts: int) -> int:
    return int(np.searchsorted(times, int(ts), side="right") - 1)


def edge_cost_econ(entry_raw: float, atr: float, stop_mult: float, target_mult: float,
                   direction: str, cost: dict):
    fee = float(cost["fee_bps"]) / 10000.0
    slip = float(cost["slippage_bps"]) / 10000.0
    fund = float(cost["funding_bps_8h"]) / 10000.0
    if direction == "long":
        entry = entry_raw * (1 + slip)
        stop = entry - stop_mult * atr
        target = entry + target_mult * atr
        stop_exec = stop * (1 - slip)
        target_exec = target * (1 - slip)
        risk = (entry - stop_exec) + entry * fee + stop_exec * fee
        reward = (target_exec - entry) - entry * fee - target_exec * fee
    else:
        entry = entry_raw * (1 - slip)
        stop = entry + stop_mult * atr
        target = max(1e-12, entry - target_mult * atr)
        stop_exec = stop * (1 + slip)
        target_exec = target * (1 + slip)
        risk = (stop_exec - entry) + entry * fee + stop_exec * fee
        reward = (entry - target_exec) - entry * fee - target_exec * fee
    if entry <= 0 or stop <= 0 or target <= 0 or risk <= 0:
        return None
    return {
        "entry": float(entry), "stop": float(stop), "target": float(target),
        "fee_rate": fee, "slip_rate": slip, "funding_rate_8h": fund,
        "risk_per_unit": float(risk), "net_rr": float(reward / risk),
        "target_net_bps": float((reward / entry) * 10000.0),
    }


def edge_exit(x: pd.DataFrame, entry_i: int, econ: dict, direction: str,
              hold_bars: int, tf: int):
    end = min(len(x) - 1, entry_i + max(1, int(hold_bars)))
    stop, target = econ["stop"], econ["target"]
    slip = econ["slip_rate"]
    for j in range(entry_i, end + 1):
        b = x.iloc[j]
        if direction == "long":
            hit_stop = float(b["low"]) <= stop
            hit_target = float(b["high"]) >= target
            if hit_stop:
                return j, stop * (1 - slip), "STOP"
            if hit_target:
                return j, target * (1 - slip), "TARGET"
        else:
            hit_stop = float(b["high"]) >= stop
            hit_target = float(b["low"]) <= target
            if hit_stop:
                return j, stop * (1 + slip), "STOP"
            if hit_target:
                return j, target * (1 + slip), "TARGET"
    close = float(x.iloc[end]["close"])
    return end, close * (1 - slip if direction == "long" else 1 + slip), "TIME"


def edge_trade_row(x: pd.DataFrame, symbol: str, tf: int, signal_i: int, direction: str,
                   strategy_id: str, family: str, regime: str, stop_mult: float,
                   target_mult: float, hold_bars: int, cost_name: str, cost: dict):
    entry_i = signal_i + 1
    if entry_i >= len(x):
        return None
    atr = float(x.iloc[signal_i].get("atr14", np.nan))
    raw_entry = float(x.iloc[entry_i]["open"])
    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(raw_entry) or raw_entry <= 0:
        return None
    econ = edge_cost_econ(raw_entry, atr, stop_mult, target_mult, direction, cost)
    if not econ or econ["net_rr"] < CFG.min_net_rr or econ["target_net_bps"] < CFG.min_net_target_bps:
        return None
    exit_i, exit_px, reason = edge_exit(x, entry_i, econ, direction, hold_bars, tf)
    held_hours = max(0.0, (exit_i - entry_i + 1) * tf / 60.0)
    funding = econ["entry"] * econ["funding_rate_8h"] * (held_hours / 8.0)
    entry_fee = econ["entry"] * econ["fee_rate"]
    exit_fee = exit_px * econ["fee_rate"]
    gross = (exit_px - econ["entry"]) if direction == "long" else (econ["entry"] - exit_px)
    net = gross - entry_fee - exit_fee - funding
    r_mult = net / econ["risk_per_unit"] if econ["risk_per_unit"] > 0 else None
    return {
        "symbol": symbol, "timeframe": tf, "strategy_id": strategy_id, "family": family,
        "direction": direction, "regime": regime, "cost_profile": cost_name,
        "entry_ts": int(x.iloc[entry_i]["time"]), "exit_ts": int(x.iloc[exit_i]["time"]),
        "r": r_mult, "reason": reason,
    }


def _edge_context(frames: dict[tuple[str, int], pd.DataFrame]):
    prepared = {}
    for (sym, tf), df in frames.items():
        if tf in (15, 60):
            x = indicators(df).reset_index(drop=True)
            bars24 = max(1, int(24 * 60 / tf))
            bars48 = max(1, int(48 * 60 / tf))
            x["ret24"] = x["close"].pct_change(bars24)
            x["ret48"] = x["close"].pct_change(bars48)
            prepared[(sym, tf)] = x
    btc = prepared.get(("XBTUSD", 60))
    if btc is None or len(btc) < 220:
        raise RuntimeError("BTC 60m context unavailable")
    regime = edge_regime_frame(frames[("XBTUSD", 60)])
    return prepared, regime


def edge_explorer(frames: dict[tuple[str, int], pd.DataFrame]):
    started = time.time()
    DBX.set("edge", {"state": "running", "started": iso(), "note": "Research-only long/short regime explorer; live engine unchanged."})
    prepared, regime_df = _edge_context(frames)
    reg_times = regime_df["time"].to_numpy(dtype=np.int64)
    reg_values = regime_df["regime"].to_numpy()
    btc60 = prepared[("XBTUSD", 60)]
    btc_times = btc60["time"].to_numpy(dtype=np.int64)
    btc_ret24 = btc60["ret24"].to_numpy(dtype=float)
    rows = []
    errors = []

    def regime_at(ts: int):
        k = _asof_index(reg_times, ts)
        return str(reg_values[k]) if k >= 0 else "unknown"

    def btc_r24_at(ts: int):
        k = _asof_index(btc_times, ts)
        return float(btc_ret24[k]) if k >= 0 and np.isfinite(btc_ret24[k]) else np.nan

    # 1) Regime-aware long/short breakouts, relative strength/weakness, and sideways mean reversion.
    sixty = [(sym, x) for (sym, tf), x in prepared.items() if tf == 60]
    total_units = len(sixty) + sum(1 for (sym, tf) in prepared if tf == 15)
    done = 0
    for symbol, x in sixty:
        try:
            busy = {}
            for i in range(220, len(x) - 2):
                r = x.iloc[i]
                ts = int(r["time"])
                reg = regime_at(ts)
                br = btc_r24_at(ts)
                if not np.isfinite(float(r.get("atr14", np.nan))):
                    continue
                specs = []
                # Regime breakout uses a wider target and can go both directions.
                if reg == "bull" and np.isfinite(r.get("prior_high20", np.nan)) and r["close"] > r["prior_high20"] and r["ema20"] > r["ema50"]:
                    specs.append(("regime_breakout_long", "regime_breakout", "long", 1.6, 3.2, 24))
                if reg == "bear" and np.isfinite(r.get("prior_low20", np.nan)) and r["close"] < r["prior_low20"] and r["ema20"] < r["ema50"]:
                    specs.append(("regime_breakout_short", "regime_breakout", "short", 1.6, 3.2, 24))
                # Relative strength versus BTC: different source of signal from pure price breakout.
                if np.isfinite(br) and np.isfinite(r.get("ret24", np.nan)):
                    spread = float(r["ret24"]) - br
                    if spread >= 0.025 and r["ema20"] > r["ema50"] and 50 <= r["rsi14"] <= 75:
                        specs.append(("relative_strength_24h_long", "relative_strength", "long", 1.5, 2.8, 18))
                    if spread <= -0.025 and r["ema20"] < r["ema50"] and 25 <= r["rsi14"] <= 50:
                        specs.append(("relative_weakness_24h_short", "relative_strength", "short", 1.5, 2.8, 18))
                # Mean reversion only when BTC is sideways.
                if reg == "sideways":
                    if np.isfinite(r.get("bb_lower", np.nan)) and r["close"] < r["bb_lower"] and r["rsi14"] < 32:
                        specs.append(("sideways_meanrev_long", "sideways_meanrev", "long", 1.2, 1.9, 12))
                    if np.isfinite(r.get("bb_upper", np.nan)) and r["close"] > r["bb_upper"] and r["rsi14"] > 68:
                        specs.append(("sideways_meanrev_short", "sideways_meanrev", "short", 1.2, 1.9, 12))
                for sid, fam, direction, sm, tm, hold in specs:
                    for cost_name, cost in LAB_COST_PROFILES.items():
                        key=(sid,cost_name)
                        if i < busy.get(key, -1):
                            continue
                        tr = edge_trade_row(x, symbol, 60, i, direction, sid, fam, reg, sm, tm, hold, cost_name, cost)
                        if tr:
                            rows.append(tr)
                            # prevent the same research rule from stacking on itself for one symbol
                            busy[key] = i + hold
        except Exception as e:
            errors.append(f"{symbol}/60m: {e}")
        done += 1
        if done == 1 or done % 3 == 0:
            DBX.set("edge", {"state":"running","started":iso(),"completed_units":done,"total_units":total_units,"trade_rows_so_far":len(rows),"current":f"{symbol}/60m","seconds":round(time.time()-started,1),"errors":errors[-5:]})

    # 2) Multi-timeframe continuation: 60m symbol trend + BTC regime, entry on 15m breakout/breakdown.
    for (symbol, tf), x15 in prepared.items():
        if tf != 15 or (symbol, 60) not in prepared:
            continue
        try:
            x60 = prepared[(symbol, 60)]
            t60 = x60["time"].to_numpy(dtype=np.int64)
            busy = {}
            for i in range(220, len(x15) - 2):
                r = x15.iloc[i]
                ts = int(r["time"])
                k = _asof_index(t60, ts)
                if k < 200:
                    continue
                h = x60.iloc[k]
                reg = regime_at(ts)
                specs=[]
                if reg != "bear" and h["ema20"] > h["ema50"] > h["ema200"] and np.isfinite(r.get("prior_high10", np.nan)) and r["close"] > r["prior_high10"]:
                    specs.append(("mtf_60trend_15break_long", "multi_timeframe", "long", 1.3, 2.6, 32))
                if reg != "bull" and h["ema20"] < h["ema50"] < h["ema200"] and np.isfinite(r.get("prior_low10", np.nan)) and r["close"] < r["prior_low10"]:
                    specs.append(("mtf_60trend_15break_short", "multi_timeframe", "short", 1.3, 2.6, 32))
                for sid,fam,direction,sm,tm,hold in specs:
                    for cost_name,cost in LAB_COST_PROFILES.items():
                        key=(sid,cost_name)
                        if i < busy.get(key,-1):
                            continue
                        tr=edge_trade_row(x15,symbol,15,i,direction,sid,fam,reg,sm,tm,hold,cost_name,cost)
                        if tr:
                            rows.append(tr); busy[key]=i+hold
        except Exception as e:
            errors.append(f"{symbol}/15m: {e}")
        done += 1
        if done % 3 == 0 or done == total_units:
            DBX.set("edge", {"state":"running","started":iso(),"completed_units":done,"total_units":total_units,"trade_rows_so_far":len(rows),"current":f"{symbol}/15m","seconds":round(time.time()-started,1),"errors":errors[-5:]})

    # 3) Cross-sectional 24h momentum on 60m data. Long leaders / short laggards, optional regime filter.
    try:
        xs = {sym:x.set_index("time") for sym,x in sixty}
        common = sorted(set.intersection(*[set(x.index) for x in xs.values()])) if xs else []
        busy = {}
        for ts in common:
            vals=[]
            for sym,x in xs.items():
                try:
                    rr=float(x.loc[ts,"ret24"])
                    if np.isfinite(rr): vals.append((sym,rr))
                except Exception: pass
            if len(vals) < 8:
                continue
            vals.sort(key=lambda z:z[1])
            dispersion=vals[-1][1]-vals[0][1]
            if dispersion < 0.05:
                continue
            picks=[(vals[-1][0],"long"),(vals[-2][0],"long"),(vals[0][0],"short"),(vals[1][0],"short")]
            reg=regime_at(int(ts))
            for sym,direction in picks:
                if (reg=="bull" and direction=="short") or (reg=="bear" and direction=="long"):
                    continue
                x=prepared[(sym,60)]
                times=x["time"].to_numpy(dtype=np.int64); i=_asof_index(times,int(ts))
                if i<220 or i>=len(x)-2: continue
                sid=f"xsec_momentum_24h_{direction}"
                for cost_name,cost in LAB_COST_PROFILES.items():
                    key=(sym,sid,cost_name)
                    if i < busy.get(key,-1): continue
                    tr=edge_trade_row(x,sym,60,i,direction,sid,"cross_sectional",reg,1.5,2.6,12,cost_name,cost)
                    if tr:
                        rows.append(tr); busy[key]=i+12
    except Exception as e:
        errors.append(f"cross_sectional: {e}")

    valid=[r for r in rows if r.get("r") is not None and np.isfinite(r["r"])]
    groups={}
    for r in valid:
        key=(r["strategy_id"],r["timeframe"],r["cost_profile"])
        groups.setdefault(key,[]).append(r)
    ranked=[]
    for (sid,tf,cost_name),rr in groups.items():
        sm=summarize_research(rr)
        if sm["trades"] < 10: continue
        fam=rr[0]["family"]; direction=rr[0]["direction"]
        row={"strategy_id":sid,"family":fam,"direction":direction,"timeframe":tf,"cost_profile":cost_name,**sm}
        row["robust_score"] = round((row.get("avg_r") or -9) * math.sqrt(max(1,row["trades"])) - 0.015*(row.get("max_drawdown_r") or 0),4)
        ranked.append(row)
    ranked.sort(key=lambda z:z["robust_score"],reverse=True)

    # Expanding walk-forward. Select up to three distinct families from training only.
    current_rows=[r for r in valid if r["cost_profile"]==EDGE_CURRENT_COST]
    if current_rows:
        lo=min(r["entry_ts"] for r in current_rows); hi=max(r["entry_ts"] for r in current_rows); span=max(1,hi-lo)
    else:
        lo=hi=0; span=1
    folds=[]; selected_oos=[]
    for fold_idx,(a,b) in enumerate(((0.40,0.60),(0.60,0.80),(0.80,1.001)),1):
        st=lo+int(span*a); en=lo+int(span*b)
        train=[r for r in current_rows if r["entry_ts"]<st]
        test=[r for r in current_rows if st<=r["entry_ts"]<en]
        tg={}
        for r in train: tg.setdefault((r["strategy_id"],r["timeframe"]),[]).append(r)
        cand=[]
        for (sid,tf),rr in tg.items():
            sm=summarize_research(rr)
            if sm["trades"]>=EDGE_MIN_TRAIN_TRADES and (sm.get("avg_r") or -99)>0 and (sm.get("profit_factor_r") or 0)>1.05:
                fam=rr[0]["family"]
                score=(sm["avg_r"] or 0)*math.sqrt(sm["trades"])-0.015*(sm.get("max_drawdown_r") or 0)
                cand.append((score,sid,tf,fam,sm))
        cand.sort(reverse=True,key=lambda z:z[0])
        chosen=[]; used=set()
        for item in cand:
            if item[3] in used: continue
            chosen.append(item); used.add(item[3])
            if len(chosen)>=3: break
        keys={(x[1],x[2]) for x in chosen}
        sel=[r for r in test if (r["strategy_id"],r["timeframe"]) in keys]
        selected_oos.extend(sel)
        sm=summarize_research(sel)
        folds.append({"fold":fold_idx,"train_rows":len(train),"eligible_candidates":len(cand),"selected":"; ".join(f"{x[1]}|{x[2]}m" for x in chosen) or None,"test_trades":sm["trades"],"avg_r":sm["avg_r"],"profit_factor_r":sm["profit_factor_r"],"total_r":sm["total_r"]})
    oos=summarize_research(selected_oos)
    profitable=sum(1 for f in folds if f.get("test_trades",0)>0 and (f.get("total_r") or 0)>0)
    wfo={"state":"done","folds":folds,"test_trades":oos["trades"],"avg_r":oos["avg_r"],"profit_factor_r":oos["profit_factor_r"],"total_r":oos["total_r"],"max_drawdown_r":oos["max_drawdown_r"],"profitable_folds":profitable}

    current_ranked=[g for g in ranked if g["cost_profile"]==EDGE_CURRENT_COST]
    best=current_ranked[0] if current_ranked else None
    checks={
        "current_cost_candidate_40_trades": bool(best and best.get("trades",0)>=40),
        "current_cost_candidate_pf_gt_1_10": bool(best and (best.get("profit_factor_r") or 0)>1.10),
        "oos_at_least_60_trades": wfo.get("test_trades",0)>=60,
        "oos_avg_r_positive": (wfo.get("avg_r") or -99)>0,
        "oos_pf_gt_1_05": (wfo.get("profit_factor_r") or 0)>1.05,
        "profitable_folds_at_least_2": profitable>=2,
    }
    gate="EDGE_CANDIDATE" if all(checks.values()) else "NO_ROBUST_EDGE_YET"
    best_by_cost=[]
    for cn in LAB_COST_PROFILES:
        rr=[g for g in ranked if g["cost_profile"]==cn]
        if rr: best_by_cost.append(rr[0])
    by_family=[]
    for fam in sorted({r["family"] for r in current_rows}):
        sm=summarize_research([r for r in current_rows if r["family"]==fam])
        by_family.append({"family":fam,**sm})
    by_family.sort(key=lambda z:(z.get("avg_r") if z.get("avg_r") is not None else -999),reverse=True)
    by_direction=[]
    for direction in ("long","short"):
        rr=[r for r in current_rows if r["direction"]==direction]
        sm=summarize_research(rr); by_direction.append({"direction":direction,**sm})

    result={
        "state":"done","finished":iso(),"seconds":round(time.time()-started,2),
        "scope":f"{RESEARCH_LOOKBACK_DAYS}d cached history · regime/relative-strength/mean-reversion/MTF/cross-sectional · long+short research · 3 cost stresses · report-only",
        "trade_rows":len(valid),"candidate_count":len(ranked),"evidence_gate":gate,"evidence_checks":checks,
        "top_candidates":ranked[:20],"best_by_cost":best_by_cost,"by_family":by_family,"by_direction":by_direction,
        "walk_forward":wfo,
        "note":"Research shorts are simulated only. No live short/order route exists. Selection is based on training data only; OOS is untouched until evaluation.",
        "errors":errors[-12:],
    }
    DBX.set("edge",result)
    return result

async def deep_research_once():
    started = time.time()
    total_units = len(RESEARCH_MARKETS) * len(RESEARCH_TIMEFRAMES)
    base_state = {
        "state": "running",
        "started": iso(),
        "lookback_days": RESEARCH_LOOKBACK_DAYS,
        "markets": len(RESEARCH_MARKETS),
        "timeframes": list(RESEARCH_TIMEFRAMES),
        "completed_units": 0,
        "total_units": total_units,
        "trades_so_far": 0,
    }
    DBX.set("research", base_state)
    DBX.set("lab", {"state": "waiting_for_deep_history", "started": iso(), "note": "Live engine remains independent."})
    DBX.set("edge", {"state": "waiting_for_deep_history", "started": iso(), "note": "Research-only; live engine remains independent."})
    trades: list[dict] = []
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    errors = []
    source_counts = {"bitstamp_deep": 0, "kraken_fallback": 0}
    completed = 0
    for symbol in RESEARCH_MARKETS:
        for tf in RESEARCH_TIMEFRAMES:
            source = "bitstamp_deep"
            try:
                try:
                    df = await asyncio.to_thread(BR.ohlc, symbol, tf, RESEARCH_LOOKBACK_DAYS)
                except Exception as deep_err:
                    source = "kraken_fallback"
                    errors.append(f"{symbol}/{tf}m deep source fallback: {deep_err}")
                    df = await asyncio.to_thread(K.ohlc, symbol, tf)
                source_counts[source] += 1
                frames[(symbol, tf)] = df
                part = await asyncio.to_thread(backtest_all_strategies, df, symbol, tf)
                trades.extend(part)
            except Exception as e:
                errors.append(f"{symbol}/{tf}m: {e}")
            completed += 1
            DBX.set("research", {
                **base_state,
                "completed_units": completed,
                "current": f"{symbol}/{tf}m",
                "trades_so_far": len(trades),
                "source_counts": source_counts,
                "seconds": round(time.time() - started, 1),
                "errors": errors[-6:],
            })
            await asyncio.sleep(0.03)

    valid = [t for t in trades if t.get("r") is not None and np.isfinite(t["r"])]
    overall = summarize_research(valid)
    by_strategy = grouped_research(valid, ("strategy",), min_trades=1)
    by_timeframe = grouped_research(valid, ("timeframe",), min_trades=1)
    by_setup = grouped_research(valid, ("strategy", "timeframe"), min_trades=10)[:24]
    wfo = walk_forward_report(valid)
    checks = {
        "sample_at_least_300": overall["trades"] >= 300,
        "base_avg_r_positive": (overall.get("avg_r") or -999) > 0,
        "base_profit_factor_gt_1_05": (overall.get("profit_factor_r") or 0) > 1.05,
        "wfo_test_at_least_100": wfo.get("test_trades", 0) >= 100,
        "wfo_avg_r_positive": (wfo.get("avg_r") or -999) > 0,
        "wfo_profitable_folds_at_least_2": wfo.get("profitable_folds", 0) >= 2,
    }
    gate = "RESEARCH_CANDIDATE" if all(checks.values()) else "INSUFFICIENT_EDGE_EVIDENCE"
    deep_units = source_counts["bitstamp_deep"]
    result = {
        "state": "done",
        "finished": iso(),
        "seconds": round(time.time() - started, 2),
        "scope": f"{RESEARCH_LOOKBACK_DAYS}d target window · 12 markets · 5/15/60m · Bitstamp deep history with Kraken fallback · fixed live strategies · report-only",
        "data_sources": source_counts,
        "deep_history_coverage_pct": round(100 * deep_units / total_units, 1) if total_units else 0.0,
        "lookback_days": RESEARCH_LOOKBACK_DAYS,
        "markets": len(RESEARCH_MARKETS),
        "timeframes": list(RESEARCH_TIMEFRAMES),
        **overall,
        "evidence_gate": gate,
        "evidence_checks": checks,
        "by_strategy": by_strategy,
        "by_timeframe": by_timeframe,
        "by_setup": by_setup,
        "walk_forward": wfo,
        "errors": errors[-12:],
    }
    DBX.set("research", result)

    # Reuse the already-downloaded history so parameter research does not hit the API again.
    try:
        await asyncio.to_thread(parameter_lab, frames)
    except Exception as e:
        DBX.set("lab", {"state": "error", "at": iso(), "error": repr(e)})
    try:
        await asyncio.to_thread(edge_explorer, frames)
    except Exception as e:
        DBX.set("edge", {"state": "error", "at": iso(), "error": repr(e)})
    return result


STOP = asyncio.Event()
ENGINE_TASK: Optional[asyncio.Task] = None
RESEARCH_TASK: Optional[asyncio.Task] = None


async def engine_loop():
    while not STOP.is_set():
        try:
            await scan_once()
        except Exception as e:
            DBX.set("engine_error", {"at": iso(), "error": repr(e)})
        try:
            await asyncio.wait_for(STOP.wait(), timeout=CFG.scan_seconds)
        except asyncio.TimeoutError:
            pass


async def research_loop():
    await asyncio.sleep(45)
    while not STOP.is_set():
        try:
            await deep_research_once()
        except Exception as e:
            DBX.set("research", {"state": "error", "at": iso(), "error": repr(e)})
        try:
            await asyncio.wait_for(STOP.wait(), timeout=RESEARCH_REFRESH_HOURS * 3600)
        except asyncio.TimeoutError:
            pass


app = FastAPI(title=APP_VERSION)


@app.on_event("startup")
async def startup():
    global ENGINE_TASK, RESEARCH_TASK
    if ENGINE_TASK is None or ENGINE_TASK.done():
        ENGINE_TASK = asyncio.create_task(engine_loop())
    if RESEARCH_TASK is None or RESEARCH_TASK.done():
        RESEARCH_TASK = asyncio.create_task(research_loop())


@app.on_event("shutdown")
async def shutdown():
    STOP.set()


def status_payload():
    eq, cash, gross = marked_equity()
    a = DBX.account()
    return {
        "version": APP_VERSION,
        "mode": "PAPER ONLY — no live-order route exists",
        "preregistration_hash": PREREGISTRATION_HASH,
        "markets": CFG.markets,
        "timeframes": CFG.timeframes,
        "scan_seconds": CFG.scan_seconds,
        "equity": eq,
        "cash": cash,
        "gross_exposure": gross,
        "gross_leverage": gross / eq if eq > 0 else None,
        "realized_pnl_all_time": a["realized_pnl"],
        "open_positions": len(DBX.positions()),
        "last_scan": DBX.get("last_scan", {}),
        "performance": performance_payload(),
        "research": DBX.get("research", {"state": "waiting"}),
        "lab": DBX.get("lab", {"state": "waiting"}),
        "edge": DBX.get("edge", {"state": "waiting"}),
        "risk": {
            "risk_per_trade_pct": CFG.risk_per_trade * 100,
            "max_daily_loss_pct": CFG.max_daily_loss * 100,
            "max_heat_pct": CFG.max_heat * 100,
            "max_open_positions": CFG.max_positions,
            "max_gross_leverage": CFG.max_gross_leverage,
            "max_position_notional_pct": CFG.max_position_notional * 100,
            "correlation_threshold": CFG.corr_threshold,
            "max_correlated_positions": CFG.max_correlated_positions,
            "fee_bps_each_side": CFG.fee_rate * 10000,
            "slippage_bps_each_side": CFG.slippage_rate * 10000,
            "funding_stress_bps_8h": CFG.funding_rate_8h * 10000,
            "min_net_rr": CFG.min_net_rr,
            "leverage_note": "portfolio gross exposure can reach 2x; trade risk remains stop-based",
            "shorts": False,
            "research_shorts": True,
        },
        "persistence": {
            "db_path": CFG.db_path,
            "persistent_state_id": DBX.get("persistent_state_id"),
            "boot_id": BOOT_ID,
            "positions_rows": DBX.count("positions"),
            "trades_rows": DBX.count("trades"),
            "signals_rows": DBX.count("signals"),
        },
    }


@app.get("/health")
def health():
    return {"ok": True, "version": APP_VERSION, "boot_id": BOOT_ID}


@app.get("/api/status")
def api_status():
    return JSONResponse(status_payload())


@app.get("/api/positions")
def api_positions():
    return JSONResponse(DBX.positions())


@app.get("/api/trades")
def api_trades(limit: int = 100):
    return JSONResponse(DBX.trades(min(max(limit, 1), 500)))


@app.get("/api/signals")
def api_signals(limit: int = 200):
    return JSONResponse(DBX.signals(min(max(limit, 1), 1000)))


@app.get("/api/performance")
def api_performance():
    return JSONResponse(performance_payload())


@app.get("/api/research")
def api_research():
    return JSONResponse(DBX.get("research", {"state": "waiting"}))


@app.get("/api/lab")
def api_lab():
    return JSONResponse(DBX.get("lab", {"state": "waiting"}))


@app.get("/api/edge")
def api_edge():
    return JSONResponse(DBX.get("edge", {"state": "waiting"}))


HTML = r"""
<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Alpha v1.6</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;margin:20px;background:#0d1117;color:#e6edf3;max-width:1350px}
h1{margin-bottom:4px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}.c{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px}.b{font-size:24px;font-weight:700}.m{color:#8b949e}.warn{background:#2d2205;padding:12px;border-radius:10px}table{width:100%;border-collapse:collapse;background:#161b22;margin-top:10px;overflow:auto}th,td{padding:8px;border-bottom:1px solid #30363d;font-size:12px;text-align:left}code{color:#79c0ff}.small{font-size:12px}
</style></head><body>
<h1>Alpha v1.6 — Regime + Relative Strength Edge Explorer</h1>
<div class="m">Live engine unchanged · new regime/relative-strength/cross-sectional/MTF research · long+short simulations · walk-forward · paper only.</div>
<p class="warn"><b>Research/paper only.</b> New short signals are simulated only. No live short or real-order route exists. Walk-forward out-of-sample evidence is the promotion gate.</p>
<div id="cards" class="grid"></div>
<h2>Live performance</h2><div id="perf"></div>
<h2>Fixed-strategy deep research</h2><div id="research"></div><div id="researchstrat"></div><div id="researchtf"></div><h3>Fixed-strategy walk-forward</h3><div id="wfo"></div>
<h2>Research Lab — parameter search & cost stress</h2><div id="lab"></div><h3>Best candidates (full sample; hypothesis generation)</h3><div id="labtop"></div><h3>Best candidate by cost stress</h3><div id="labcost"></div><h3>Lab walk-forward selection</h3><div id="labwfo"></div>
<h2>v1.6 Edge Explorer — new sources of edge</h2><div id="edge"></div><h3>Top edge candidates</h3><div id="edgetop"></div><h3>Best by cost stress</h3><div id="edgecost"></div><h3>Current-cost family / direction diagnostics</h3><div id="edgefam"></div><div id="edgedir"></div><h3>Edge walk-forward selection</h3><div id="edgewfo"></div>
<h2>Live strategy performance</h2><div id="strat"></div>
<h2>Open positions</h2><div id="pos"></div>
<h2>Recent trades</h2><div id="trades"></div>
<h2>Recent signals</h2><div id="sig"></div>
<script>
const n=(x,d=2)=>x==null?'—':Number(x).toFixed(d);
function tbl(r,c){if(!r||!r.length)return'<div class="m">None</div>';return'<div style="overflow:auto"><table><tr>'+c.map(x=>'<th>'+x+'</th>').join('')+'</tr>'+r.map(a=>'<tr>'+c.map(x=>'<td>'+String(a[x]??'')+'</td>').join('')+'</tr>').join('')+'</table></div>'}
async function go(){
 let[s,p,t,g]=await Promise.all([fetch('/api/status').then(r=>r.json()),fetch('/api/positions').then(r=>r.json()),fetch('/api/trades?limit=30').then(r=>r.json()),fetch('/api/signals?limit=50').then(r=>r.json())]);
 let l=s.last_scan||{},f=s.performance||{},r=s.research||{},a=s.lab||{},e=s.edge||{};
 document.getElementById('cards').innerHTML=[['Equity','$'+n(s.equity)],['Net P&L','$'+n(f.net_pnl)],['Closed trades',f.closed_trades||0],['Win rate',n(f.win_rate_pct,1)+'%'],['Open positions',s.open_positions],['Gross leverage',n(s.gross_leverage,2)+'x'],['Markets',(s.markets||[]).length],['Qualified',l.economically_qualified||0],['Opened last scan',l.opened||0],['Heat',n(l.portfolio_heat_pct,3)+'%']].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b">'+x[1]+'</div></div>').join('');
 document.getElementById('perf').innerHTML='<div class="grid">'+[['Profit factor',n(f.profit_factor)],['Avg R',n(f.avg_r,3)],['Fees','$'+n(f.fees_paid)],['Funding','$'+n(f.funding_paid)],['Trades today',f.trades_today||0],['Max positions',s.risk.max_open_positions]].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b">'+x[1]+'</div></div>').join('')+'</div>';
 document.getElementById('research').innerHTML='<div class="grid">'+[['State',r.state||'waiting'],['Progress',(r.completed_units??'—')+'/'+(r.total_units??'—')],['Historical trades',r.trades??r.trades_so_far??0],['Avg R',n(r.avg_r,3)],['Win rate',n(r.win_rate_pct,1)+'%'],['Profit factor',n(r.profit_factor_r,2)],['Max DD',n(r.max_drawdown_r,2)+'R'],['Evidence',r.evidence_gate||'running'],['Deep coverage',n(r.deep_history_coverage_pct,1)+'%'],['Scope',r.scope||((r.lookback_days||90)+'d research starts ~45s after deploy')]].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div>';
 document.getElementById('researchstrat').innerHTML=tbl(r.by_strategy||[],['strategy','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r']);
 document.getElementById('researchtf').innerHTML='<h3>By timeframe</h3>'+tbl(r.by_timeframe||[],['timeframe','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r']);
 let w=r.walk_forward||{}; document.getElementById('wfo').innerHTML='<div class="grid">'+[['State',w.state||'waiting'],['OOS trades',w.test_trades||0],['OOS Avg R',n(w.avg_r,3)],['OOS PF',n(w.profit_factor_r,2)],['OOS Total R',n(w.total_r,2)],['Profitable folds',(w.profitable_folds||0)+'/3']].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div>'+tbl(w.folds||[],['fold','train_trades','eligible_setups','test_trades','avg_r','profit_factor_r','total_r']);
 document.getElementById('lab').innerHTML='<div class="grid">'+[['State',a.state||'waiting'],['Progress',(a.completed_units??'—')+'/'+(a.total_units??'—')],['Trade rows',a.trade_rows??a.trade_rows_so_far??0],['Candidates',a.candidate_count??'—'],['Evidence',a.evidence_gate||'running'],['Seconds',n(a.seconds,1)],['Scope',a.scope||a.note||'waiting for deep history']].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div><div class="m small">'+(a.multiple_testing_note||'')+'</div>';
 let top=(a.top_candidates||[]).map(x=>({...x,params:JSON.stringify(x.params||{})})); document.getElementById('labtop').innerHTML=tbl(top,['variant_id','timeframe','cost_profile','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r','robust_score','params']);
 let bc=(a.best_by_cost||[]).map(x=>({...x,params:JSON.stringify(x.params||{})})); document.getElementById('labcost').innerHTML=tbl(bc,['cost_profile','variant_id','timeframe','trades','avg_r','profit_factor_r','total_r','max_drawdown_r','params']);
 let lw=a.walk_forward||{}; document.getElementById('labwfo').innerHTML='<div class="grid">'+[['State',lw.state||'waiting'],['OOS trades',lw.test_trades||0],['OOS Avg R',n(lw.avg_r,3)],['OOS PF',n(lw.profit_factor_r,2)],['OOS Total R',n(lw.total_r,2)],['Profitable folds',(lw.profitable_folds||0)+'/3']].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div>'+tbl(lw.folds||[],['fold','train_trade_rows','eligible_candidates','selected','selected_train_trades','selected_train_avg_r','test_trades','avg_r','profit_factor_r','total_r']);
 document.getElementById('edge').innerHTML='<div class="grid">'+[['State',e.state||'waiting'],['Progress',(e.completed_units??'—')+'/'+(e.total_units??'—')],['Trade rows',e.trade_rows??e.trade_rows_so_far??0],['Candidates',e.candidate_count??'—'],['Evidence',e.evidence_gate||'running'],['Seconds',n(e.seconds,1)],['Scope',e.scope||e.note||'waiting for deep history']].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div><div class="m small">'+(e.note||'')+'</div>';
 let et=e.top_candidates||[]; document.getElementById('edgetop').innerHTML=tbl(et,['strategy_id','family','direction','timeframe','cost_profile','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r','robust_score']);
 document.getElementById('edgecost').innerHTML=tbl(e.best_by_cost||[],['cost_profile','strategy_id','family','direction','timeframe','trades','avg_r','profit_factor_r','total_r','max_drawdown_r']);
 document.getElementById('edgefam').innerHTML='<h4>By family</h4>'+tbl(e.by_family||[],['family','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r']);
 document.getElementById('edgedir').innerHTML='<h4>By direction</h4>'+tbl(e.by_direction||[],['direction','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r']);
 let ew=e.walk_forward||{}; document.getElementById('edgewfo').innerHTML='<div class="grid">'+[['State',ew.state||'waiting'],['OOS trades',ew.test_trades||0],['OOS Avg R',n(ew.avg_r,3)],['OOS PF',n(ew.profit_factor_r,2)],['OOS Total R',n(ew.total_r,2)],['Profitable folds',(ew.profitable_folds||0)+'/3']].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div>'+tbl(ew.folds||[],['fold','train_rows','eligible_candidates','selected','test_trades','avg_r','profit_factor_r','total_r']);
 document.getElementById('strat').innerHTML=tbl(f.by_strategy||[],['strategy','trades','net_pnl','win_rate_pct','profit_factor','avg_r']);
 document.getElementById('pos').innerHTML=tbl(p,['symbol','timeframe','strategy','entry_price','stop_price','target_price','last_price']);
 document.getElementById('trades').innerHTML=tbl(t,['exit_time','symbol','timeframe','strategy','net_pnl','r_multiple','fees','funding','exit_reason']);
 document.getElementById('sig').innerHTML=tbl(g,['ts','symbol','timeframe','strategy','status','detail']);
}
go();setInterval(go,10000);
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(HTML)
