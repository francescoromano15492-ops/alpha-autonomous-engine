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

APP_VERSION = "Alpha v1.3 — Ranked Edge + 2x Paper Engine"
KRAKEN_BASE = "https://api.kraken.com/0/public"
DEFAULT_MARKETS = [
    "XBTUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD", "LINKUSD", "LTCUSD",
    "AVAXUSD", "DOTUSD", "BCHUSD", "ATOMUSD", "XLMUSD", "UNIUSD", "AAVEUSD", "ETCUSD",
    "ALGOUSD", "NEARUSD", "FILUSD", "ICPUSD", "INJUSD", "SUIUSD", "ARBUSD", "OPUSD", "TRXUSD",
]
RESEARCH_MARKETS = DEFAULT_MARKETS[:12]
RESEARCH_TIMEFRAMES = (15, 60)
MAX_HOLD_BARS = {5: 72, 15: 48, 60: 24}

PREREGISTRATION = {
    "version": "v1.3-ranked-edge-2x-paper",
    "mode": "paper_only",
    "data_source": "Kraken public REST",
    "markets_default": DEFAULT_MARKETS,
    "timeframes": [5, 15, 60],
    "design": {
        "candidate_ranking": True,
        "correlation_guard": True,
        "mandatory_stop": True,
        "breakeven_trailing": True,
        "stale_trade_exit": True,
        "quick_historical_research": True,
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
        # V1.3 uses new cap variable names intentionally, so old v1.2 ALPHA_MAX_POSITIONS=5
        # cannot silently keep the engine at five slots.
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
        self.s.headers.update({"User-Agent": "Alpha-v1.3-paper-research/1.0"})

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
    x["prior_high10"] = h.shift(1).rolling(10).max()
    x["prior_high20"] = h.shift(1).rolling(20).max()
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
    max_bars = MAX_HOLD_BARS.get(tf, 24)
    end = min(len(x) - 1, entry_i + max_bars)
    for j in range(entry_i, end + 1):
        bar = x.iloc[j]
        hit_stop = bar["low"] <= stop
        hit_target = bar["high"] >= target
        if hit_stop and hit_target:
            return j, stop * (1 - CFG.slippage_rate), "STOP_BOTH"
        if hit_stop:
            return j, stop * (1 - CFG.slippage_rate), "STOP"
        if hit_target:
            return j, target * (1 - CFG.slippage_rate), "TARGET"
    return end, float(x.iloc[end]["close"]) * (1 - CFG.slippage_rate), "TIME"


def backtest_one(df: pd.DataFrame, symbol: str, tf: int):
    x = indicators(df).reset_index(drop=True)
    trades = []
    i = 220
    while i < len(x) - 2:
        sigs = signals_at(x, i)
        if not sigs:
            i += 1
            continue
        entry_i = i + 1
        raw_entry = float(x.iloc[entry_i]["open"])
        entry = raw_entry * (1 + CFG.slippage_rate)
        candidates = []
        for strategy, atr, sm, tm in sigs:
            stop = entry - sm * atr
            target = entry + tm * atr
            ok, _, econ = economic_precheck(entry, stop, target)
            if ok:
                score = econ["rr"] + min(max(econ["bps"], 0.0), 600.0) / 600.0
                candidates.append((score, strategy, stop, target, econ))
        if not candidates:
            i += 1
            continue
        candidates.sort(reverse=True, key=lambda z: z[0])
        _, strategy, stop, target, econ = candidates[0]
        exit_i, exit_px, reason = historical_exit(x, entry_i, entry, stop, target, tf)
        held_hours = max(0.0, (exit_i - entry_i + 1) * tf / 60.0)
        funding_per_unit = entry * CFG.funding_rate_8h * (held_hours / 8.0)
        net_per_unit = (exit_px - entry) - entry * CFG.fee_rate - exit_px * CFG.fee_rate - funding_per_unit
        r_mult = net_per_unit / econ["loss"] if econ["loss"] > 0 else None
        trades.append({"symbol": symbol, "timeframe": tf, "strategy": strategy, "r": r_mult, "reason": reason})
        i = max(i + 1, exit_i + 1)
    return trades


async def quick_research_once():
    started = time.time()
    DBX.set("research", {"state": "running", "started": iso(), "markets": RESEARCH_MARKETS, "timeframes": RESEARCH_TIMEFRAMES})
    trades = []
    errors = []
    for symbol in RESEARCH_MARKETS:
        for tf in RESEARCH_TIMEFRAMES:
            try:
                df = await asyncio.to_thread(K.ohlc, symbol, tf)
                trades.extend(await asyncio.to_thread(backtest_one, df, symbol, tf))
            except Exception as e:
                errors.append(f"{symbol}/{tf}: {e}")
            await asyncio.sleep(0.15)
    valid = [t for t in trades if t["r"] is not None and np.isfinite(t["r"])]
    by = []
    for strategy in sorted({t["strategy"] for t in valid}):
        sr = [t["r"] for t in valid if t["strategy"] == strategy]
        wins = [r for r in sr if r > 0]
        losses = [r for r in sr if r < 0]
        gp, gl = sum(wins), abs(sum(losses))
        by.append({
            "strategy": strategy,
            "trades": len(sr),
            "avg_r": round(float(np.mean(sr)), 3) if sr else None,
            "win_rate_pct": round(100 * len(wins) / len(sr), 1) if sr else 0.0,
            "profit_factor_r": round(gp / gl, 2) if gl > 0 else None,
            "total_r": round(sum(sr), 2),
        })
    rs = [t["r"] for t in valid]
    result = {
        "state": "done",
        "finished": iso(),
        "seconds": round(time.time() - started, 2),
        "scope": "recent Kraken OHLC window; screening only, not proof of future profitability",
        "markets": len(RESEARCH_MARKETS),
        "timeframes": list(RESEARCH_TIMEFRAMES),
        "trades": len(rs),
        "avg_r": round(float(np.mean(rs)), 3) if rs else None,
        "win_rate_pct": round(100 * sum(r > 0 for r in rs) / len(rs), 1) if rs else 0.0,
        "total_r": round(sum(rs), 2) if rs else 0.0,
        "by_strategy": by,
        "errors": errors[-10:],
    }
    DBX.set("research", result)
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
    await asyncio.sleep(90)
    while not STOP.is_set():
        try:
            await quick_research_once()
        except Exception as e:
            DBX.set("research", {"state": "error", "at": iso(), "error": repr(e)})
        try:
            await asyncio.wait_for(STOP.wait(), timeout=6 * 3600)
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


HTML = r"""
<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Alpha v1.3</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;margin:20px;background:#0d1117;color:#e6edf3;max-width:1250px}
h1{margin-bottom:4px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}.c{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px}.b{font-size:24px;font-weight:700}.m{color:#8b949e}.warn{background:#2d2205;padding:12px;border-radius:10px}table{width:100%;border-collapse:collapse;background:#161b22;margin-top:10px;overflow:auto}th,td{padding:8px;border-bottom:1px solid #30363d;font-size:12px;text-align:left}code{color:#79c0ff}
</style></head><body>
<h1>Alpha v1.3 — Ranked Edge + 2x Paper Engine</h1>
<div class="m">25 crypto markets · 3 timeframes · ranked candidates · max 2x gross exposure · paper only.</div>
<p class="warn"><b>Research/paper only.</b> Leverage increases losses as well as gains. Hard stops, heat limits, correlation guard and funding stress remain active.</p>
<div id="cards" class="grid"></div>
<h2>Live performance</h2><div id="perf"></div>
<h2>Quick historical screening</h2><div id="research"></div><div id="researchstrat"></div>
<h2>Live strategy performance</h2><div id="strat"></div>
<h2>Open positions</h2><div id="pos"></div>
<h2>Recent trades</h2><div id="trades"></div>
<h2>Recent signals</h2><div id="sig"></div>
<script>
const n=(x,d=2)=>x==null?'—':Number(x).toFixed(d);
function tbl(r,c){if(!r||!r.length)return'<div class="m">None</div>';return'<div style="overflow:auto"><table><tr>'+c.map(x=>'<th>'+x+'</th>').join('')+'</tr>'+r.map(a=>'<tr>'+c.map(x=>'<td>'+String(a[x]??'')+'</td>').join('')+'</tr>').join('')+'</table></div>'}
async function go(){
 let[s,p,t,g]=await Promise.all([fetch('/api/status').then(r=>r.json()),fetch('/api/positions').then(r=>r.json()),fetch('/api/trades?limit=30').then(r=>r.json()),fetch('/api/signals?limit=50').then(r=>r.json())]);
 let l=s.last_scan||{},f=s.performance||{},r=s.research||{};
 document.getElementById('cards').innerHTML=[['Equity','$'+n(s.equity)],['Net P&L','$'+n(f.net_pnl)],['Closed trades',f.closed_trades||0],['Win rate',n(f.win_rate_pct,1)+'%'],['Open positions',s.open_positions],['Gross leverage',n(s.gross_leverage,2)+'x'],['Markets',(s.markets||[]).length],['Qualified',l.economically_qualified||0],['Opened last scan',l.opened||0],['Heat',n(l.portfolio_heat_pct,3)+'%']].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b">'+x[1]+'</div></div>').join('');
 document.getElementById('perf').innerHTML='<div class="grid">'+[['Profit factor',n(f.profit_factor)],['Avg R',n(f.avg_r,3)],['Fees','$'+n(f.fees_paid)],['Funding','$'+n(f.funding_paid)],['Trades today',f.trades_today||0],['Max positions',s.risk.max_open_positions]].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b">'+x[1]+'</div></div>').join('')+'</div>';
 document.getElementById('research').innerHTML='<div class="grid">'+[['State',r.state||'waiting'],['Historical trades',r.trades||0],['Avg R',n(r.avg_r,3)],['Win rate',n(r.win_rate_pct,1)+'%'],['Total R',n(r.total_r,2)],['Scope',r.scope||'starts ~90s after deploy']].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div>';
 document.getElementById('researchstrat').innerHTML=tbl(r.by_strategy||[],['strategy','trades','avg_r','win_rate_pct','profit_factor_r','total_r']);
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
