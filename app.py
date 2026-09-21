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

APP_VERSION = "Alpha v2.1 — Parallel Research + Shadow Forward"
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

# v1.7 specialist research is intentionally separate from the older 90d labs.
# It uses 60m history only, so extending to 180d and more markets does not multiply
# the heavy 5m/15m data load. It remains report-only and cannot change live rules.
RS_LOOKBACK_DAYS = max(120, min(365, int(os.getenv("ALPHA_RS_LOOKBACK_DAYS", "180"))))
RS_MARKETS = tuple(
    x.strip().upper() for x in os.getenv("ALPHA_RS_MARKETS", ",".join(DEFAULT_MARKETS)).split(",") if x.strip()
)
RS_WINDOWS_H = (12, 24, 48, 72)
RS_THRESHOLD_SCALES = (0.8, 1.2)
RS_RISK_PROFILES = ((1.5, 2.8), (1.8, 3.4))
RS_MIN_TRAIN_TRADES = max(20, int(os.getenv("ALPHA_RS_MIN_TRAIN_TRADES", "30")))
RS_MIN_DEEP_MARKETS = max(6, int(os.getenv("ALPHA_RS_MIN_DEEP_MARKETS", "10")))

# v1.8 architecture research uses the SAME complete 180d 60m frames downloaded by v1.7.
# Signals and raw exits are generated once, before transaction costs are applied.
# Every cost scenario is then evaluated on the exact same trade IDs (apples-to-apples).
ARCH_MIN_TRAIN_TRADES = max(20, int(os.getenv("ALPHA_ARCH_MIN_TRAIN_TRADES", "35")))
ARCH_MIN_DEEP_MARKETS = max(8, int(os.getenv("ALPHA_ARCH_MIN_DEEP_MARKETS", "10")))
ARCH_MAX_FAMILIES_PER_FOLD = max(1, min(4, int(os.getenv("ALPHA_ARCH_MAX_FAMILIES_PER_FOLD", "3"))))
ARCH_FOLD_BOUNDS = ((0.30,0.44),(0.44,0.58),(0.58,0.72),(0.72,0.86),(0.86,1.001))

# Research Lab is deliberately report-only: it never changes live rules or sends orders.
# Cost profiles are stress scenarios, not claims about any specific venue.
LAB_COST_PROFILES = {
    "current_40fee_5slip": {"fee_bps": 40.0, "slippage_bps": 5.0, "funding_bps_8h": 1.0},
    "mid_20fee_4slip": {"fee_bps": 20.0, "slippage_bps": 4.0, "funding_bps_8h": 1.0},
    "lean_10fee_3slip": {"fee_bps": 10.0, "slippage_bps": 3.0, "funding_bps_8h": 0.5},
}

# v1.8.1 architecture-only cost overlays.
# Kraken fee inputs are reference values for Kraken Pro spot tiers observed in Sep 2026.
# They do NOT imply the final execution venue or guarantee maker fills.
# Slippage/funding remain explicit research assumptions rather than exchange promises.
ARCH_COST_PROFILES = {
    "kraken_t1_all_taker": {
        "entry_fee_bps": 80.0,
        "target_exit_fee_bps": 80.0,
        "stop_exit_fee_bps": 80.0,
        "time_exit_fee_bps": 80.0,
        "entry_slippage_bps": 5.0,
        "target_exit_slippage_bps": 5.0,
        "stop_exit_slippage_bps": 5.0,
        "time_exit_slippage_bps": 5.0,
        "funding_bps_8h": 1.0,
        "type": "venue_reference",
    },
    "kraken_t1_taker_entry_maker_target": {
        "entry_fee_bps": 80.0,
        "target_exit_fee_bps": 40.0,
        "stop_exit_fee_bps": 80.0,
        "time_exit_fee_bps": 80.0,
        "entry_slippage_bps": 5.0,
        "target_exit_slippage_bps": 0.0,
        "stop_exit_slippage_bps": 5.0,
        "time_exit_slippage_bps": 5.0,
        "funding_bps_8h": 1.0,
        "type": "venue_reference_mixed_fill_assumption",
    },
    "kraken_t3_all_taker": {
        "entry_fee_bps": 38.0,
        "target_exit_fee_bps": 38.0,
        "stop_exit_fee_bps": 38.0,
        "time_exit_fee_bps": 38.0,
        "entry_slippage_bps": 5.0,
        "target_exit_slippage_bps": 5.0,
        "stop_exit_slippage_bps": 5.0,
        "time_exit_slippage_bps": 5.0,
        "funding_bps_8h": 1.0,
        "type": "venue_reference",
    },
    "research_lean_10fee_3slip": {
        "entry_fee_bps": 10.0,
        "target_exit_fee_bps": 10.0,
        "stop_exit_fee_bps": 10.0,
        "time_exit_fee_bps": 10.0,
        "entry_slippage_bps": 3.0,
        "target_exit_slippage_bps": 3.0,
        "stop_exit_slippage_bps": 3.0,
        "time_exit_slippage_bps": 3.0,
        "funding_bps_8h": 0.5,
        "type": "synthetic_stress_only",
    },
}
ARCH_CURRENT_COST = "kraken_t1_all_taker"

# v1.9 low-turnover execution research. This is report-only and does not change the live engine.
# It deliberately uses slower synthetic bars and much wider targets so we can ask a concrete
# question: can any signal survive real venue-level execution costs instead of only cheap
# synthetic research assumptions?
EXEC_LOOKBACK_DAYS = max(240, min(540, int(os.getenv("ALPHA_EXEC_LOOKBACK_DAYS", "365"))))
EXEC_MARKETS = tuple(
    x.strip().upper() for x in os.getenv("ALPHA_EXEC_MARKETS", ",".join(DEFAULT_MARKETS[:12])).split(",") if x.strip()
)
EXEC_BAR_HOURS = (4, 12)
EXEC_MIN_DEEP_MARKETS = max(6, int(os.getenv("ALPHA_EXEC_MIN_DEEP_MARKETS", "8")))
EXEC_MIN_TRAIN_TRADES = max(15, int(os.getenv("ALPHA_EXEC_MIN_TRAIN_TRADES", "25")))
EXEC_FOLD_BOUNDS = ((0.30,0.44),(0.44,0.58),(0.58,0.72),(0.72,0.86),(0.86,1.001))
EXEC_MIN_RISK_BPS = max(35.0, float(os.getenv("ALPHA_EXEC_MIN_RISK_BPS", "70.0")))
EXEC_MAX_ABS_R = max(4.0, float(os.getenv("ALPHA_EXEC_MAX_ABS_R", "15.0")))

# Official Kraken reference fee inputs checked in Sep 2026:
# spot Tier 1 maker/taker 40/80 bps; spot Tier 3 maker/taker 22/38 bps;
# derivatives Tier 1 maker/taker 2/5 bps. Funding below is a research stress assumption,
# not historical funding data and not a promise about future funding.
EXEC_COST_PROFILES = {
    "kraken_spot_t1_all_taker": {
        "entry_fee_bps":80.0,"target_exit_fee_bps":80.0,"stop_exit_fee_bps":80.0,"time_exit_fee_bps":80.0,
        "entry_slippage_bps":5.0,"target_exit_slippage_bps":5.0,"stop_exit_slippage_bps":5.0,"time_exit_slippage_bps":5.0,
        "funding_bps_8h":0.0,"product":"spot","assumption":"all_taker",
    },
    "kraken_spot_t3_all_taker": {
        "entry_fee_bps":38.0,"target_exit_fee_bps":38.0,"stop_exit_fee_bps":38.0,"time_exit_fee_bps":38.0,
        "entry_slippage_bps":5.0,"target_exit_slippage_bps":5.0,"stop_exit_slippage_bps":5.0,"time_exit_slippage_bps":5.0,
        "funding_bps_8h":0.0,"product":"spot","assumption":"all_taker",
    },
    "kraken_futures_t1_all_taker_funding1": {
        "entry_fee_bps":5.0,"target_exit_fee_bps":5.0,"stop_exit_fee_bps":5.0,"time_exit_fee_bps":5.0,
        "entry_slippage_bps":3.0,"target_exit_slippage_bps":3.0,"stop_exit_slippage_bps":3.0,"time_exit_slippage_bps":3.0,
        "funding_bps_8h":1.0,"product":"perpetual_reference","assumption":"all_taker_funding_stress_1bp_8h",
    },
    "kraken_futures_t1_all_taker_funding5": {
        "entry_fee_bps":5.0,"target_exit_fee_bps":5.0,"stop_exit_fee_bps":5.0,"time_exit_fee_bps":5.0,
        "entry_slippage_bps":3.0,"target_exit_slippage_bps":3.0,"stop_exit_slippage_bps":3.0,"time_exit_slippage_bps":3.0,
        "funding_bps_8h":5.0,"product":"perpetual_reference","assumption":"all_taker_funding_stress_5bp_8h",
    },
    "kraken_futures_t1_maker_entry_target_funding1": {
        "entry_fee_bps":2.0,"target_exit_fee_bps":2.0,"stop_exit_fee_bps":5.0,"time_exit_fee_bps":5.0,
        "entry_slippage_bps":0.0,"target_exit_slippage_bps":0.0,"stop_exit_slippage_bps":3.0,"time_exit_slippage_bps":3.0,
        "funding_bps_8h":1.0,"product":"perpetual_reference","assumption":"maker_entry_and_target_fill_not_guaranteed",
    },
}
# Evidence must survive an all-taker futures profile with deliberately harsh funding stress.
EXEC_PRIMARY_COST = "kraken_futures_t1_all_taker_funding5"

# v2.0 forward-only shadow lab. These rules are frozen from the v1.9 hypotheses.
# They never place real or paper-account orders and have their own persistent ledger.
SHADOW_STRATEGIES = {
    "exec_12h_breakout20_short": {"bar_hours": 12, "stop_mult": 2.5, "target_mult": 7.0, "hold_bars": 20},
    "exec_4h_breakout40_short": {"bar_hours": 4, "stop_mult": 2.5, "target_mult": 6.5, "hold_bars": 42},
}
SHADOW_MARKETS = tuple(x.strip().upper() for x in os.getenv(
    "ALPHA_SHADOW_MARKETS", ",".join(DEFAULT_MARKETS[:12])
).split(",") if x.strip())
SHADOW_HISTORY_DAYS = max(110, min(220, int(os.getenv("ALPHA_SHADOW_HISTORY_DAYS", "140"))))
SHADOW_SCAN_SECONDS = max(60, int(os.getenv("ALPHA_SHADOW_SCAN_SECONDS", "60")))
SHADOW_MAX_SIGNAL_DELAY_MIN = max(5, int(os.getenv("ALPHA_SHADOW_MAX_SIGNAL_DELAY_MIN", "20")))
SHADOW_COST_PROFILE = "kraken_futures_t1_all_taker_funding5"
SHADOW_COST = EXEC_COST_PROFILES[SHADOW_COST_PROFILE]
SHADOW_MIN_FORWARD_DAYS = max(30, int(os.getenv("ALPHA_SHADOW_MIN_FORWARD_DAYS", "60")))
SHADOW_MIN_CLOSED_TRADES = max(20, int(os.getenv("ALPHA_SHADOW_MIN_CLOSED_TRADES", "50")))
LAB_MIN_TRAIN_TRADES = max(12, int(os.getenv("ALPHA_LAB_MIN_TRAIN_TRADES", "20")))
LAB_TOP_FULL_SAMPLE = max(5, min(30, int(os.getenv("ALPHA_LAB_TOP_FULL_SAMPLE", "15"))))

PREREGISTRATION = {
    "version": "v2.1-parallel-research-shadow-forward-paper",
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
        "relative_strength_specialist_180d": True,
        "five_fold_walk_forward": True,
        "multi_horizon_relative_strength": [12, 24, 48, 72],
        "multi_regime_multi_family_architecture": True,
        "same_entry_cost_stress": True,
        "outer_walk_forward_5fold": True,
        "oos_family_survivor_diagnostics": True,
        "architecture_sanity_filters": True,
        "kraken_reference_cost_model": True,
        "low_turnover_execution_edge": True,
        "synthetic_4h_12h_research_bars": True,
        "spot_vs_futures_cost_stress": True,
        "funding_stress_scenarios": True,
        "forward_only_shadow_lab": True,
        "persistent_shadow_ledger": True,
        "frozen_shadow_hypotheses": ["exec_12h_breakout20_short", "exec_4h_breakout40_short"],
        "parallel_robustness_lab": True,
        "deterministic_block_bootstrap": True,
        "cross_market_breadth": True,
        "chronological_stability_slices": 6,
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
# v1.8.1 architecture sanity thresholds. These affect research only.
ARCH_MIN_ATR_PCT = max(0.001, ef("ALPHA_ARCH_MIN_ATR_PCT", 0.005))
ARCH_MIN_RISK_BPS = max(10.0, ef("ALPHA_ARCH_MIN_RISK_BPS", 50.0))
ARCH_MAX_ABS_R = max(5.0, ef("ALPHA_ARCH_MAX_ABS_R", 10.0))
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
            CREATE TABLE IF NOT EXISTS shadow_positions(
              id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id TEXT NOT NULL, symbol TEXT NOT NULL,
              bar_hours INTEGER NOT NULL, direction TEXT NOT NULL, signal_bar_ts INTEGER NOT NULL,
              signal_bar_end_ts INTEGER NOT NULL, entry_time TEXT NOT NULL, entry_ts INTEGER NOT NULL,
              entry_raw REAL NOT NULL, entry_exec REAL NOT NULL, stop_raw REAL NOT NULL, target_raw REAL NOT NULL,
              risk_raw REAL NOT NULL, hold_hours_max REAL NOT NULL, last_price REAL NOT NULL,
              cost_profile TEXT NOT NULL, UNIQUE(strategy_id,symbol));
            CREATE TABLE IF NOT EXISTS shadow_trades(
              id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id TEXT NOT NULL, symbol TEXT NOT NULL,
              bar_hours INTEGER NOT NULL, direction TEXT NOT NULL, signal_bar_ts INTEGER NOT NULL,
              signal_bar_end_ts INTEGER NOT NULL, entry_time TEXT NOT NULL, exit_time TEXT NOT NULL,
              entry_raw REAL NOT NULL, entry_exec REAL NOT NULL, exit_raw REAL NOT NULL, exit_exec REAL NOT NULL,
              stop_raw REAL NOT NULL, target_raw REAL NOT NULL, held_hours REAL NOT NULL,
              gross_per_unit REAL NOT NULL, fees_per_unit REAL NOT NULL, funding_per_unit REAL NOT NULL,
              net_per_unit REAL NOT NULL, r_multiple REAL, exit_reason TEXT NOT NULL, cost_profile TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS shadow_signals(
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, strategy_id TEXT NOT NULL, symbol TEXT NOT NULL,
              bar_hours INTEGER NOT NULL, signal_bar_ts INTEGER NOT NULL, signal_bar_end_ts INTEGER NOT NULL,
              status TEXT NOT NULL, detail TEXT, UNIQUE(strategy_id,symbol,signal_bar_ts));
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



    # ---- v2.0 persistent shadow-forward ledger ----
    def shadow_positions(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM shadow_positions ORDER BY id").fetchall()]

    def shadow_trades(self, n=500):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM shadow_trades ORDER BY id DESC LIMIT ?", (int(n),)
        ).fetchall()]

    def shadow_signals(self, n=500):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM shadow_signals ORDER BY id DESC LIMIT ?", (int(n),)
        ).fetchall()]

    def shadow_signal_exists(self, strategy_id: str, symbol: str, signal_bar_ts: int) -> bool:
        return bool(self.conn.execute(
            "SELECT 1 FROM shadow_signals WHERE strategy_id=? AND symbol=? AND signal_bar_ts=?",
            (strategy_id, symbol, int(signal_bar_ts))
        ).fetchone())

    def shadow_has_position(self, strategy_id: str, symbol: str) -> bool:
        return bool(self.conn.execute(
            "SELECT 1 FROM shadow_positions WHERE strategy_id=? AND symbol=?", (strategy_id, symbol)
        ).fetchone())

    def shadow_signal(self, strategy_id, symbol, bar_hours, signal_bar_ts, signal_bar_end_ts, status, detail):
        try:
            self.conn.execute(
                "INSERT INTO shadow_signals(ts,strategy_id,symbol,bar_hours,signal_bar_ts,signal_bar_end_ts,status,detail) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (iso(), strategy_id, symbol, int(bar_hours), int(signal_bar_ts), int(signal_bar_end_ts), status, detail)
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass

    def shadow_open(self, strategy_id, symbol, bar_hours, direction, signal_bar_ts, signal_bar_end_ts,
                    entry_raw, entry_exec, stop_raw, target_raw, risk_raw, hold_hours_max, cost_profile):
        with self.conn:
            self.conn.execute(
                """INSERT INTO shadow_positions(strategy_id,symbol,bar_hours,direction,signal_bar_ts,signal_bar_end_ts,
                   entry_time,entry_ts,entry_raw,entry_exec,stop_raw,target_raw,risk_raw,hold_hours_max,last_price,cost_profile)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (strategy_id,symbol,int(bar_hours),direction,int(signal_bar_ts),int(signal_bar_end_ts),
                 iso(),int(time.time()),float(entry_raw),float(entry_exec),float(stop_raw),float(target_raw),
                 float(risk_raw),float(hold_hours_max),float(entry_raw),cost_profile)
            )

    def shadow_mark(self, pos_id: int, px: float):
        with self.conn:
            self.conn.execute("UPDATE shadow_positions SET last_price=? WHERE id=?", (float(px), int(pos_id)))

    def shadow_close(self, pos_id: int, exit_raw: float, reason: str):
        with self.conn:
            row=self.conn.execute("SELECT * FROM shadow_positions WHERE id=?",(int(pos_id),)).fetchone()
            if row is None:
                return None
            p=dict(row); direction=p["direction"]
            fee=float(SHADOW_COST["stop_exit_fee_bps" if str(reason).startswith("STOP") else
                                  "target_exit_fee_bps" if str(reason).startswith("TARGET") else "time_exit_fee_bps"])/10000.0
            slip=float(SHADOW_COST["stop_exit_slippage_bps" if str(reason).startswith("STOP") else
                                   "target_exit_slippage_bps" if str(reason).startswith("TARGET") else "time_exit_slippage_bps"])/10000.0
            x0=float(exit_raw)
            exit_exec=x0*(1-slip) if direction=="long" else x0*(1+slip)
            entry_exec=float(p["entry_exec"]); entry_fee_rate=float(SHADOW_COST["entry_fee_bps"])/10000.0
            entry_fee=entry_exec*entry_fee_rate; exit_fee=exit_exec*fee
            try:
                held=max(0.0,(utcnow()-datetime.fromisoformat(p["entry_time"])).total_seconds()/3600.0)
            except Exception:
                held=max(0.0,(int(time.time())-int(p["entry_ts"]))/3600.0)
            funding=entry_exec*(float(SHADOW_COST.get("funding_bps_8h",0.0))/10000.0)*(held/8.0)
            gross=(exit_exec-entry_exec) if direction=="long" else (entry_exec-exit_exec)
            net=gross-entry_fee-exit_fee-funding
            risk=float(p["risk_raw"]); r_mult=net/risk if risk>0 else None
            self.conn.execute(
                """INSERT INTO shadow_trades(strategy_id,symbol,bar_hours,direction,signal_bar_ts,signal_bar_end_ts,
                   entry_time,exit_time,entry_raw,entry_exec,exit_raw,exit_exec,stop_raw,target_raw,held_hours,
                   gross_per_unit,fees_per_unit,funding_per_unit,net_per_unit,r_multiple,exit_reason,cost_profile)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (p["strategy_id"],p["symbol"],p["bar_hours"],direction,p["signal_bar_ts"],p["signal_bar_end_ts"],
                 p["entry_time"],iso(),p["entry_raw"],entry_exec,x0,exit_exec,p["stop_raw"],p["target_raw"],held,
                 gross,entry_fee+exit_fee,funding,net,r_mult,str(reason),p["cost_profile"])
            )
            self.conn.execute("DELETE FROM shadow_positions WHERE id=?",(int(pos_id),))
            return {"r_multiple":r_mult,"net_per_unit":net,"held_hours":held}

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


# -------------------- v1.7 Relative Strength Specialist (research-only) --------------------
# Focuses on the only family that showed a positive full-sample signal in v1.6.
# Data selection, parameter selection and OOS evaluation are kept separate.

def _rs_threshold(window_h: int, scale: float) -> float:
    # 2.5% at 24h, volatility-time scaled; scale gives one permissive and one strict variant.
    return 0.025 * math.sqrt(max(1.0, float(window_h)) / 24.0) * float(scale)


def _rs_hold_bars(window_h: int) -> int:
    return {12: 12, 24: 18, 48: 24, 72: 36}.get(int(window_h), 18)


def rs_specialist_from_frames(frames: dict[str, pd.DataFrame], source_meta: dict | None = None):
    started = time.time()
    if "XBTUSD" not in frames:
        raise RuntimeError("BTC 60m deep history unavailable for relative-strength benchmark")
    if len(frames) < RS_MIN_DEEP_MARKETS:
        raise RuntimeError(f"insufficient deep-market coverage: {len(frames)}/{len(RS_MARKETS)}")

    prepared: dict[str, pd.DataFrame] = {}
    for sym, df in frames.items():
        x = indicators(df).reset_index(drop=True)
        for wh in RS_WINDOWS_H:
            x[f"ret{wh}"] = x["close"].pct_change(int(wh))
        prepared[sym] = x

    btc = prepared["XBTUSD"]
    btc_times = btc["time"].to_numpy(dtype=np.int64)
    btc_returns = {wh: btc[f"ret{wh}"].to_numpy(dtype=float) for wh in RS_WINDOWS_H}
    regime_df = edge_regime_frame(frames["XBTUSD"])
    reg_times = regime_df["time"].to_numpy(dtype=np.int64)
    reg_values = regime_df["regime"].to_numpy()

    def regime_at(ts: int) -> str:
        k = _asof_index(reg_times, ts)
        return str(reg_values[k]) if k >= 0 else "unknown"

    def btc_ret_at(ts: int, wh: int):
        k = _asof_index(btc_times, ts)
        arr = btc_returns[wh]
        return float(arr[k]) if k >= 0 and np.isfinite(arr[k]) else np.nan

    rows=[]
    errors=[]
    symbols=[sym for sym in prepared if sym != "XBTUSD"]
    total_units=max(1, len(symbols) * len(RS_WINDOWS_H))
    completed=0

    for symbol in symbols:
        x=prepared[symbol]
        try:
            for wh in RS_WINDOWS_H:
                busy={}
                hold=_rs_hold_bars(wh)
                for i in range(max(220, wh + 5), len(x)-2):
                    r=x.iloc[i]
                    ts=int(r["time"])
                    ar=float(r.get("ret"+str(wh), np.nan))
                    br=btc_ret_at(ts, wh)
                    atr=float(r.get("atr14", np.nan))
                    if not (np.isfinite(ar) and np.isfinite(br) and np.isfinite(atr) and atr > 0):
                        continue
                    spread=ar-br
                    reg=regime_at(ts)
                    for scale in RS_THRESHOLD_SCALES:
                        thr=_rs_threshold(wh, scale)
                        direction=None
                        if spread >= thr and r["ema20"] > r["ema50"] and 48 <= r["rsi14"] <= 78:
                            direction="long"
                        elif spread <= -thr and r["ema20"] < r["ema50"] and 22 <= r["rsi14"] <= 52:
                            direction="short"
                        if direction is None:
                            continue
                        for regime_filter in ("all", "aligned"):
                            if regime_filter == "aligned":
                                if direction == "long" and reg == "bear":
                                    continue
                                if direction == "short" and reg == "bull":
                                    continue
                        for sm,tm in RS_RISK_PROFILES:
                            sid=(f"rs_{wh}h_{'strict' if scale>1 else 'base'}_"
                                 f"{regime_filter}_{direction}_S{sm}_T{tm}")
                            for cost_name,cost in LAB_COST_PROFILES.items():
                                key=(sid,cost_name)
                                if i < busy.get(key,-1):
                                    continue
                                tr=edge_trade_row(x,symbol,60,i,direction,sid,"relative_strength_specialist",reg,sm,tm,hold,cost_name,cost)
                                if tr:
                                    tr["window_h"]=wh
                                    tr["threshold_pct"]=round(thr*100,4)
                                    tr["regime_filter"]=regime_filter
                                    tr["spread_at_signal"]=float(spread)
                                    rows.append(tr)
                                    busy[key]=i+hold
                completed += 1
                if completed == 1 or completed % 8 == 0 or completed == total_units:
                    DBX.set("rs_specialist", {
                        "state":"running", "started":iso(), "completed_units":completed,
                        "total_units":total_units, "current":f"{symbol}/{wh}h",
                        "trade_rows_so_far":len(rows), "deep_markets":len(frames),
                        "target_markets":len(RS_MARKETS), "seconds":round(time.time()-started,1),
                        "errors":errors[-6:],
                    })
        except Exception as e:
            errors.append(f"{symbol}: {e}")

    valid=[r for r in rows if r.get("r") is not None and np.isfinite(r["r"])]
    groups={}
    for r in valid:
        key=(r["strategy_id"],r["cost_profile"])
        groups.setdefault(key,[]).append(r)
    ranked=[]
    for (sid,cost_name),rr in groups.items():
        sm=summarize_research(rr)
        if sm["trades"] < 12:
            continue
        sample=rr[0]
        row={
            "strategy_id":sid,"direction":sample["direction"],"window_h":sample["window_h"],
            "regime_filter":sample["regime_filter"],"threshold_pct":sample["threshold_pct"],
            "cost_profile":cost_name,**sm,
        }
        row["robust_score"]=round((row.get("avg_r") or -9)*math.sqrt(max(1,row["trades"])) - 0.012*(row.get("max_drawdown_r") or 0),4)
        ranked.append(row)
    ranked.sort(key=lambda z:z["robust_score"],reverse=True)

    current=[r for r in valid if r["cost_profile"]==EDGE_CURRENT_COST]
    if current:
        lo=min(r["entry_ts"] for r in current); hi=max(r["entry_ts"] for r in current); span=max(1,hi-lo)
    else:
        lo=hi=0; span=1

    fold_bounds=((0.30,0.44),(0.44,0.58),(0.58,0.72),(0.72,0.86),(0.86,1.001))
    folds=[]; selected_oos=[]
    for fold_idx,(a,b) in enumerate(fold_bounds,1):
        st=lo+int(span*a); en=lo+int(span*b)
        train=[r for r in current if r["entry_ts"] < st]
        test=[r for r in current if st <= r["entry_ts"] < en]
        tg={}
        for r in train:
            tg.setdefault(r["strategy_id"],[]).append(r)
        candidates=[]
        for sid,rr in tg.items():
            sm=summarize_research(rr)
            if sm["trades"] < RS_MIN_TRAIN_TRADES:
                continue
            if (sm.get("avg_r") or -99) <= 0 or (sm.get("profit_factor_r") or 0) <= 1.08:
                continue
            first=rr[0]
            score=(sm["avg_r"] or 0)*math.sqrt(sm["trades"])-0.012*(sm.get("max_drawdown_r") or 0)
            candidates.append((score,sid,first["direction"],first["window_h"],sm))
        candidates.sort(reverse=True,key=lambda z:z[0])

        # At most one long and one short rule; this reduces correlated parameter stacking.
        chosen=[]; used_dir=set()
        for item in candidates:
            if item[2] in used_dir:
                continue
            chosen.append(item); used_dir.add(item[2])
            if len(chosen)>=2:
                break
        keys={x[1] for x in chosen}
        sel=[r for r in test if r["strategy_id"] in keys]
        selected_oos.extend(sel)
        sm=summarize_research(sel)
        folds.append({
            "fold":fold_idx,"train_rows":len(train),"eligible_candidates":len(candidates),
            "selected":"; ".join(x[1] for x in chosen) or None,
            "test_trades":sm["trades"],"avg_r":sm["avg_r"],"profit_factor_r":sm["profit_factor_r"],
            "total_r":sm["total_r"],
        })

    oos=summarize_research(selected_oos)
    profitable=sum(1 for f in folds if f.get("test_trades",0)>0 and (f.get("total_r") or 0)>0)
    wfo={
        "state":"done","folds":folds,"test_trades":oos["trades"],"avg_r":oos["avg_r"],
        "profit_factor_r":oos["profit_factor_r"],"total_r":oos["total_r"],
        "max_drawdown_r":oos["max_drawdown_r"],"profitable_folds":profitable,"fold_count":5,
    }

    current_ranked=[r for r in ranked if r["cost_profile"]==EDGE_CURRENT_COST]
    best=current_ranked[0] if current_ranked else None
    checks={
        "deep_markets_at_least_minimum":len(frames)>=RS_MIN_DEEP_MARKETS,
        "current_cost_candidate_80_trades":bool(best and best.get("trades",0)>=80),
        "current_cost_candidate_pf_gt_1_10":bool(best and (best.get("profit_factor_r") or 0)>1.10),
        "current_cost_candidate_avg_r_positive":bool(best and (best.get("avg_r") or -99)>0),
        "oos_at_least_80_trades":wfo.get("test_trades",0)>=80,
        "oos_avg_r_positive":(wfo.get("avg_r") or -99)>0,
        "oos_pf_gt_1_05":(wfo.get("profit_factor_r") or 0)>1.05,
        "profitable_folds_at_least_3_of_5":profitable>=3,
    }
    gate="RS_EDGE_CANDIDATE" if all(checks.values()) else "NO_ROBUST_RS_EDGE_YET"

    best_by_cost=[]
    for cn in LAB_COST_PROFILES:
        rr=[g for g in ranked if g["cost_profile"]==cn]
        if rr:
            best_by_cost.append(rr[0])
    by_window=[]
    for wh in RS_WINDOWS_H:
        rr=[r for r in current if r.get("window_h")==wh]
        sm=summarize_research(rr)
        by_window.append({"window_h":wh,**sm})
    by_direction=[]
    for d in ("long","short"):
        rr=[r for r in current if r.get("direction")==d]
        sm=summarize_research(rr)
        by_direction.append({"direction":d,**sm})

    meta=source_meta or {}
    result={
        "state":"done","finished":iso(),"seconds":round(time.time()-started,2),
        "scope":f"{RS_LOOKBACK_DAYS}d · 60m · up to {len(RS_MARKETS)} markets · 12/24/48/72h relative strength/weakness · 5-fold walk-forward · report-only",
        "deep_markets":len(frames),"target_markets":len(RS_MARKETS),
        "deep_coverage_pct":round(100*len(frames)/max(1,len(RS_MARKETS)),1),
        "data_sources":meta,"trade_rows":len(valid),"candidate_count":len(ranked),
        "evidence_gate":gate,"evidence_checks":checks,"top_candidates":ranked[:20],
        "best_by_cost":best_by_cost,"by_window":by_window,"by_direction":by_direction,
        "walk_forward":wfo,
        "note":"Specialist research only. No live strategy/risk change. Deep-only markets are used; shallow Kraken fallback is deliberately excluded from the 180d specialist test.",
        "errors":errors[-12:],
    }
    DBX.set("rs_specialist",result)
    return result


# -------------------- v1.8.1 Multi-Regime / Multi-Family Architecture (research-only) --------------------
# Methodology change versus earlier labs:
#   1) build signals and raw stop/target/time exits ONCE without transaction costs;
#   2) clone every raw trade across all cost scenarios;
#   3) evaluate cost sensitivity on identical trade IDs;
#   4) use only the conservative current-cost + economically-qualified rows for WFO selection;
#   5) select at most one rule per family inside each training fold.
# This prevents cheaper costs from silently creating a different set of entries.

def _arch_prepare(df: pd.DataFrame) -> pd.DataFrame:
    x = indicators(df).reset_index(drop=True)
    x["ret12"] = x["close"].pct_change(12)
    x["ret24"] = x["close"].pct_change(24)
    x["ret48"] = x["close"].pct_change(48)
    x["ret72"] = x["close"].pct_change(72)
    x["atr_pct"] = x["atr14"] / x["close"].replace(0, np.nan)
    x["atr_pct_mean24"] = x["atr_pct"].shift(1).rolling(24).mean()
    x["atr_pct_mean72"] = x["atr_pct"].shift(1).rolling(72).mean()
    x["vol_ratio"] = x["volume"] / x["prior_volmed20"].replace(0, np.nan)
    return x


def _arch_raw_exit(x: pd.DataFrame, entry_i: int, direction: str, stop_raw: float,
                   target_raw: float, hold_bars: int):
    end = min(len(x) - 1, entry_i + max(1, int(hold_bars)))
    for j in range(entry_i, end + 1):
        b = x.iloc[j]
        if direction == "long":
            if float(b["low"]) <= stop_raw:
                return j, float(stop_raw), "STOP"
            if float(b["high"]) >= target_raw:
                return j, float(target_raw), "TARGET"
        else:
            if float(b["high"]) >= stop_raw:
                return j, float(stop_raw), "STOP"
            if float(b["low"]) <= target_raw:
                return j, float(target_raw), "TARGET"
    return end, float(x.iloc[end]["close"]), "TIME"


def _arch_raw_trade(x: pd.DataFrame, symbol: str, signal_i: int, direction: str,
                    strategy_id: str, family: str, regime: str, stop_mult: float,
                    target_mult: float, hold_bars: int, extra: dict | None = None):
    entry_i = signal_i + 1
    if entry_i >= len(x):
        return None
    atr = float(x.iloc[signal_i].get("atr14", np.nan))
    entry_raw = float(x.iloc[entry_i]["open"])
    if not (np.isfinite(atr) and atr > 0 and np.isfinite(entry_raw) and entry_raw > 0):
        return None
    # Structural levels are deliberately cost-independent.
    if direction == "long":
        stop_raw = entry_raw - stop_mult * atr
        target_raw = entry_raw + target_mult * atr
    else:
        stop_raw = entry_raw + stop_mult * atr
        target_raw = entry_raw - target_mult * atr
    if stop_raw <= 0 or target_raw <= 0:
        return None
    risk_raw = abs(entry_raw - stop_raw)
    if risk_raw <= 0:
        return None
    risk_bps = (risk_raw / entry_raw) * 10000.0
    # Reject structurally tiny denominators instead of allowing pathological R-multiples.
    if not np.isfinite(risk_bps) or risk_bps < ARCH_MIN_RISK_BPS:
        return None
    exit_i, exit_raw, reason = _arch_raw_exit(x, entry_i, direction, stop_raw, target_raw, hold_bars)
    entry_ts = int(x.iloc[entry_i]["time"])
    raw = {
        "trade_id": f"{strategy_id}|{symbol}|{entry_ts}|{direction}",
        "symbol": symbol, "timeframe": 60, "strategy_id": strategy_id,
        "family": family, "direction": direction, "regime": regime,
        "entry_ts": entry_ts, "exit_ts": int(x.iloc[exit_i]["time"]),
        "entry_i": int(entry_i), "exit_i": int(exit_i),
        "entry_raw": float(entry_raw), "exit_raw": float(exit_raw),
        "stop_raw": float(stop_raw), "target_raw": float(target_raw),
        "risk_raw": float(risk_raw), "risk_bps": float(risk_bps), "reason": reason,
        "held_hours": float(max(0, exit_i - entry_i + 1)),
        "stop_mult": float(stop_mult), "target_mult": float(target_mult),
    }
    if extra:
        raw.update(extra)
    return raw


def _arch_apply_cost(raw: dict, cost_name: str, cost: dict):
    """Apply execution costs AFTER the raw trade is fixed.

    v1.8.1 supports separate entry/target/stop/time fee and slippage assumptions.
    This keeps trade IDs identical across cost models while allowing a venue-aware
    maker/taker overlay. It also rejects non-finite/pathological R values instead
    of letting tiny denominators contaminate rankings.
    """
    def bps(key: str, fallback_key: str | None = None, default: float = 0.0) -> float:
        if key in cost:
            return float(cost[key]) / 10000.0
        if fallback_key and fallback_key in cost:
            return float(cost[fallback_key]) / 10000.0
        return float(default) / 10000.0

    direction = raw["direction"]
    reason = str(raw.get("reason") or "TIME").upper()
    e0, x0 = float(raw["entry_raw"]), float(raw["exit_raw"])

    entry_fee_rate = bps("entry_fee_bps", "fee_bps")
    entry_slip = bps("entry_slippage_bps", "slippage_bps")

    if reason.startswith("TARGET"):
        exit_fee_rate = bps("target_exit_fee_bps", "fee_bps")
        exit_slip = bps("target_exit_slippage_bps", "slippage_bps")
    elif reason.startswith("STOP"):
        exit_fee_rate = bps("stop_exit_fee_bps", "fee_bps")
        exit_slip = bps("stop_exit_slippage_bps", "slippage_bps")
    else:
        exit_fee_rate = bps("time_exit_fee_bps", "fee_bps")
        exit_slip = bps("time_exit_slippage_bps", "slippage_bps")

    # Entry economics are always judged against the target-exit assumptions.
    target_fee_rate = bps("target_exit_fee_bps", "fee_bps")
    target_slip = bps("target_exit_slippage_bps", "slippage_bps")
    fund = float(cost.get("funding_bps_8h", 0.0)) / 10000.0

    if direction == "long":
        entry = e0 * (1 + entry_slip)
        exit_px = x0 * (1 - exit_slip)
        target_exec = float(raw["target_raw"]) * (1 - target_slip)
        gross = exit_px - entry
        target_gross = target_exec - entry
    else:
        entry = e0 * (1 - entry_slip)
        exit_px = x0 * (1 + exit_slip)
        target_exec = float(raw["target_raw"]) * (1 + target_slip)
        gross = entry - exit_px
        target_gross = entry - target_exec

    entry_fee = entry * entry_fee_rate
    exit_fee = exit_px * exit_fee_rate
    target_exit_fee = target_exec * target_fee_rate
    funding = entry * fund * (float(raw["held_hours"]) / 8.0)
    net = gross - entry_fee - exit_fee - funding

    # Fixed structural denominator is identical across all cost scenarios.
    risk = float(raw["risk_raw"])
    r_mult = net / risk if risk > 0 else None
    gross_raw = (x0 - e0) if direction == "long" else (e0 - x0)
    gross_r = gross_raw / risk if risk > 0 else None

    target_net = target_gross - entry_fee - target_exit_fee
    net_rr = target_net / risk if risk > 0 else None
    target_net_bps = (target_net / entry) * 10000.0 if entry > 0 else None

    sanity_rejected = bool(
        r_mult is None or not np.isfinite(r_mult) or
        gross_r is None or not np.isfinite(gross_r) or
        abs(float(r_mult)) > ARCH_MAX_ABS_R or
        abs(float(gross_r)) > ARCH_MAX_ABS_R
    )

    row = {k: v for k, v in raw.items() if k not in {"entry_i", "exit_i"}}
    row.update({
        "cost_profile": cost_name,
        "r": None if sanity_rejected else float(r_mult),
        "gross_r": None if sanity_rejected else float(gross_r),
        "fees_per_unit": float(entry_fee + exit_fee),
        "funding_per_unit": float(funding),
        "entry_fee_bps": round(entry_fee_rate * 10000.0, 4),
        "exit_fee_bps": round(exit_fee_rate * 10000.0, 4),
        "entry_slippage_bps": round(entry_slip * 10000.0, 4),
        "exit_slippage_bps": round(exit_slip * 10000.0, 4),
        "net_rr_at_entry": float(net_rr) if net_rr is not None and np.isfinite(net_rr) else None,
        "target_net_bps": float(target_net_bps) if target_net_bps is not None and np.isfinite(target_net_bps) else None,
        "sanity_rejected": sanity_rejected,
        "economically_qualified": bool(
            not sanity_rejected and
            net_rr is not None and target_net_bps is not None and
            np.isfinite(net_rr) and np.isfinite(target_net_bps) and
            net_rr >= CFG.min_net_rr and target_net_bps >= CFG.min_net_target_bps
        ),
    })
    return row


def _arch_expand_costs(raw_trades: list[dict]):
    rows = []
    for tr in raw_trades:
        for cost_name, cost in ARCH_COST_PROFILES.items():
            rows.append(_arch_apply_cost(tr, cost_name, cost))
    return rows


def _arch_cost_pair_audit(rows: list[dict]):
    ids = {}
    for cn in ARCH_COST_PROFILES:
        ids[cn] = {r["trade_id"] for r in rows if r.get("cost_profile") == cn}
    names = list(ARCH_COST_PROFILES)
    base = ids[names[0]] if names else set()
    exact = all(ids[n] == base for n in names)
    return {
        "exact_same_entries": bool(exact),
        "unique_raw_trades": len(base),
        "rows_by_cost": {n: len(ids[n]) for n in names},
        "note": "Cost scenarios are applied after signal/exit generation; entry IDs are identical across costs.",
    }


def _arch_group_summary(rows: list[dict], key_fields: tuple[str, ...], min_trades: int = 1):
    groups = {}
    for r in rows:
        key = tuple(r.get(k) for k in key_fields)
        groups.setdefault(key, []).append(r)
    out = []
    for key, rr in groups.items():
        sm = summarize_research(rr)
        if sm["trades"] < min_trades:
            continue
        rec = {k: v for k, v in zip(key_fields, key)}
        rec.update(sm)
        rec["qualified_pct"] = round(100 * sum(bool(x.get("economically_qualified")) for x in rr) / max(1, len(rr)), 1)
        out.append(rec)
    out.sort(key=lambda z: (z.get("avg_r") if z.get("avg_r") is not None else -999), reverse=True)
    return out


def multi_family_architecture_from_frames(frames: dict[str, pd.DataFrame], source_meta: dict | None = None):
    started = time.time()
    DBX.set("architecture", {
        "state": "running", "started": iso(),
        "note": "Generating cost-independent raw trades across independent strategy families.",
    })
    if "XBTUSD" not in frames:
        raise RuntimeError("BTC 60m deep history unavailable")
    if len(frames) < ARCH_MIN_DEEP_MARKETS:
        raise RuntimeError(f"insufficient deep-market coverage: {len(frames)}/{len(RS_MARKETS)}")

    prepared = {sym: _arch_prepare(df) for sym, df in frames.items()}
    regime_df = edge_regime_frame(frames["XBTUSD"])
    reg_times = regime_df["time"].to_numpy(dtype=np.int64)
    reg_vals = regime_df["regime"].to_numpy()
    btc = prepared["XBTUSD"]
    btc_times = btc["time"].to_numpy(dtype=np.int64)
    btc_ret24 = btc["ret24"].to_numpy(dtype=float)
    btc_ret72 = btc["ret72"].to_numpy(dtype=float)

    def regime_at(ts: int):
        k = _asof_index(reg_times, ts)
        return str(reg_vals[k]) if k >= 0 else "unknown"

    def btc_ret_at(ts: int, wh: int):
        k = _asof_index(btc_times, ts)
        arr = btc_ret24 if wh == 24 else btc_ret72
        return float(arr[k]) if k >= 0 and np.isfinite(arr[k]) else np.nan

    raw_trades = []
    errors = []
    symbols = [s for s in prepared if s != "XBTUSD"]
    total_units = max(1, len(symbols) + 1)
    completed = 0

    # Families 1-4: trend breakout, volatility expansion, regime mean-reversion,
    # and relative strength/weakness. All use bar i information and enter next open.
    for symbol in symbols:
        x = prepared[symbol]
        busy = {}
        try:
            for i in range(220, len(x) - 2):
                r = x.iloc[i]
                ts = int(r["time"])
                reg = regime_at(ts)
                atr_pct = float(r.get("atr_pct", np.nan))
                if not np.isfinite(atr_pct) or atr_pct < ARCH_MIN_ATR_PCT:
                    continue
                specs = []

                # 1) Trend breakout: wide, persistent trend with liquidity confirmation.
                if (reg != "bear" and r["ema20"] > r["ema50"] > r["ema200"] and
                    np.isfinite(r.get("prior_high40", np.nan)) and r["close"] > r["prior_high40"] and
                    float(r.get("vol_ratio", 0.0)) >= 1.0):
                    specs.append(("arch_trend_breakout_long", "trend_breakout", "long", 1.6, 3.2, 36, {}))
                if (reg != "bull" and r["ema20"] < r["ema50"] < r["ema200"] and
                    np.isfinite(r.get("prior_low40", np.nan)) and r["close"] < r["prior_low40"] and
                    float(r.get("vol_ratio", 0.0)) >= 1.0):
                    specs.append(("arch_trend_breakout_short", "trend_breakout", "short", 1.6, 3.2, 36, {}))

                # 2) Volatility expansion after compression, not just another lookback tweak.
                m24 = float(r.get("atr_pct_mean24", np.nan)); m72 = float(r.get("atr_pct_mean72", np.nan))
                compressed = np.isfinite(m24) and np.isfinite(m72) and m24 < 0.82 * m72
                if compressed and atr_pct > 1.12 * m24:
                    if (reg != "bear" and r["close"] > r["prior_high20"] and r["ema20"] > r["ema50"]):
                        specs.append(("arch_vol_expansion_long", "volatility_expansion", "long", 1.5, 3.0, 24, {}))
                    if (reg != "bull" and r["close"] < r["prior_low20"] and r["ema20"] < r["ema50"]):
                        specs.append(("arch_vol_expansion_short", "volatility_expansion", "short", 1.5, 3.0, 24, {}))

                # 3) Mean reversion is allowed only in a BTC sideways regime.
                if reg == "sideways":
                    if np.isfinite(r.get("bb_lower", np.nan)) and r["close"] < r["bb_lower"] and r["rsi14"] < 28:
                        specs.append(("arch_sideways_meanrev_long", "sideways_meanrev", "long", 1.3, 2.4, 12, {}))
                    if np.isfinite(r.get("bb_upper", np.nan)) and r["close"] > r["bb_upper"] and r["rsi14"] > 72:
                        specs.append(("arch_sideways_meanrev_short", "sideways_meanrev", "short", 1.3, 2.4, 12, {}))

                # 4) Relative strength / weakness retained as one family, not the whole search.
                for wh, spread_thr in ((24, 0.03), (72, 0.055)):
                    ar = float(r.get(f"ret{wh}", np.nan)); br = btc_ret_at(ts, wh)
                    if not (np.isfinite(ar) and np.isfinite(br)):
                        continue
                    spread = ar - br
                    if spread >= spread_thr and reg != "bear" and r["ema20"] > r["ema50"] and 48 <= r["rsi14"] <= 76:
                        specs.append((f"arch_rs_{wh}h_long", "relative_strength", "long", 1.6, 3.0, min(36, wh), {"spread": float(spread), "window_h": wh}))
                    if spread <= -spread_thr and reg != "bull" and r["ema20"] < r["ema50"] and 24 <= r["rsi14"] <= 52:
                        specs.append((f"arch_rs_{wh}h_short", "relative_strength", "short", 1.6, 3.0, min(36, wh), {"spread": float(spread), "window_h": wh}))

                for sid, fam, direction, sm, tm, hold, extra in specs:
                    key = (sid, direction)
                    if i < busy.get(key, -1):
                        continue
                    tr = _arch_raw_trade(x, symbol, i, direction, sid, fam, reg, sm, tm, hold, extra)
                    if tr:
                        raw_trades.append(tr)
                        busy[key] = int(tr["exit_i"]) + 1
        except Exception as e:
            errors.append(f"{symbol}: {e}")
        completed += 1
        if completed == 1 or completed % 4 == 0:
            DBX.set("architecture", {
                "state": "running", "started": iso(), "completed_units": completed,
                "total_units": total_units, "current": symbol,
                "raw_trades_so_far": len(raw_trades), "seconds": round(time.time()-started,1),
                "errors": errors[-6:],
            })

    # Family 5: cross-sectional momentum. Rank the cross-section, then enter next bar.
    try:
        xs = {sym: x.set_index("time") for sym, x in prepared.items() if sym != "XBTUSD"}
        common = sorted(set.intersection(*[set(x.index) for x in xs.values()])) if xs else []
        busy = {}
        for ts in common:
            vals = []
            for sym, xidx in xs.items():
                try:
                    r24 = float(xidx.loc[ts, "ret24"])
                    r72 = float(xidx.loc[ts, "ret72"])
                    if np.isfinite(r24) and np.isfinite(r72):
                        vals.append((sym, 0.6*r24 + 0.4*r72))
                except Exception:
                    pass
            if len(vals) < 10:
                continue
            vals.sort(key=lambda z: z[1])
            if vals[-1][1] - vals[0][1] < 0.06:
                continue
            reg = regime_at(int(ts))
            picks = [(vals[-1][0], "long"), (vals[0][0], "short")]
            for sym, direction in picks:
                if reg == "bull" and direction == "short":
                    continue
                if reg == "bear" and direction == "long":
                    continue
                x = prepared[sym]
                i = _asof_index(x["time"].to_numpy(dtype=np.int64), int(ts))
                if i < 220 or i >= len(x)-2:
                    continue
                # v1.8.1: cross-sectional trades must pass the same volatility
                # floor as every other architecture family.
                atr_pct = float(x.iloc[i].get("atr_pct", np.nan))
                if not np.isfinite(atr_pct) or atr_pct < ARCH_MIN_ATR_PCT:
                    continue
                sid = f"arch_xsec_momentum_{direction}"
                key = (sym, sid)
                if i < busy.get(key, -1):
                    continue
                tr = _arch_raw_trade(x, sym, i, direction, sid, "cross_sectional", reg, 1.6, 3.0, 18,
                                     {"xsec_score": float(vals[-1][1] if direction == "long" else vals[0][1])})
                if tr:
                    raw_trades.append(tr)
                    busy[key] = int(tr["exit_i"]) + 1
    except Exception as e:
        errors.append(f"cross_sectional: {e}")

    # De-duplicate before cost expansion. This makes the pairing audit meaningful.
    dedup = {}
    for tr in raw_trades:
        dedup[tr["trade_id"]] = tr
    raw_trades = list(dedup.values())
    rows = _arch_expand_costs(raw_trades)
    pairing = _arch_cost_pair_audit(rows)
    valid = [r for r in rows if r.get("r") is not None and np.isfinite(r["r"]) and not r.get("sanity_rejected")]
    sanity_rejected_rows = len(rows) - len(valid)

    # Apples-to-apples cost comparison: same raw trade IDs under each architecture cost profile.
    paired_groups = _arch_group_summary(valid, ("strategy_id", "family", "direction", "cost_profile"), min_trades=12)
    current_all = [r for r in valid if r["cost_profile"] == ARCH_CURRENT_COST]
    current_exec = [r for r in current_all if r.get("economically_qualified")]
    exec_groups = _arch_group_summary(current_exec, ("strategy_id", "family", "direction"), min_trades=12)
    for row in exec_groups:
        row["robust_score"] = round((row.get("avg_r") or -9)*math.sqrt(max(1,row["trades"])) - 0.012*(row.get("max_drawdown_r") or 0),4)
    exec_groups.sort(key=lambda z:z.get("robust_score",-999), reverse=True)

    top_ids = [x["strategy_id"] for x in exec_groups[:6]]
    same_entry_cost_comparison = [x for x in paired_groups if x["strategy_id"] in top_ids]
    same_entry_cost_comparison.sort(key=lambda z:(top_ids.index(z["strategy_id"]) if z["strategy_id"] in top_ids else 999,
                                                  list(ARCH_COST_PROFILES).index(z["cost_profile"])))

    # 5-fold outer WFO. Selection sees current-cost training data only.
    current = current_exec
    if current:
        lo=min(r["entry_ts"] for r in current); hi=max(r["entry_ts"] for r in current); span=max(1,hi-lo)
    else:
        lo=hi=0; span=1
    folds=[]; selected_oos=[]
    family_fold_results=[]
    for fold_idx,(a,b) in enumerate(ARCH_FOLD_BOUNDS,1):
        st=lo+int(span*a); en=lo+int(span*b)
        train=[r for r in current if r["entry_ts"] < st]
        test=[r for r in current if st <= r["entry_ts"] < en]
        tg={}
        for r in train:
            tg.setdefault(r["strategy_id"],[]).append(r)
        cand=[]
        for sid,rr in tg.items():
            sm=summarize_research(rr)
            if sm["trades"] < ARCH_MIN_TRAIN_TRADES:
                continue
            if (sm.get("avg_r") or -99) <= 0 or (sm.get("profit_factor_r") or 0) <= 1.05:
                continue
            fam=rr[0]["family"]; direction=rr[0]["direction"]
            score=(sm["avg_r"] or 0)*math.sqrt(sm["trades"])-0.012*(sm.get("max_drawdown_r") or 0)
            cand.append((score,sid,fam,direction,sm))
        cand.sort(reverse=True,key=lambda z:z[0])
        chosen=[]; used_families=set()
        for item in cand:
            if item[2] in used_families:
                continue
            chosen.append(item); used_families.add(item[2])
            if len(chosen)>=ARCH_MAX_FAMILIES_PER_FOLD:
                break
        keys={x[1] for x in chosen}
        sel=[r for r in test if r["strategy_id"] in keys]
        selected_oos.extend(sel)
        sm=summarize_research(sel)
        folds.append({
            "fold":fold_idx,"train_rows":len(train),"eligible_candidates":len(cand),
            "selected":"; ".join(f"{x[1]} [{x[2]}]" for x in chosen) or None,
            "test_trades":sm["trades"],"avg_r":sm["avg_r"],"profit_factor_r":sm["profit_factor_r"],"total_r":sm["total_r"],
        })
        for fam in sorted(used_families):
            fs=[r for r in sel if r["family"]==fam]
            fsm=summarize_research(fs)
            family_fold_results.append({"fold":fold_idx,"family":fam,**fsm})

    oos=summarize_research(selected_oos)
    profitable=sum(1 for f in folds if f.get("test_trades",0)>0 and (f.get("total_r") or 0)>0)
    oos_by_family=[]; survivors=[]
    for fam in sorted({r["family"] for r in selected_oos}):
        rr=[r for r in selected_oos if r["family"]==fam]
        sm=summarize_research(rr)
        folds_positive=sum(1 for f in family_fold_results if f["family"]==fam and f.get("trades",0)>0 and (f.get("total_r") or 0)>0)
        row={"family":fam,**sm,"profitable_folds":folds_positive}
        oos_by_family.append(row)
        if sm["trades"]>=20 and (sm.get("avg_r") or -99)>0 and (sm.get("profit_factor_r") or 0)>1.05 and folds_positive>=2:
            survivors.append(fam)
    oos_by_family.sort(key=lambda z:(z.get("avg_r") if z.get("avg_r") is not None else -999),reverse=True)
    wfo={"state":"done","folds":folds,"test_trades":oos["trades"],"avg_r":oos["avg_r"],
         "profit_factor_r":oos["profit_factor_r"],"total_r":oos["total_r"],"max_drawdown_r":oos["max_drawdown_r"],
         "profitable_folds":profitable,"fold_count":5}

    best = exec_groups[0] if exec_groups else None
    checks={
        "cost_pairing_exact": bool(pairing.get("exact_same_entries")),
        "sanity_rejected_cost_rows_zero": sanity_rejected_rows == 0,
        "deep_markets_at_least_minimum": len(frames)>=ARCH_MIN_DEEP_MARKETS,
        "at_least_four_families_tested": len({r["family"] for r in raw_trades})>=4,
        "current_cost_candidate_60_trades": bool(best and best.get("trades",0)>=60),
        "current_cost_candidate_pf_gt_1_10": bool(best and (best.get("profit_factor_r") or 0)>1.10),
        "current_cost_candidate_avg_r_positive": bool(best and (best.get("avg_r") or -99)>0),
        "oos_at_least_80_trades": wfo.get("test_trades",0)>=80,
        "oos_avg_r_positive": (wfo.get("avg_r") or -99)>0,
        "oos_pf_gt_1_05": (wfo.get("profit_factor_r") or 0)>1.05,
        "profitable_folds_at_least_3_of_5": profitable>=3,
        "at_least_two_oos_survivor_families": len(survivors)>=2,
    }
    gate="ARCH_EDGE_CANDIDATE" if all(checks.values()) else "NO_ROBUST_MULTI_FAMILY_EDGE_YET"

    by_family=_arch_group_summary(current_exec,("family",),min_trades=1)
    by_regime=_arch_group_summary(current_exec,("regime",),min_trades=1)
    meta=source_meta or {}
    result={
        "state":"done","finished":iso(),"seconds":round(time.time()-started,2),
        "scope":f"{RS_LOOKBACK_DAYS}d deep 60m · up to {len(RS_MARKETS)} markets · 5 independent families · same-entry venue-aware cost stress · 5-fold outer WFO · report-only",
        "deep_markets":len(frames),"target_markets":len(RS_MARKETS),
        "deep_coverage_pct":round(100*len(frames)/max(1,len(RS_MARKETS)),1),"data_sources":meta,
        "raw_trades":len(raw_trades),"paired_cost_rows":len(rows),"valid_cost_rows":len(valid),
        "sanity_rejected_cost_rows":sanity_rejected_rows,
        "current_cost_profile":ARCH_CURRENT_COST,"current_cost_qualified_rows":len(current_exec),
        "candidate_count":len(exec_groups),"cost_pairing_audit":pairing,
        "evidence_gate":gate,"evidence_checks":checks,
        "top_current_candidates":exec_groups[:20],
        "same_entry_cost_comparison":same_entry_cost_comparison,
        "by_family":by_family,"by_regime":by_regime,
        "walk_forward":wfo,"oos_by_family":oos_by_family,"oos_survivor_families":survivors,
        "note":"v1.8.1: all architecture cost scenarios use identical raw signals/entries/exits; pathological R rows are rejected. The evidence gate uses the Kraken Tier-1 all-taker reference profile. Maker-target and Tier-3 profiles are diagnostics only; final venue/execution assumptions are not yet locked.",
        "errors":errors[-12:],
    }
    DBX.set("architecture",result)
    return result


# -------------------- v1.9 Low-Turnover Execution Edge (research-only) --------------------
def _exec_resample(df60: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Aggregate completed 60m candles into fixed UTC N-hour bars without look-ahead."""
    if hours not in EXEC_BAR_HOURS:
        raise ValueError(f"unsupported execution research bar: {hours}h")
    x = df60[["time","open","high","low","close","volume"]].copy()
    x = x.dropna().sort_values("time").drop_duplicates("time")
    bucket_s = int(hours) * 3600
    x["bucket"] = (x["time"].astype("int64") // bucket_s) * bucket_s
    g = x.groupby("bucket", sort=True)
    y = g.agg(open=("open","first"), high=("high","max"), low=("low","min"),
              close=("close","last"), volume=("volume","sum"), count=("time","count")).reset_index()
    y = y[y["count"] >= int(hours)].rename(columns={"bucket":"time"})
    return y[["time","open","high","low","close","volume"]].reset_index(drop=True)


def _exec_prepare(df60: pd.DataFrame, hours: int) -> pd.DataFrame:
    x = indicators(_exec_resample(df60, hours)).reset_index(drop=True)
    x["atr_pct"] = x["atr14"] / x["close"].replace(0, np.nan)
    x["vol_ratio"] = x["volume"] / x["prior_volmed20"].replace(0, np.nan)
    x["ret10"] = x["close"].pct_change(10)
    x["ret20"] = x["close"].pct_change(20)
    x["ret40"] = x["close"].pct_change(40)
    return x


def _exec_raw_exit(x: pd.DataFrame, entry_i: int, direction: str, stop_raw: float,
                   target_raw: float, hold_bars: int):
    end = min(len(x)-1, entry_i + max(1, int(hold_bars)))
    for j in range(entry_i, end+1):
        b=x.iloc[j]
        if direction == "long":
            if float(b["low"]) <= stop_raw:
                return j, float(stop_raw), "STOP"
            if float(b["high"]) >= target_raw:
                return j, float(target_raw), "TARGET"
        else:
            if float(b["high"]) >= stop_raw:
                return j, float(stop_raw), "STOP"
            if float(b["low"]) <= target_raw:
                return j, float(target_raw), "TARGET"
    return end, float(x.iloc[end]["close"]), "TIME"


def _exec_raw_trade(x: pd.DataFrame, symbol: str, signal_i: int, hours: int, direction: str,
                    strategy_id: str, family: str, stop_mult: float, target_mult: float,
                    hold_bars: int, extra: dict | None = None):
    entry_i=signal_i+1
    if entry_i >= len(x):
        return None
    atr=float(x.iloc[signal_i].get("atr14",np.nan)); entry_raw=float(x.iloc[entry_i]["open"])
    if not (np.isfinite(atr) and atr>0 and np.isfinite(entry_raw) and entry_raw>0):
        return None
    atr_pct=atr/entry_raw
    if not np.isfinite(atr_pct) or atr_pct < 0.004:
        return None
    if direction == "long":
        stop_raw=entry_raw-stop_mult*atr; target_raw=entry_raw+target_mult*atr
    else:
        stop_raw=entry_raw+stop_mult*atr; target_raw=entry_raw-target_mult*atr
    if stop_raw<=0 or target_raw<=0:
        return None
    risk_raw=abs(entry_raw-stop_raw); risk_bps=risk_raw/entry_raw*10000.0
    if not np.isfinite(risk_bps) or risk_bps < EXEC_MIN_RISK_BPS:
        return None
    exit_i,exit_raw,reason=_exec_raw_exit(x,entry_i,direction,stop_raw,target_raw,hold_bars)
    entry_ts=int(x.iloc[entry_i]["time"]); exit_ts=int(x.iloc[exit_i]["time"])
    held_hours=float(max(hours, (exit_i-entry_i+1)*hours))
    row={
        "trade_id":f"{strategy_id}|{symbol}|{entry_ts}|{direction}",
        "symbol":symbol,"bar_hours":int(hours),"strategy_id":strategy_id,"family":family,"direction":direction,
        "entry_ts":entry_ts,"exit_ts":exit_ts,"entry_i":int(entry_i),"exit_i":int(exit_i),
        "entry_raw":float(entry_raw),"exit_raw":float(exit_raw),"stop_raw":float(stop_raw),"target_raw":float(target_raw),
        "risk_raw":float(risk_raw),"risk_bps":float(risk_bps),"reason":reason,"held_hours":held_hours,
        "stop_mult":float(stop_mult),"target_mult":float(target_mult),
    }
    if extra: row.update(extra)
    return row


def _exec_apply_cost(raw: dict, cost_name: str, cost: dict):
    # Reuse the tested v1.8.1 execution overlay, then apply v1.9-specific sanity limits.
    z=_arch_apply_cost(raw,cost_name,cost)
    r=z.get("r"); gr=z.get("gross_r")
    bad=bool(z.get("sanity_rejected") or r is None or gr is None or not np.isfinite(r) or not np.isfinite(gr)
             or abs(float(r))>EXEC_MAX_ABS_R or abs(float(gr))>EXEC_MAX_ABS_R)
    z["sanity_rejected"]=bad
    if bad:
        z["r"]=None; z["gross_r"]=None; z["economically_qualified"]=False
    z["product"]=cost.get("product")
    z["execution_assumption"]=cost.get("assumption")
    return z


def _exec_expand_costs(raw_trades: list[dict]):
    return [_exec_apply_cost(tr,cn,cost) for tr in raw_trades for cn,cost in EXEC_COST_PROFILES.items()]


def _exec_pair_audit(rows: list[dict]):
    ids={cn:{r["trade_id"] for r in rows if r.get("cost_profile")==cn} for cn in EXEC_COST_PROFILES}
    names=list(EXEC_COST_PROFILES); base=ids[names[0]] if names else set()
    return {"exact_same_entries":all(ids[n]==base for n in names),"unique_raw_trades":len(base),
            "rows_by_cost":{n:len(ids[n]) for n in names}}


def _exec_specs(x: pd.DataFrame, i: int, hours: int):
    if i < 220 or i >= len(x)-2:
        return []
    r=x.iloc[i]; p=x.iloc[i-1]
    need=["close","atr14","ema20","ema50","ema200","rsi14","prior_high10","prior_high20","prior_high40",
          "prior_low10","prior_low20","prior_low40","vol_ratio"]
    if not all(np.isfinite(r.get(k,np.nan)) for k in need):
        return []
    atr_pct=float(r["atr14"])/float(r["close"]) if float(r["close"])>0 else 0.0
    if atr_pct < 0.004:
        return []
    vol=float(r.get("vol_ratio",0.0)); out=[]
    strong_up=r["ema20"]>r["ema50"]>r["ema200"]; strong_dn=r["ema20"]<r["ema50"]<r["ema200"]
    trend_up=r["ema20"]>r["ema50"]; trend_dn=r["ema20"]<r["ema50"]

    if hours==4:
        if strong_up and r["close"]>r["prior_high20"] and vol>=0.85:
            out.append(("exec_4h_breakout20_long","slow_breakout","long",2.2,5.5,30))
        if strong_dn and r["close"]<r["prior_low20"] and vol>=0.85:
            out.append(("exec_4h_breakout20_short","slow_breakout","short",2.2,5.5,30))
        if strong_up and r["close"]>r["prior_high40"] and vol>=0.80:
            out.append(("exec_4h_breakout40_long","slow_breakout","long",2.5,6.5,42))
        if strong_dn and r["close"]<r["prior_low40"] and vol>=0.80:
            out.append(("exec_4h_breakout40_short","slow_breakout","short",2.5,6.5,42))
        if trend_up and p["close"]<=p["ema20"] and r["close"]>r["ema20"] and 48<=r["rsi14"]<=64:
            out.append(("exec_4h_pullback_long","slow_pullback","long",2.0,5.0,30))
        if trend_dn and p["close"]>=p["ema20"] and r["close"]<r["ema20"] and 36<=r["rsi14"]<=52:
            out.append(("exec_4h_pullback_short","slow_pullback","short",2.0,5.0,30))
    elif hours==12:
        if strong_up and r["close"]>r["prior_high10"] and vol>=0.80:
            out.append(("exec_12h_breakout10_long","macro_breakout","long",2.2,6.0,14))
        if strong_dn and r["close"]<r["prior_low10"] and vol>=0.80:
            out.append(("exec_12h_breakout10_short","macro_breakout","short",2.2,6.0,14))
        if strong_up and r["close"]>r["prior_high20"]:
            out.append(("exec_12h_breakout20_long","macro_breakout","long",2.5,7.0,20))
        if strong_dn and r["close"]<r["prior_low20"]:
            out.append(("exec_12h_breakout20_short","macro_breakout","short",2.5,7.0,20))
        if trend_up and p["close"]<=p["ema20"] and r["close"]>r["ema20"] and 48<=r["rsi14"]<=62:
            out.append(("exec_12h_pullback_long","macro_pullback","long",2.3,6.0,16))
        if trend_dn and p["close"]>=p["ema20"] and r["close"]<r["ema20"] and 38<=r["rsi14"]<=52:
            out.append(("exec_12h_pullback_short","macro_pullback","short",2.3,6.0,16))
    return out


def _parallel_block_bootstrap(rows: list[dict], iterations: int = 2000):
    """Deterministic calendar-block bootstrap; preserves clustered crypto regimes better than IID sampling."""
    clean=[r for r in rows if r.get("r") is not None and np.isfinite(r["r"])]
    if not clean:
        return {"trades":0,"blocks":0,"iterations":0,"mean_r_p05":None,"mean_r_p50":None,
                "mean_r_p95":None,"prob_mean_r_positive_pct":None}
    block_s=30*86400
    blocks={}
    for r in clean:
        blocks.setdefault(int(r["entry_ts"])//block_s,[]).append(float(r["r"]))
    values=list(blocks.values()); rng=np.random.default_rng(21092026); sims=[]
    n=max(1,len(values))
    for _ in range(max(200,int(iterations))):
        draw=[]
        for j in rng.integers(0,n,size=n): draw.extend(values[int(j)])
        if draw: sims.append(float(np.mean(draw)))
    if not sims:
        return {"trades":len(clean),"blocks":len(values),"iterations":0,"mean_r_p05":None,
                "mean_r_p50":None,"mean_r_p95":None,"prob_mean_r_positive_pct":None}
    q=np.quantile(np.asarray(sims,dtype=float),[0.05,0.50,0.95])
    return {"trades":len(clean),"blocks":len(values),"iterations":len(sims),
            "mean_r_p05":round(float(q[0]),4),"mean_r_p50":round(float(q[1]),4),
            "mean_r_p95":round(float(q[2]),4),
            "prob_mean_r_positive_pct":round(100*sum(x>0 for x in sims)/len(sims),1)}


def parallel_research_lab(primary_rows: list[dict], all_cost_rows: list[dict], deep_markets: int):
    """Fast, report-only robustness checks for the two frozen v1.9 hypotheses.

    This is intentionally independent of the shadow ledger and cannot change live risk or orders.
    It shortens the iteration cycle by running cross-market, chronological and block-bootstrap
    diagnostics immediately while genuine forward observations continue accumulating.
    """
    started=time.time(); frozen=set(SHADOW_STRATEGIES)
    rows=[r for r in primary_rows if r.get("strategy_id") in frozen and r.get("r") is not None]
    cost_rows=[r for r in all_cost_rows if r.get("strategy_id") in frozen and r.get("r") is not None]
    strategies=[]
    for sid in SHADOW_STRATEGIES:
        rr=sorted([r for r in rows if r.get("strategy_id")==sid],key=lambda z:z["entry_ts"])
        overall=summarize_research(rr); by_market=[]
        for sym in sorted({r["symbol"] for r in rr}):
            sm=summarize_research([r for r in rr if r["symbol"]==sym])
            by_market.append({"symbol":sym,**sm})
        positive_markets=sum(1 for x in by_market if x["trades"]>=3 and (x.get("avg_r") or -99)>0)
        eligible_markets=sum(1 for x in by_market if x["trades"]>=3)

        slices=[]
        if rr:
            lo=min(r["entry_ts"] for r in rr); hi=max(r["entry_ts"] for r in rr); span=max(1,hi-lo+1)
            for i in range(6):
                a=lo+int(span*i/6); b=lo+int(span*(i+1)/6)
                part=[r for r in rr if a<=r["entry_ts"]<(b if i<5 else hi+1)]
                sm=summarize_research(part)
                slices.append({"slice":i+1,"start":datetime.fromtimestamp(a,timezone.utc).date().isoformat(),
                               "end":datetime.fromtimestamp(min(b,hi),timezone.utc).date().isoformat(),**sm})
        positive_slices=sum(1 for x in slices if x["trades"]>0 and (x.get("total_r") or 0)>0)
        boot=_parallel_block_bootstrap(rr)

        cost_stability=[]
        for cn in EXEC_COST_PROFILES:
            sm=summarize_research([r for r in cost_rows if r.get("strategy_id")==sid and r.get("cost_profile")==cn])
            cost_stability.append({"cost_profile":cn,**sm})
        rapid_checks={
            "sample_at_least_50":overall.get("trades",0)>=50,
            "primary_avg_r_positive":bool((overall.get("avg_r") or -99)>0),
            "primary_pf_gt_1_10":bool((overall.get("profit_factor_r") or 0)>1.10),
            "positive_market_breadth_at_least_60pct":bool(eligible_markets and positive_markets/eligible_markets>=0.60),
            "positive_time_slices_at_least_4_of_6":positive_slices>=4,
            "bootstrap_p05_positive":bool((boot.get("mean_r_p05") if boot.get("mean_r_p05") is not None else -99)>0),
        }
        strategies.append({"strategy_id":sid,**overall,"positive_markets":positive_markets,
            "eligible_markets":eligible_markets,"positive_slices":positive_slices,
            "market_breadth_pct":round(100*positive_markets/max(1,eligible_markets),1),
            "bootstrap":boot,"rapid_checks":rapid_checks,
            "rapid_gate":"RAPID_ROBUSTNESS_CANDIDATE" if all(rapid_checks.values()) else "NOT_RAPIDLY_ROBUST",
            "by_market":by_market,"time_slices":slices,"cost_stability":cost_stability})
    candidates=[x["strategy_id"] for x in strategies if x["rapid_gate"]=="RAPID_ROBUSTNESS_CANDIDATE"]
    result={"state":"done","finished":iso(),"seconds":round(time.time()-started,2),
        "deep_markets":deep_markets,"frozen_strategy_count":len(SHADOW_STRATEGIES),
        "primary_cost_profile":EXEC_PRIMARY_COST,"bootstrap_method":"deterministic 30-day calendar blocks",
        "strategies":strategies,"candidate_strategies":candidates,
        "evidence_gate":"PARALLEL_CANDIDATE_FOUND" if candidates else "NO_PARALLEL_ROBUSTNESS_YET",
        "note":"Immediate robustness diagnostics only: cross-market breadth, six chronological slices, all cost profiles and deterministic block bootstrap. Genuine forward evidence continues independently and is not replaced by these tests."}
    DBX.set("parallel_lab",result); return result


def execution_edge_from_frames(frames: dict[str,pd.DataFrame], source_meta: dict|None=None):
    started=time.time()
    DBX.set("execution_edge",{"state":"running","started":iso(),"note":"Building 4h/12h low-turnover raw trades before venue costs."})
    if len(frames)<EXEC_MIN_DEEP_MARKETS:
        raise RuntimeError(f"insufficient execution-edge market coverage: {len(frames)}/{len(EXEC_MARKETS)}")
    raw=[]; errors=[]; units=0; total=max(1,len(frames)*len(EXEC_BAR_HOURS))
    for symbol,df60 in frames.items():
        for hours in EXEC_BAR_HOURS:
            try:
                x=_exec_prepare(df60,hours); busy={}
                for i in range(220,len(x)-2):
                    for sid,fam,direction,sm,tm,hold in _exec_specs(x,i,hours):
                        key=(sid,direction)
                        if i < busy.get(key,-1): continue
                        tr=_exec_raw_trade(x,symbol,i,hours,direction,sid,fam,sm,tm,hold)
                        if tr:
                            raw.append(tr); busy[key]=int(tr["exit_i"])+1
            except Exception as e:
                errors.append(f"{symbol}/{hours}h: {e}")
            units+=1
            if units==1 or units%4==0:
                DBX.set("execution_edge",{"state":"running","started":iso(),"completed_units":units,"total_units":total,
                    "current":f"{symbol}/{hours}h","raw_trades_so_far":len(raw),"seconds":round(time.time()-started,1),"errors":errors[-6:]})

    dedup={tr["trade_id"]:tr for tr in raw}; raw=list(dedup.values())
    rows=_exec_expand_costs(raw)
    valid=[r for r in rows if r.get("r") is not None and np.isfinite(r["r"]) and not r.get("sanity_rejected")]
    pairing=_exec_pair_audit(valid)
    primary_all=[r for r in valid if r["cost_profile"]==EXEC_PRIMARY_COST]
    primary=[r for r in primary_all if r.get("economically_qualified")]
    try:
        parallel_research_lab(primary,valid,len(frames))
    except Exception as e:
        DBX.set("parallel_lab",{"state":"error","at":iso(),"error":repr(e)})

    groups=_arch_group_summary(primary,("strategy_id","family","direction","bar_hours"),min_trades=8)
    for g in groups:
        g["robust_score"]=round((g.get("avg_r") or -9)*math.sqrt(max(1,g["trades"]))-0.01*(g.get("max_drawdown_r") or 0),4)
    groups.sort(key=lambda z:z.get("robust_score",-999),reverse=True)
    top_ids=[g["strategy_id"] for g in groups[:8]]
    all_groups=_arch_group_summary(valid,("strategy_id","family","direction","bar_hours","cost_profile"),min_trades=8)
    cost_compare=[g for g in all_groups if g["strategy_id"] in top_ids]
    order={k:i for i,k in enumerate(EXEC_COST_PROFILES)}
    cost_compare.sort(key=lambda g:(top_ids.index(g["strategy_id"]) if g["strategy_id"] in top_ids else 999,order.get(g["cost_profile"],999)))

    # Five-fold expanding walk-forward on the harsh all-taker futures funding-stress profile.
    if primary:
        lo=min(r["entry_ts"] for r in primary); hi=max(r["entry_ts"] for r in primary); span=max(1,hi-lo)
    else:
        lo=hi=0; span=1
    folds=[]; selected_oos=[]; selected_family_folds=[]
    for fi,(a,b) in enumerate(EXEC_FOLD_BOUNDS,1):
        st=lo+int(span*a); en=lo+int(span*b)
        train=[r for r in primary if r["entry_ts"]<st]; test=[r for r in primary if st<=r["entry_ts"]<en]
        tg={}
        for r in train: tg.setdefault(r["strategy_id"],[]).append(r)
        cand=[]
        for sid,rr in tg.items():
            sm=summarize_research(rr)
            if sm["trades"]<EXEC_MIN_TRAIN_TRADES: continue
            if (sm.get("avg_r") or -99)<=0 or (sm.get("profit_factor_r") or 0)<=1.08: continue
            fam=rr[0]["family"]; bh=rr[0]["bar_hours"]
            score=(sm.get("avg_r") or 0)*math.sqrt(sm["trades"])-0.01*(sm.get("max_drawdown_r") or 0)
            cand.append((score,sid,fam,bh,sm))
        cand.sort(reverse=True,key=lambda z:z[0])
        chosen=[]; used=set()
        for c in cand:
            if c[2] in used: continue
            chosen.append(c); used.add(c[2])
            if len(chosen)>=2: break
        keys={c[1] for c in chosen}; sel=[r for r in test if r["strategy_id"] in keys]
        selected_oos.extend(sel); sm=summarize_research(sel)
        folds.append({"fold":fi,"train_rows":len(train),"eligible_candidates":len(cand),
            "selected":"; ".join(f"{c[1]} [{c[2]}]" for c in chosen) or None,
            "test_trades":sm["trades"],"avg_r":sm["avg_r"],"profit_factor_r":sm["profit_factor_r"],"total_r":sm["total_r"]})
        for fam in used:
            fs=[r for r in sel if r["family"]==fam]; fsm=summarize_research(fs)
            selected_family_folds.append({"fold":fi,"family":fam,**fsm})
    oos=summarize_research(selected_oos)
    profitable=sum(1 for f in folds if f.get("test_trades",0)>0 and (f.get("total_r") or 0)>0)
    by_oos_family=[]
    for fam in sorted({r["family"] for r in selected_oos}):
        rr=[r for r in selected_oos if r["family"]==fam]; sm=summarize_research(rr)
        fp=sum(1 for f in selected_family_folds if f["family"]==fam and f.get("trades",0)>0 and (f.get("total_r") or 0)>0)
        by_oos_family.append({"family":fam,**sm,"profitable_folds":fp})
    wfo={"state":"done","fold_count":5,"folds":folds,"test_trades":oos["trades"],"avg_r":oos["avg_r"],
         "profit_factor_r":oos["profit_factor_r"],"total_r":oos["total_r"],"max_drawdown_r":oos["max_drawdown_r"],
         "profitable_folds":profitable}

    best=groups[0] if groups else None
    checks={
        "cost_pairing_exact":bool(pairing.get("exact_same_entries")),
        "deep_markets_at_least_minimum":len(frames)>=EXEC_MIN_DEEP_MARKETS,
        "primary_candidate_at_least_50_trades":bool(best and best.get("trades",0)>=50),
        "primary_candidate_avg_r_positive":bool(best and (best.get("avg_r") or -99)>0),
        "primary_candidate_pf_gt_1_10":bool(best and (best.get("profit_factor_r") or 0)>1.10),
        "oos_at_least_50_trades":wfo.get("test_trades",0)>=50,
        "oos_avg_r_positive":bool((wfo.get("avg_r") or -99)>0),
        "oos_pf_gt_1_05":bool((wfo.get("profit_factor_r") or 0)>1.05),
        "profitable_folds_at_least_3_of_5":profitable>=3,
    }
    gate="EXECUTION_EDGE_CANDIDATE" if all(checks.values()) else "NO_ROBUST_EXECUTION_EDGE_YET"
    result={
        "state":"done","finished":iso(),"seconds":round(time.time()-started,2),
        "scope":f"{EXEC_LOOKBACK_DAYS}d · {len(frames)} deep markets · synthetic 4h/12h · low turnover · spot/futures same-entry costs · 5-fold WFO · report-only",
        "lookback_days":EXEC_LOOKBACK_DAYS,"deep_markets":len(frames),"target_markets":len(EXEC_MARKETS),
        "deep_coverage_pct":round(100*len(frames)/max(1,len(EXEC_MARKETS)),1),"data_sources":source_meta or {},
        "raw_trades":len(raw),"paired_cost_rows":len(valid),"cost_pairing_audit":pairing,
        "primary_cost_profile":EXEC_PRIMARY_COST,"primary_qualified_rows":len(primary),"candidate_count":len(groups),
        "evidence_gate":gate,"evidence_checks":checks,"top_primary_candidates":groups[:20],
        "same_entry_cost_comparison":cost_compare,"by_family":_arch_group_summary(primary,("family",),min_trades=1),
        "by_bar_hours":_arch_group_summary(primary,("bar_hours",),min_trades=1),
        "walk_forward":wfo,"oos_by_family":by_oos_family,
        "cost_profiles":{k:{"product":v.get("product"),"assumption":v.get("assumption"),
            "entry_fee_bps":v.get("entry_fee_bps"),"target_exit_fee_bps":v.get("target_exit_fee_bps"),
            "funding_bps_8h":v.get("funding_bps_8h")} for k,v in EXEC_COST_PROFILES.items()},
        "note":"v1.9 is execution research only. Futures funding is stress-tested as a constant cost because historical funding data is not available in this engine. Maker-fill profiles are diagnostics only and cannot pass the evidence gate by themselves. Spot-cost rows for simulated shorts are cost-sensitivity diagnostics, not a plain-spot short implementation; borrow/margin mechanics are not modeled.",
        "errors":errors[-12:],
    }
    DBX.set("execution_edge",result); return result


async def execution_edge_once():
    started=time.time(); frames={}; errors=[]
    DBX.set("parallel_lab",{"state":"waiting_for_execution_history","started":iso(),
        "note":"Parallel robustness lab will reuse the exact v1.9 deep-history trades; shadow forward remains independent."})
    DBX.set("execution_edge",{"state":"downloading_deep_history","started":iso(),"lookback_days":EXEC_LOOKBACK_DAYS,
        "target_markets":len(EXEC_MARKETS),"completed_markets":0,"note":"Downloading 60m history for 4h/12h low-turnover research."})
    for idx,symbol in enumerate(EXEC_MARKETS,1):
        try:
            df=await asyncio.to_thread(BR.ohlc,symbol,60,EXEC_LOOKBACK_DAYS); frames[symbol]=df
        except Exception as e:
            errors.append(f"{symbol}: {e}")
        DBX.set("execution_edge",{"state":"downloading_deep_history","started":iso(),"lookback_days":EXEC_LOOKBACK_DAYS,
            "target_markets":len(EXEC_MARKETS),"completed_markets":idx,"deep_markets":len(frames),
            "deep_coverage_pct":round(100*len(frames)/max(1,len(EXEC_MARKETS)),1),"current":symbol,
            "seconds":round(time.time()-started,1),"errors":errors[-8:]})
        await asyncio.sleep(0.06)
    if len(frames)<EXEC_MIN_DEEP_MARKETS:
        result={"state":"insufficient_market_coverage","finished":iso(),"deep_markets":len(frames),
            "target_markets":len(EXEC_MARKETS),"evidence_gate":"NO_ROBUST_EXECUTION_EDGE_YET","errors":errors[-12:]}
        DBX.set("execution_edge",result); return result
    meta={"bitstamp_deep":len(frames),"failed_markets":len(EXEC_MARKETS)-len(frames)}
    return await asyncio.to_thread(execution_edge_from_frames,frames,meta)



# -------------------- v2.0 Shadow Forward Lab (forward-only, persistent, isolated) --------------------
SHADOW_FRAMES: dict[str, pd.DataFrame] = {}
SHADOW_LAST_REFRESH_HOUR: int | None = None


def _shadow_strategy_hash() -> str:
    payload={"strategies":SHADOW_STRATEGIES,"markets":SHADOW_MARKETS,"cost":SHADOW_COST_PROFILE,
             "max_signal_delay_min":SHADOW_MAX_SIGNAL_DELAY_MIN}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:16]


def _shadow_signal_on_latest(x: pd.DataFrame, strategy_id: str):
    if len(x) < 221:
        return None
    i=len(x)-1; r=x.iloc[i]
    need=["close","atr14","ema20","ema50","ema200","rsi14","prior_low20","prior_low40","vol_ratio"]
    if not all(np.isfinite(r.get(k,np.nan)) for k in need):
        return None
    strong_dn=bool(r["ema20"] < r["ema50"] < r["ema200"])
    if not strong_dn:
        return None
    if strategy_id=="exec_12h_breakout20_short":
        ok=bool(r["close"] < r["prior_low20"])
    elif strategy_id=="exec_4h_breakout40_short":
        ok=bool(r["close"] < r["prior_low40"] and float(r.get("vol_ratio",0.0)) >= 0.80)
    else:
        return None
    if not ok:
        return None
    return {"signal_i":i,"signal_bar_ts":int(r["time"]),"atr":float(r["atr14"]),"close":float(r["close"])}


def _shadow_summary():
    epoch=int(DBX.get("shadow_forward_epoch",int(time.time())))
    now=int(time.time()); days=max(0.0,(now-epoch)/86400.0)
    trades=DBX.shadow_trades(100000); pos=DBX.shadow_positions(); sig=DBX.shadow_signals(100000)
    valid=[{"r":float(t["r_multiple"]),"entry_ts":int(t["signal_bar_ts"]),"symbol":t["symbol"],"strategy":t["strategy_id"]}
           for t in trades if t.get("r_multiple") is not None and np.isfinite(float(t["r_multiple"]))]
    overall=summarize_research(valid)
    by=[]
    for sid in SHADOW_STRATEGIES:
        rr=[r for r in valid if r["strategy"]==sid]
        sm=summarize_research(rr); by.append({"strategy_id":sid,**sm})
    opened=sum(1 for x in sig if x.get("status")=="OPENED")
    missed=sum(1 for x in sig if x.get("status")=="MISSED_STALE")
    gate=("FORWARD_EVIDENCE_CANDIDATE" if days>=SHADOW_MIN_FORWARD_DAYS and overall["trades"]>=SHADOW_MIN_CLOSED_TRADES
          and (overall.get("avg_r") or -99)>0 and (overall.get("profit_factor_r") or 0)>1.10
          else "FORWARD_COLLECTING")
    return {"state":"running","started_at_epoch":epoch,"started_at":datetime.fromtimestamp(epoch,timezone.utc).isoformat(),
            "days_live":round(days,2),"strategy_hash":_shadow_strategy_hash(),"frozen_strategies":SHADOW_STRATEGIES,
            "markets":len(SHADOW_MARKETS),"cost_profile":SHADOW_COST_PROFILE,"closed_trades":overall["trades"],
            "open_positions":len(pos),"signals_opened":opened,"missed_stale_signals":missed,
            "avg_r":overall["avg_r"],"profit_factor_r":overall["profit_factor_r"],"total_r":overall["total_r"],
            "max_drawdown_r":overall["max_drawdown_r"],"by_strategy":by,"positions":pos[:50],
            "recent_trades":trades[:50],"recent_signals":sig[:50],"evidence_gate":gate,
            "evidence_min_days":SHADOW_MIN_FORWARD_DAYS,"evidence_min_closed_trades":SHADOW_MIN_CLOSED_TRADES,
            "note":"Forward-only shadow execution. No live/paper-account orders. Frozen v1.9 hypotheses; signals after v2.0 epoch only. Conservative futures all-taker + 5bp/8h funding stress."}


async def _shadow_initialise_frames():
    global SHADOW_LAST_REFRESH_HOUR
    errors=[]
    for sym in SHADOW_MARKETS:
        try:
            SHADOW_FRAMES[sym]=await asyncio.to_thread(BR.ohlc,sym,60,SHADOW_HISTORY_DAYS)
        except Exception as e:
            errors.append(f"{sym}: {e}")
        await asyncio.sleep(0.05)
    SHADOW_LAST_REFRESH_HOUR=int(time.time())//3600
    DBX.set("shadow_frame_status",{"at":iso(),"loaded":len(SHADOW_FRAMES),"target":len(SHADOW_MARKETS),"errors":errors[-12:]})


async def _shadow_refresh_frames_if_needed():
    global SHADOW_LAST_REFRESH_HOUR
    h=int(time.time())//3600
    if SHADOW_LAST_REFRESH_HOUR==h:
        return
    errors=[]
    for sym in SHADOW_MARKETS:
        try:
            recent=await asyncio.to_thread(K.ohlc,sym,60)
            old=SHADOW_FRAMES.get(sym)
            if old is None or old.empty:
                SHADOW_FRAMES[sym]=await asyncio.to_thread(BR.ohlc,sym,60,SHADOW_HISTORY_DAYS)
            else:
                x=pd.concat([old,recent[["time","open","high","low","close","volume"]]],ignore_index=True)
                x=x.drop_duplicates("time",keep="last").sort_values("time").reset_index(drop=True)
                cutoff=int(time.time())-SHADOW_HISTORY_DAYS*86400
                SHADOW_FRAMES[sym]=x[x["time"]>=cutoff].reset_index(drop=True)
        except Exception as e:
            errors.append(f"{sym}: {e}")
        await asyncio.sleep(0.03)
    SHADOW_LAST_REFRESH_HOUR=h
    DBX.set("shadow_frame_status",{"at":iso(),"loaded":len(SHADOW_FRAMES),"target":len(SHADOW_MARKETS),"errors":errors[-12:]})


async def shadow_manage_positions():
    cache={}
    for p in DBX.shadow_positions():
        try:
            px=cache.get(p["symbol"])
            if px is None:
                px=await asyncio.to_thread(K.price,p["symbol"]); cache[p["symbol"]]=px
            DBX.shadow_mark(p["id"],px)
            reason=None
            if p["direction"]=="short":
                if px >= float(p["stop_raw"]): reason="STOP"
                elif px <= float(p["target_raw"]): reason="TARGET"
            else:
                if px <= float(p["stop_raw"]): reason="STOP"
                elif px >= float(p["target_raw"]): reason="TARGET"
            try:
                held=(utcnow()-datetime.fromisoformat(p["entry_time"])).total_seconds()/3600.0
            except Exception:
                held=0.0
            if reason is None and held>=float(p["hold_hours_max"]): reason="TIME"
            if reason:
                DBX.shadow_close(p["id"],px,reason)
        except Exception as e:
            DBX.set("shadow_last_position_error",{"at":iso(),"position_id":p.get("id"),"error":repr(e)})


async def shadow_scan_signals():
    epoch=int(DBX.get("shadow_forward_epoch",int(time.time())))
    now=int(time.time())
    for sym,df60 in list(SHADOW_FRAMES.items()):
        for sid,cfg in SHADOW_STRATEGIES.items():
            try:
                hours=int(cfg["bar_hours"]); x=_exec_prepare(df60,hours)
                sig=_shadow_signal_on_latest(x,sid)
                if not sig: continue
                bar_ts=int(sig["signal_bar_ts"]); bar_end=bar_ts+hours*3600
                if bar_end <= epoch: continue
                if DBX.shadow_signal_exists(sid,sym,bar_ts): continue
                delay_min=(now-bar_end)/60.0
                if delay_min < -1:
                    continue
                if delay_min > SHADOW_MAX_SIGNAL_DELAY_MIN:
                    DBX.shadow_signal(sid,sym,hours,bar_ts,bar_end,"MISSED_STALE",json.dumps({"delay_min":round(delay_min,1)}))
                    continue
                if DBX.shadow_has_position(sid,sym):
                    DBX.shadow_signal(sid,sym,hours,bar_ts,bar_end,"SKIPPED_ALREADY_OPEN","{}")
                    continue
                raw_px=float(await asyncio.to_thread(K.price,sym)); atr=float(sig["atr"])
                sm=float(cfg["stop_mult"]); tm=float(cfg["target_mult"])
                stop=raw_px+sm*atr; target=raw_px-tm*atr
                risk=abs(stop-raw_px); risk_bps=risk/raw_px*10000.0 if raw_px>0 else 0.0
                if target<=0 or risk_bps<EXEC_MIN_RISK_BPS:
                    DBX.shadow_signal(sid,sym,hours,bar_ts,bar_end,"REJECTED_SANITY",json.dumps({"risk_bps":risk_bps}))
                    continue
                entry_slip=float(SHADOW_COST["entry_slippage_bps"])/10000.0
                entry_exec=raw_px*(1-entry_slip) # adverse sell fill for short
                DBX.shadow_open(sid,sym,hours,"short",bar_ts,bar_end,raw_px,entry_exec,stop,target,risk,
                                float(cfg["hold_bars"])*hours,SHADOW_COST_PROFILE)
                DBX.shadow_signal(sid,sym,hours,bar_ts,bar_end,"OPENED",json.dumps({
                    "entry_raw":raw_px,"entry_exec":entry_exec,"stop":stop,"target":target,"atr":atr,
                    "delay_min":round(delay_min,1),"cost_profile":SHADOW_COST_PROFILE},separators=(",",":")))
            except Exception as e:
                DBX.set("shadow_last_signal_error",{"at":iso(),"symbol":sym,"strategy":sid,"error":repr(e)})


async def shadow_loop():
    if DBX.get("shadow_forward_epoch") is None:
        DBX.set("shadow_forward_epoch",int(time.time()))
        DBX.set("shadow_strategy_hash",_shadow_strategy_hash())
        DBX.set("shadow_preregistered_at",iso())
    await asyncio.sleep(8)
    try:
        await _shadow_initialise_frames()
    except Exception as e:
        DBX.set("shadow_frame_status",{"at":iso(),"error":repr(e)})
    while not STOP.is_set():
        try:
            await shadow_manage_positions()
            await _shadow_refresh_frames_if_needed()
            await shadow_scan_signals()
            DBX.set("shadow_forward",_shadow_summary())
        except Exception as e:
            DBX.set("shadow_forward",{"state":"error","at":iso(),"error":repr(e)})
        try:
            await asyncio.wait_for(STOP.wait(),timeout=SHADOW_SCAN_SECONDS)
        except asyncio.TimeoutError:
            pass

async def relative_strength_specialist_once():
    started=time.time()
    DBX.set("rs_specialist", {
        "state":"downloading_deep_history","started":iso(),"lookback_days":RS_LOOKBACK_DAYS,
        "target_markets":len(RS_MARKETS),"completed_markets":0,
        "note":"Downloading 60m deep history only; live engine remains independent.",
    })
    frames={}; errors=[]
    for idx,symbol in enumerate(RS_MARKETS,1):
        try:
            # Deep specialist deliberately uses only complete Bitstamp history.
            df=await asyncio.to_thread(BR.ohlc,symbol,60,RS_LOOKBACK_DAYS)
            frames[symbol]=df
        except Exception as e:
            errors.append(f"{symbol}: {e}")
        DBX.set("rs_specialist", {
            "state":"downloading_deep_history","started":iso(),"lookback_days":RS_LOOKBACK_DAYS,
            "target_markets":len(RS_MARKETS),"completed_markets":idx,"deep_markets":len(frames),
            "deep_coverage_pct":round(100*len(frames)/max(1,len(RS_MARKETS)),1),
            "current":symbol,"seconds":round(time.time()-started,1),"errors":errors[-8:],
            "note":"180d specialist accepts only complete deep history; no shallow fallback.",
        })
        await asyncio.sleep(0.08)
    if "XBTUSD" not in frames or len(frames)<RS_MIN_DEEP_MARKETS:
        result={
            "state":"insufficient_market_coverage","finished":iso(),"deep_markets":len(frames),
            "target_markets":len(RS_MARKETS),"deep_coverage_pct":round(100*len(frames)/max(1,len(RS_MARKETS)),1),
            "evidence_gate":"NO_ROBUST_RS_EDGE_YET","errors":errors[-12:],
            "note":"Not enough complete 180d markets to run a defensible relative-strength test.",
        }
        DBX.set("rs_specialist",result)
        DBX.set("architecture", {
            "state":"insufficient_market_coverage","finished":iso(),"deep_markets":len(frames),
            "target_markets":len(RS_MARKETS),"evidence_gate":"NO_ROBUST_MULTI_FAMILY_EDGE_YET",
            "note":"The v1.8 architecture uses the same complete deep-market set and cannot run defensibly with insufficient coverage.",
        })
        try:
            await execution_edge_once()
        except Exception as e:
            DBX.set("execution_edge", {"state":"error","at":iso(),"error":repr(e)})
        return result
    meta={"bitstamp_deep":len(frames),"failed_markets":len(RS_MARKETS)-len(frames)}
    rs_result=await asyncio.to_thread(rs_specialist_from_frames,frames,meta)
    try:
        await asyncio.to_thread(multi_family_architecture_from_frames,frames,meta)
    except Exception as e:
        DBX.set("architecture", {"state":"error","at":iso(),"error":repr(e)})
    try:
        await execution_edge_once()
    except Exception as e:
        DBX.set("execution_edge", {"state":"error","at":iso(),"error":repr(e)})
    return rs_result

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
    DBX.set("rs_specialist", {"state": "waiting_for_base_research", "started": iso(), "note": "180d specialist will run after base research."})
    DBX.set("architecture", {"state": "waiting_for_deep_180d", "started": iso(), "note": "v1.8 architecture will reuse the specialist 180d download."})
    DBX.set("execution_edge", {"state": "waiting_for_prior_research", "started": iso(), "note": "v1.9 low-turnover execution research runs after architecture."})
    DBX.set("parallel_lab", {"state": "waiting_for_execution_history", "started": iso(), "note": "v2.1 parallel validation runs automatically after v1.9 raw trades are rebuilt."})
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
    try:
        await relative_strength_specialist_once()
    except Exception as e:
        DBX.set("rs_specialist", {"state": "error", "at": iso(), "error": repr(e)})
        DBX.set("architecture", {"state": "error", "at": iso(), "error": repr(e)})
        try:
            await execution_edge_once()
        except Exception as ex:
            DBX.set("execution_edge", {"state": "error", "at": iso(), "error": repr(ex)})
    return result


STOP = asyncio.Event()
ENGINE_TASK: Optional[asyncio.Task] = None
RESEARCH_TASK: Optional[asyncio.Task] = None
SHADOW_TASK: Optional[asyncio.Task] = None


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
    global ENGINE_TASK, RESEARCH_TASK, SHADOW_TASK
    if ENGINE_TASK is None or ENGINE_TASK.done():
        ENGINE_TASK = asyncio.create_task(engine_loop())
    if RESEARCH_TASK is None or RESEARCH_TASK.done():
        RESEARCH_TASK = asyncio.create_task(research_loop())
    if SHADOW_TASK is None or SHADOW_TASK.done():
        SHADOW_TASK = asyncio.create_task(shadow_loop())


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
        "rs_specialist": DBX.get("rs_specialist", {"state": "waiting"}),
        "architecture": DBX.get("architecture", {"state": "waiting"}),
        "execution_edge": DBX.get("execution_edge", {"state": "waiting"}),
        "parallel_lab": DBX.get("parallel_lab", {"state": "waiting"}),
        "shadow_forward": DBX.get("shadow_forward", _shadow_summary()),
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
            "shadow_positions_rows": len(DBX.shadow_positions()),
            "shadow_trades_rows": len(DBX.shadow_trades(100000)),
            "shadow_signals_rows": len(DBX.shadow_signals(100000)),
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


@app.get("/api/rs-specialist")
def api_rs_specialist():
    return JSONResponse(DBX.get("rs_specialist", {"state": "waiting"}))


@app.get("/api/architecture")
def api_architecture():
    return JSONResponse(DBX.get("architecture", {"state": "waiting"}))


@app.get("/api/execution-edge")
def api_execution_edge():
    return JSONResponse(DBX.get("execution_edge", {"state": "waiting"}))


@app.get("/api/parallel-lab")
def api_parallel_lab():
    return JSONResponse(DBX.get("parallel_lab", {"state": "waiting"}))


@app.get("/api/shadow-forward")
def api_shadow_forward():
    return JSONResponse(DBX.get("shadow_forward", _shadow_summary()))


@app.get("/api/shadow-positions")
def api_shadow_positions():
    return JSONResponse(DBX.shadow_positions())


@app.get("/api/shadow-trades")
def api_shadow_trades(limit: int = 200):
    return JSONResponse(DBX.shadow_trades(min(max(limit,1),1000)))


HTML = r"""
<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Alpha v2.1</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;margin:20px;background:#0d1117;color:#e6edf3;max-width:1350px}
h1{margin-bottom:4px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}.c{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px}.b{font-size:24px;font-weight:700}.m{color:#8b949e}.warn{background:#2d2205;padding:12px;border-radius:10px}table{width:100%;border-collapse:collapse;background:#161b22;margin-top:10px;overflow:auto}th,td{padding:8px;border-bottom:1px solid #30363d;font-size:12px;text-align:left}code{color:#79c0ff}.small{font-size:12px}
</style></head><body>
<h1>Alpha v2.1 — Parallel Research + Shadow Forward</h1>
<div class="m">Live engine unchanged · rapid parallel robustness diagnostics · v2.0 forward epoch and persistent shadow ledger preserved · paper only.</div>
<p class="warn"><b>Research/paper only.</b> New short signals are simulated only. No live short or real-order route exists. Walk-forward out-of-sample evidence is the promotion gate.</p>
<div id="cards" class="grid"></div>
<h2>Live performance</h2><div id="perf"></div>
<h2>Fixed-strategy deep research</h2><div id="research"></div><div id="researchstrat"></div><div id="researchtf"></div><h3>Fixed-strategy walk-forward</h3><div id="wfo"></div>
<h2>Research Lab — parameter search & cost stress</h2><div id="lab"></div><h3>Best candidates (full sample; hypothesis generation)</h3><div id="labtop"></div><h3>Best candidate by cost stress</h3><div id="labcost"></div><h3>Lab walk-forward selection</h3><div id="labwfo"></div>
<h2>v1.6 Edge Explorer — broad sources of edge</h2><div id="edge"></div><h3>Top edge candidates</h3><div id="edgetop"></div><h3>Best by cost stress</h3><div id="edgecost"></div><h3>Current-cost family / direction diagnostics</h3><div id="edgefam"></div><div id="edgedir"></div><h3>Edge walk-forward selection</h3><div id="edgewfo"></div>
<h2>v1.7 Relative Strength Specialist — 180d</h2><div id="rs"></div><h3>Top specialist candidates</h3><div id="rstop"></div><h3>Best by cost stress</h3><div id="rscost"></div><h3>Current-cost diagnostics</h3><div id="rswindow"></div><div id="rsdir"></div><h3>5-fold walk-forward selection</h3><div id="rswfo"></div>
<h2>v1.8.1 Architecture — sanity filters + venue-aware costs</h2><div id="arch"></div><h3>Top current-cost candidates</h3><div id="archtop"></div><h3>Same-entry cost comparison</h3><div id="archcost"></div><h3>Current-cost family / regime diagnostics</h3><div id="archfam"></div><div id="archreg"></div><h3>5-fold outer walk-forward</h3><div id="archwfo"></div><h3>OOS family diagnostics</h3><div id="archoosfam"></div>
<h2>v1.9 Low-Turnover Execution Edge — 365d</h2><div id="exec"></div><h3>Top primary-cost candidates</h3><div id="exectop"></div><h3>Same-entry spot / futures comparison</h3><div id="execcost"></div><h3>Primary-cost diagnostics</h3><div id="execfam"></div><div id="exectf"></div><h3>5-fold execution walk-forward</h3><div id="execwfo"></div><h3>Execution OOS family diagnostics</h3><div id="execoosfam"></div>
<h2>v2.1 Parallel Robustness Lab — immediate diagnostics</h2><div id="parallel"></div><h3>Frozen strategy robustness</h3><div id="parallelstrat"></div><h3>Chronological stability</h3><div id="paralleltime"></div><h3>Cross-market breadth</h3><div id="parallelmarket"></div><h3>Cost stability</h3><div id="parallelcost"></div>
<h2>v2.0 Shadow Forward Lab — forward-only</h2><div id="shadow"></div><h3>Frozen strategy results</h3><div id="shadowstrat"></div><h3>Open shadow positions</h3><div id="shadowpos"></div><h3>Recent shadow trades</h3><div id="shadowtrades"></div>
<h2>Live strategy performance</h2><div id="strat"></div>
<h2>Open positions</h2><div id="pos"></div>
<h2>Recent trades</h2><div id="trades"></div>
<h2>Recent signals</h2><div id="sig"></div>
<script>
const n=(x,d=2)=>x==null?'—':Number(x).toFixed(d);
function tbl(r,c){if(!r||!r.length)return'<div class="m">None</div>';return'<div style="overflow:auto"><table><tr>'+c.map(x=>'<th>'+x+'</th>').join('')+'</tr>'+r.map(a=>'<tr>'+c.map(x=>'<td>'+String(a[x]??'')+'</td>').join('')+'</tr>').join('')+'</table></div>'}
async function go(){
 let[s,p,t,g]=await Promise.all([fetch('/api/status').then(r=>r.json()),fetch('/api/positions').then(r=>r.json()),fetch('/api/trades?limit=30').then(r=>r.json()),fetch('/api/signals?limit=50').then(r=>r.json())]);
 let l=s.last_scan||{},f=s.performance||{},r=s.research||{},a=s.lab||{},e=s.edge||{},q=s.rs_specialist||{},z=s.architecture||{},v=s.execution_edge||{},u=s.parallel_lab||{},h=s.shadow_forward||{};
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

 document.getElementById('rs').innerHTML='<div class="grid">'+[['State',q.state||'waiting'],['History',(q.lookback_days||180)+'d'],['Deep markets',(q.deep_markets??0)+'/'+(q.target_markets??25)],['Deep coverage',n(q.deep_coverage_pct,1)+'%'],['Trade rows',q.trade_rows??q.trade_rows_so_far??0],['Candidates',q.candidate_count??'—'],['Evidence',q.evidence_gate||'running'],['Seconds',n(q.seconds,1)],['Scope',q.scope||q.note||'waiting']].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div><div class="m small">'+(q.note||'')+'</div>';
 document.getElementById('rstop').innerHTML=tbl(q.top_candidates||[],['strategy_id','direction','window_h','regime_filter','threshold_pct','cost_profile','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r','robust_score']);
 document.getElementById('rscost').innerHTML=tbl(q.best_by_cost||[],['cost_profile','strategy_id','direction','window_h','trades','avg_r','profit_factor_r','total_r','max_drawdown_r']);
 document.getElementById('rswindow').innerHTML='<h4>By horizon</h4>'+tbl(q.by_window||[],['window_h','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r']);
 document.getElementById('rsdir').innerHTML='<h4>By direction</h4>'+tbl(q.by_direction||[],['direction','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r']);
 let qw=q.walk_forward||{}; document.getElementById('rswfo').innerHTML='<div class="grid">'+[['State',qw.state||'waiting'],['OOS trades',qw.test_trades||0],['OOS Avg R',n(qw.avg_r,3)],['OOS PF',n(qw.profit_factor_r,2)],['OOS Total R',n(qw.total_r,2)],['Profitable folds',(qw.profitable_folds||0)+'/'+(qw.fold_count||5)]].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div>'+tbl(qw.folds||[],['fold','train_rows','eligible_candidates','selected','test_trades','avg_r','profit_factor_r','total_r']);
 document.getElementById('arch').innerHTML='<div class="grid">'+[['State',z.state||'waiting'],['Deep markets',(z.deep_markets??0)+'/'+(z.target_markets??25)],['Raw trades',z.raw_trades??z.raw_trades_so_far??0],['Paired cost rows',z.paired_cost_rows??0],['Valid cost rows',z.valid_cost_rows??0],['Sanity rejected',z.sanity_rejected_cost_rows??0],['Cost baseline',z.current_cost_profile||'—'],['Current qualified',z.current_cost_qualified_rows??0],['Candidates',z.candidate_count??'—'],['Cost pairing',(z.cost_pairing_audit||{}).exact_same_entries===true?'EXACT':'—'],['Evidence',z.evidence_gate||'running'],['Seconds',n(z.seconds,1)],['Scope',z.scope||z.note||'waiting']].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div><div class="m small">'+(z.note||'')+'</div>';
 document.getElementById('archtop').innerHTML=tbl(z.top_current_candidates||[],['strategy_id','family','direction','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r','qualified_pct','robust_score']);
 document.getElementById('archcost').innerHTML=tbl(z.same_entry_cost_comparison||[],['strategy_id','family','direction','cost_profile','trades','avg_r','profit_factor_r','total_r','max_drawdown_r','qualified_pct']);
 document.getElementById('archfam').innerHTML='<h4>By family</h4>'+tbl(z.by_family||[],['family','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r','qualified_pct']);
 document.getElementById('archreg').innerHTML='<h4>By regime</h4>'+tbl(z.by_regime||[],['regime','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r','qualified_pct']);
 let zw=z.walk_forward||{}; document.getElementById('archwfo').innerHTML='<div class="grid">'+[['State',zw.state||'waiting'],['OOS trades',zw.test_trades||0],['OOS Avg R',n(zw.avg_r,3)],['OOS PF',n(zw.profit_factor_r,2)],['OOS Total R',n(zw.total_r,2)],['Profitable folds',(zw.profitable_folds||0)+'/'+(zw.fold_count||5)],['OOS survivor families',(z.oos_survivor_families||[]).join(', ')||'None']].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div>'+tbl(zw.folds||[],['fold','train_rows','eligible_candidates','selected','test_trades','avg_r','profit_factor_r','total_r']);
 document.getElementById('archoosfam').innerHTML=tbl(z.oos_by_family||[],['family','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r','profitable_folds']);
 document.getElementById('exec').innerHTML='<div class="grid">'+[['State',v.state||'waiting'],['History',(v.lookback_days||365)+'d'],['Deep markets',(v.deep_markets??0)+'/'+(v.target_markets??12)],['Raw trades',v.raw_trades??v.raw_trades_so_far??0],['Paired cost rows',v.paired_cost_rows??0],['Primary cost',v.primary_cost_profile||'—'],['Primary qualified',v.primary_qualified_rows??0],['Candidates',v.candidate_count??'—'],['Cost pairing',(v.cost_pairing_audit||{}).exact_same_entries===true?'EXACT':'—'],['Evidence',v.evidence_gate||'running'],['Seconds',n(v.seconds,1)],['Scope',v.scope||v.note||'waiting']].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div><div class="m small">'+(v.note||'')+'</div>';
 document.getElementById('exectop').innerHTML=tbl(v.top_primary_candidates||[],['strategy_id','family','direction','bar_hours','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r','qualified_pct','robust_score']);
 document.getElementById('execcost').innerHTML=tbl(v.same_entry_cost_comparison||[],['strategy_id','family','direction','bar_hours','cost_profile','trades','avg_r','profit_factor_r','total_r','max_drawdown_r','qualified_pct']);
 document.getElementById('execfam').innerHTML='<h4>By family</h4>'+tbl(v.by_family||[],['family','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r','qualified_pct']);
 document.getElementById('exectf').innerHTML='<h4>By bar size</h4>'+tbl(v.by_bar_hours||[],['bar_hours','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r','qualified_pct']);
 let vw=v.walk_forward||{}; document.getElementById('execwfo').innerHTML='<div class="grid">'+[['State',vw.state||'waiting'],['OOS trades',vw.test_trades||0],['OOS Avg R',n(vw.avg_r,3)],['OOS PF',n(vw.profit_factor_r,2)],['OOS Total R',n(vw.total_r,2)],['Profitable folds',(vw.profitable_folds||0)+'/'+(vw.fold_count||5)]].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div>'+tbl(vw.folds||[],['fold','train_rows','eligible_candidates','selected','test_trades','avg_r','profit_factor_r','total_r']);
 document.getElementById('execoosfam').innerHTML=tbl(v.oos_by_family||[],['family','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r','profitable_folds']);
 let us=u.strategies||[];
 document.getElementById('parallel').innerHTML='<div class="grid">'+[['State',u.state||'waiting'],['Deep markets',u.deep_markets??'—'],['Frozen strategies',u.frozen_strategy_count??2],['Bootstrap',u.bootstrap_method||'waiting'],['Evidence',u.evidence_gate||'running'],['Seconds',n(u.seconds,2)]].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div><div class="m small">'+(u.note||'')+'</div>';
 document.getElementById('parallelstrat').innerHTML=tbl(us.map(x=>({strategy_id:x.strategy_id,trades:x.trades,avg_r:x.avg_r,profit_factor_r:x.profit_factor_r,total_r:x.total_r,market_breadth_pct:x.market_breadth_pct,positive_slices:(x.positive_slices||0)+'/6',bootstrap_p05:(x.bootstrap||{}).mean_r_p05,bootstrap_p50:(x.bootstrap||{}).mean_r_p50,prob_positive_pct:(x.bootstrap||{}).prob_mean_r_positive_pct,rapid_gate:x.rapid_gate})),['strategy_id','trades','avg_r','profit_factor_r','total_r','market_breadth_pct','positive_slices','bootstrap_p05','bootstrap_p50','prob_positive_pct','rapid_gate']);
 document.getElementById('paralleltime').innerHTML=tbl(us.flatMap(x=>(x.time_slices||[]).map(y=>({strategy_id:x.strategy_id,...y}))),['strategy_id','slice','start','end','trades','avg_r','profit_factor_r','total_r','max_drawdown_r']);
 document.getElementById('parallelmarket').innerHTML=tbl(us.flatMap(x=>(x.by_market||[]).map(y=>({strategy_id:x.strategy_id,...y}))),['strategy_id','symbol','trades','avg_r','profit_factor_r','total_r','max_drawdown_r']);
 document.getElementById('parallelcost').innerHTML=tbl(us.flatMap(x=>(x.cost_stability||[]).map(y=>({strategy_id:x.strategy_id,...y}))),['strategy_id','cost_profile','trades','avg_r','profit_factor_r','total_r','max_drawdown_r']);
 document.getElementById('shadow').innerHTML='<div class="grid">'+[['State',h.state||'waiting'],['Days forward',n(h.days_live,2)],['Closed trades',h.closed_trades||0],['Open shadow',h.open_positions||0],['Avg R',n(h.avg_r,3)],['PF',n(h.profit_factor_r,2)],['Total R',n(h.total_r,2)],['Evidence',h.evidence_gate||'collecting'],['Cost',h.cost_profile||'—'],['Hash',h.strategy_hash||'—']].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b" style="font-size:16px">'+x[1]+'</div></div>').join('')+'</div><div class="m small">'+(h.note||'')+'</div>';
 document.getElementById('shadowstrat').innerHTML=tbl(h.by_strategy||[],['strategy_id','trades','avg_r','win_rate_pct','profit_factor_r','total_r','max_drawdown_r']);
 document.getElementById('shadowpos').innerHTML=tbl(h.positions||[],['strategy_id','symbol','bar_hours','direction','entry_time','entry_raw','stop_raw','target_raw','last_price']);
 document.getElementById('shadowtrades').innerHTML=tbl(h.recent_trades||[],['exit_time','strategy_id','symbol','bar_hours','entry_raw','exit_raw','held_hours','r_multiple','exit_reason']);
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
