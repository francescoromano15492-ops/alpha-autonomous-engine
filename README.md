# Alpha v1.0 — Autonomous 24/7 Paper Trading Engine

This is the first always-on Alpha build. It is intentionally **paper-only**:
there is no broker key, no live-order endpoint, no leverage and no shorting.

## What it does
- Runs continuously as a web service.
- Pulls public Kraken OHLC data for liquid crypto markets.
- Scans 5m / 15m / 60m by default.
- Uses three frozen intraday strategy families.
- Uses only completed candles for signal generation.
- Applies deterministic portfolio risk limits before every paper entry.
- Persists account state, positions, signals and closed trades in SQLite.
- Serves a live browser dashboard and JSON API.

## Frozen v1.0 Batch A
Preregistration hash: `e411da0c10b819a1`

## Deploy: Railway
1. Put these files in a new GitHub repository or a new branch.
2. In Railway, create a service from that repository.
3. Add a persistent volume mounted at `/data`.
4. Set `ALPHA_DB_PATH=/data/alpha_v1.db`.
5. Deploy. `railway.json` already starts one Uvicorn worker.
6. Open the generated public URL. The dashboard updates every 10 seconds.

**Use one process/worker only.** Multiple Uvicorn workers would run duplicate trading loops.

## Why this is not in Streamlit
Streamlit is kept for research. An H24 engine needs an always-on process with persistent state.
This service is the execution/paper layer; the research app and this service should remain separate.

## Default risk
- 0.25% equity risk per trade
- 1.50% daily realized-loss kill switch
- 1.25% maximum portfolio heat
- 5 positions maximum
- 80% gross exposure cap
- 20% notional cap per position
- no leverage
- long-only
- configurable conservative fees/slippage

## API
- `/` live dashboard
- `/health`
- `/api/status`
- `/api/positions`
- `/api/trades`
- `/api/signals`

## Important
This is a research/paper environment. Historical or simulated profitability does not establish
future profitability. The v1.0 rules should be frozen while forward evidence is collected.
