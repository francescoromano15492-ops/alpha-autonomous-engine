from __future__ import annotations
import asyncio, hashlib, json, math, os, sqlite3, time, uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

APP_VERSION="Alpha v1.2 — Active Intraday 24/7 Paper Engine"
KRAKEN_BASE="https://api.kraken.com/0/public"
DEFAULT_MARKETS=["XBTUSD","ETHUSD","SOLUSD","XRPUSD","ADAUSD","DOGEUSD","LINKUSD","LTCUSD","AVAXUSD","DOTUSD","BCHUSD","ATOMUSD","XLMUSD","UNIUSD","AAVEUSD","ETCUSD","ALGOUSD","NEARUSD","FILUSD","ICPUSD","INJUSD","SUIUSD","ARBUSD","OPUSD","TRXUSD"]
PREREGISTRATION={"version":"v1.2-active-intraday","mode":"paper_only","data_source":"Kraken public REST","markets_default":DEFAULT_MARKETS,"timeframes":[5,15,60],"risk":{"risk_per_trade_pct":0.20,"max_daily_loss_pct":1.50,"max_heat_pct":1.25,"max_positions":5,"fee_bps_each_side":40.0,"slippage_bps_each_side":5.0,"min_net_rr":1.25}}
PREREGISTRATION_HASH=hashlib.sha256(json.dumps(PREREGISTRATION,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:16]

def utcnow(): return datetime.now(timezone.utc)
def iso(): return utcnow().isoformat(timespec="seconds")
def ef(n,d):
    try:return float(os.getenv(n,str(d)).strip())
    except:return d
def ei(n,d):
    try:return int(os.getenv(n,str(d)).strip())
    except:return d

@dataclass(frozen=True)
class Config:
    db_path:str; markets:Tuple[str,...]; timeframes:Tuple[int,...]; scan_seconds:int; starting_cash:float
    risk_per_trade:float; max_daily_loss:float; max_heat:float; max_positions:int
    max_gross_exposure:float; max_position_notional:float; fee_rate:float; slippage_rate:float
    min_notional:float; min_net_rr:float; min_net_target_bps:float
    @classmethod
    def from_env(cls):
        return cls(
            os.getenv("ALPHA_DB_PATH","/data/alpha_v1.db"),
            tuple(x.strip().upper() for x in os.getenv("ALPHA_MARKETS",",".join(DEFAULT_MARKETS)).split(",") if x.strip()),
            tuple(int(x) for x in os.getenv("ALPHA_TIMEFRAMES","5,15,60").split(",") if x.strip()),
            max(30,ei("ALPHA_SCAN_SECONDS",60)),ef("ALPHA_STARTING_CASH",10000.0),
            ef("ALPHA_RISK_PER_TRADE_PCT",0.20)/100,ef("ALPHA_MAX_DAILY_LOSS_PCT",1.50)/100,
            ef("ALPHA_MAX_HEAT_PCT",1.25)/100,max(0,ei("ALPHA_MAX_POSITIONS",5)),
            ef("ALPHA_MAX_GROSS_EXPOSURE_PCT",80.0)/100,ef("ALPHA_MAX_POSITION_NOTIONAL_PCT",20.0)/100,
            ef("ALPHA_FEE_BPS",40.0)/10000,ef("ALPHA_SLIPPAGE_BPS",5.0)/10000,
            ef("ALPHA_MIN_NOTIONAL",25.0),max(0,ef("ALPHA_MIN_NET_RR",1.25)),max(0,ef("ALPHA_MIN_NET_TARGET_BPS",10.0))
        )
CFG=Config.from_env(); BOOT_ID=uuid.uuid4().hex[:12]

class DB:
    def __init__(self,p,cash):
        os.makedirs(os.path.dirname(p) or ".",exist_ok=True); self.path=p
        self.conn=sqlite3.connect(p,check_same_thread=False,timeout=30); self.conn.row_factory=sqlite3.Row
        self.conn.executescript("""PRAGMA journal_mode=WAL;PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS account(id INTEGER PRIMARY KEY CHECK(id=1),cash REAL NOT NULL,starting_cash REAL NOT NULL,realized_pnl REAL NOT NULL DEFAULT 0,updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS positions(id INTEGER PRIMARY KEY AUTOINCREMENT,symbol TEXT NOT NULL,timeframe INTEGER NOT NULL,strategy TEXT NOT NULL,entry_time TEXT NOT NULL,entry_price REAL NOT NULL,qty REAL NOT NULL,stop_price REAL NOT NULL,target_price REAL NOT NULL,initial_risk REAL NOT NULL,entry_fee REAL NOT NULL,last_price REAL NOT NULL,signal_bar_ts INTEGER NOT NULL,UNIQUE(symbol,timeframe,strategy));
CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT,symbol TEXT NOT NULL,timeframe INTEGER NOT NULL,strategy TEXT NOT NULL,entry_time TEXT NOT NULL,exit_time TEXT NOT NULL,entry_price REAL NOT NULL,exit_price REAL NOT NULL,qty REAL NOT NULL,gross_pnl REAL NOT NULL,fees REAL NOT NULL,net_pnl REAL NOT NULL,r_multiple REAL,exit_reason TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS signals(id INTEGER PRIMARY KEY AUTOINCREMENT,ts TEXT NOT NULL,symbol TEXT NOT NULL,timeframe INTEGER NOT NULL,strategy TEXT NOT NULL,signal_bar_ts INTEGER NOT NULL,status TEXT NOT NULL,detail TEXT,UNIQUE(symbol,timeframe,strategy,signal_bar_ts));
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY,value TEXT NOT NULL);"""); self.conn.commit()
        if self.conn.execute("SELECT 1 FROM account WHERE id=1").fetchone() is None:
            self.conn.execute("INSERT INTO account VALUES(1,?,?,0,?)",(cash,cash,iso())); self.conn.commit()
        if self.get("persistent_state_id") is None:
            self.set("persistent_state_id",uuid.uuid4().hex[:16]); self.set("persistent_state_created_at",iso())
    def account(self): return dict(self.conn.execute("SELECT * FROM account WHERE id=1").fetchone())
    def positions(self): return [dict(r) for r in self.conn.execute("SELECT * FROM positions ORDER BY id").fetchall()]
    def trades(self,n=100): return [dict(r) for r in self.conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?",(n,)).fetchall()]
    def signals(self,n=200): return [dict(r) for r in self.conn.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?",(n,)).fetchall()]
    def count(self,t):
        if t not in {"positions","trades","signals"}: raise ValueError
        return int(self.conn.execute(f"SELECT COUNT(*) n FROM {t}").fetchone()["n"])
    def set(self,k,v):
        self.conn.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(k,json.dumps(v))); self.conn.commit()
    def get(self,k,d=None):
        r=self.conn.execute("SELECT value FROM kv WHERE key=?",(k,)).fetchone()
        if not r:return d
        try:return json.loads(r["value"])
        except:return d
    def exists(self,s,tf,st,bar): return bool(self.conn.execute("SELECT 1 FROM signals WHERE symbol=? AND timeframe=? AND strategy=? AND signal_bar_ts=?",(s,tf,st,int(bar))).fetchone())
    def signal(self,s,tf,st,bar,status,detail):
        try:self.conn.execute("INSERT INTO signals(ts,symbol,timeframe,strategy,signal_bar_ts,status,detail) VALUES(?,?,?,?,?,?,?)",(iso(),s,tf,st,int(bar),status,detail)); self.conn.commit()
        except sqlite3.IntegrityError:pass
    def open(self,s,tf,st,e,q,sl,tp,risk,fee,bar):
        with self.conn:
            a=self.account(); cash=a["cash"]-e*q-fee
            if cash<-1e-6:raise ValueError("insufficient cash")
            self.conn.execute("INSERT INTO positions(symbol,timeframe,strategy,entry_time,entry_price,qty,stop_price,target_price,initial_risk,entry_fee,last_price,signal_bar_ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(s,tf,st,iso(),e,q,sl,tp,risk,fee,e,int(bar)))
            self.conn.execute("UPDATE account SET cash=?,updated_at=? WHERE id=1",(cash,iso()))
    def mark(self,i,p):
        with self.conn:self.conn.execute("UPDATE positions SET last_price=? WHERE id=?",(p,i))
    def close(self,i,px,reason,fr):
        with self.conn:
            row=self.conn.execute("SELECT * FROM positions WHERE id=?",(i,)).fetchone()
            if not row:return
            p=dict(row); fee=px*p["qty"]*fr; proceeds=px*p["qty"]-fee
            gross=(px-p["entry_price"])*p["qty"]; fees=p["entry_fee"]+fee; net=gross-fees
            rm=net/p["initial_risk"] if p["initial_risk"]>0 else None; a=self.account()
            self.conn.execute("UPDATE account SET cash=?,realized_pnl=?,updated_at=? WHERE id=1",(a["cash"]+proceeds,a["realized_pnl"]+net,iso()))
            self.conn.execute("INSERT INTO trades(symbol,timeframe,strategy,entry_time,exit_time,entry_price,exit_price,qty,gross_pnl,fees,net_pnl,r_multiple,exit_reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",(p["symbol"],p["timeframe"],p["strategy"],p["entry_time"],iso(),p["entry_price"],px,p["qty"],gross,fees,net,rm,reason))
            self.conn.execute("DELETE FROM positions WHERE id=?",(i,))
DBX=DB(CFG.db_path,CFG.starting_cash)

class Kraken:
    def __init__(self):
        self.s=requests.Session(); self.s.headers.update({"User-Agent":"Alpha-v1.2-paper/1.0"})
    def get(self,path,params):
        r=self.s.get(f"{KRAKEN_BASE}/{path}",params=params,timeout=12); r.raise_for_status(); d=r.json()
        if d.get("error"):raise RuntimeError("; ".join(d["error"]))
        return d["result"]
    def ohlc(self,s,tf):
        d=self.get("OHLC",{"pair":s,"interval":tf}); k=next(k for k in d if k!="last")
        x=pd.DataFrame(d[k],columns=["time","open","high","low","close","vwap","volume","count"])
        if len(x)<220:raise RuntimeError("not enough bars")
        for c in ["open","high","low","close","vwap","volume"]:x[c]=pd.to_numeric(x[c],errors="coerce")
        x["time"]=pd.to_numeric(x["time"],errors="coerce").astype("int64"); x=x.dropna().sort_values("time").reset_index(drop=True)
        return x.iloc[:-1].copy()
    def price(self,s):
        d=self.get("Ticker",{"pair":s}); return float(d[next(iter(d))]["c"][0])
K=Kraken()

def ind(df):
    x=df.copy(); c,h,l=x.close,x.high,x.low
    for n in [9,20,50,200]:x[f"ema{n}"]=c.ewm(span=n,adjust=False).mean()
    de=c.diff(); g=de.clip(lower=0); lo=-de.clip(upper=0); ag=g.ewm(alpha=1/14,adjust=False,min_periods=14).mean(); al=lo.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rs=ag/al.replace(0,np.nan); x["rsi14"]=100-100/(1+rs); x["rsi14"]=x["rsi14"].fillna(50)
    pc=c.shift(1); tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); x["atr14"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    m=c.rolling(20).mean(); sd=c.rolling(20).std(ddof=0); x["bb_lower"]=m-2*sd
    x["prior_high10"]=h.shift(1).rolling(10).max(); x["prior_high20"]=h.shift(1).rolling(20).max(); x["prior_volmed20"]=x.volume.shift(1).rolling(20).median()
    return x

def signals(df):
    x=ind(df); 
    if len(x)<220:return []
    r,p=x.iloc[-1],x.iloc[-2]; need=["close","atr14","ema9","ema20","ema50","ema200","rsi14","prior_volmed20"]
    if not all(np.isfinite(r[k]) for k in need):return []
    out=[]; trend=r.ema20>r.ema50; strong=r.ema20>r.ema50>r.ema200; liquid=r.volume>=.8*r.prior_volmed20
    if trend and r.close>r.prior_high20 and r.volume>r.prior_volmed20:out.append(("fast_breakout",r.atr14,1.4,3.0))
    if trend and r.close>r.prior_high10 and liquid:out.append(("donchian_10_breakout",r.atr14,1.3,2.8))
    if strong and p.rsi14<=50<r.rsi14 and r.close>r.ema20:out.append(("rsi_reclaim_fast",r.atr14,1.2,2.6))
    if strong and p.close<=p.ema20 and r.close>r.ema20 and 45<=r.rsi14<=68:out.append(("trend_pullback",r.atr14,1.2,2.6))
    if trend and p.ema9<=p.ema20 and r.ema9>r.ema20 and 50<=r.rsi14<=72 and liquid:out.append(("ema9_momentum",r.atr14,1.2,2.5))
    if strong and p.close<p.bb_lower and r.close>r.bb_lower and r.rsi14<60:out.append(("bollinger_reentry_fast",r.atr14,1.1,2.4))
    return [(a,float(b),c,d) for a,b,c,d in out]

def equity(pr=None):
    a=DBX.account(); cash=float(a["cash"]); gross=0
    for p in DBX.positions():gross+=(pr or {}).get(p["symbol"],p["last_price"])*p["qty"]
    return cash+gross,cash,gross
def heat(eq):return 1 if eq<=0 else sum(p["initial_risk"] for p in DBX.positions())/eq
def daily():
    d=utcnow().date().isoformat(); return float(DBX.conn.execute("SELECT COALESCE(SUM(net_pnl),0) x FROM trades WHERE substr(exit_time,1,10)=?",(d,)).fetchone()["x"])
def perf():
    rows=[dict(r) for r in DBX.conn.execute("SELECT * FROM trades ORDER BY id").fetchall()]
    if not rows:return {"closed_trades":0,"win_rate_pct":0,"net_pnl":0,"fees_paid":0,"profit_factor":None,"avg_r":None,"trades_today":0,"by_strategy":[]}
    n=[float(r["net_pnl"]) for r in rows]; w=[x for x in n if x>0]; l=[x for x in n if x<0]; rs=[float(r["r_multiple"]) for r in rows if r["r_multiple"] is not None]; today=utcnow().date().isoformat()
    by=[]
    for st in sorted({r["strategy"] for r in rows}):
        sr=[r for r in rows if r["strategy"]==st]; sn=[float(r["net_pnl"]) for r in sr]; sw=sum(x>0 for x in sn); gp=sum(x for x in sn if x>0); gl=abs(sum(x for x in sn if x<0)); rr=[float(r["r_multiple"]) for r in sr if r["r_multiple"] is not None]
        by.append({"strategy":st,"trades":len(sr),"net_pnl":sum(sn),"win_rate_pct":100*sw/len(sr),"profit_factor":gp/gl if gl else None,"avg_r":sum(rr)/len(rr) if rr else None})
    gp=sum(w); gl=abs(sum(l))
    return {"closed_trades":len(rows),"win_rate_pct":100*len(w)/len(rows),"net_pnl":sum(n),"fees_paid":sum(float(r["fees"]) for r in rows),"profit_factor":gp/gl if gl else None,"avg_r":sum(rs)/len(rs) if rs else None,"trades_today":sum(str(r["exit_time"]).startswith(today) for r in rows),"by_strategy":by}

def costs(e,sl,tp):
    sx=sl*(1-CFG.slippage_rate); tx=tp*(1-CFG.slippage_rate); ef=e*CFG.fee_rate; sf=sx*CFG.fee_rate; tf=tx*CFG.fee_rate
    loss=max(0,(e-sx)+ef+sf); win=(tx-e)-ef-tf
    return {"loss":loss,"win":win,"rr":win/loss if loss>0 else -math.inf,"bps":win/e*10000 if e>0 else -math.inf}
def gate(s,e,sl,tp):
    ps=DBX.positions(); ec=costs(e,sl,tp)
    if CFG.max_positions<=0:return False,"entries_paused",None,None,ec
    if len(ps)>=CFG.max_positions:return False,"max_positions",None,None,ec
    if any(p["symbol"]==s for p in ps):return False,"symbol_already_open",None,None,ec
    eq,cash,gross=equity()
    if daily()<=-CFG.max_daily_loss*eq:return False,"daily_loss_kill",None,None,ec
    if ec["win"]<=0:return False,"negative_net_target",None,None,ec
    if ec["bps"]<CFG.min_net_target_bps:return False,"net_target_too_small",None,None,ec
    if ec["rr"]<CFG.min_net_rr:return False,"net_rr_too_low",None,None,ec
    q=min(eq*CFG.risk_per_trade/ec["loss"],eq*CFG.max_position_notional/e,max(0,eq*CFG.max_gross_exposure-gross)/e,max(0,cash/(1+CFG.fee_rate))/e)
    if q<=0 or q*e<CFG.min_notional:return False,"no_capacity",None,None,ec
    risk=q*ec["loss"]
    if heat(eq)+risk/eq>CFG.max_heat:return False,"portfolio_heat",None,None,ec
    return True,"accepted",q,risk,ec

async def manage(pc):
    for p in DBX.positions():
        try:
            px=pc.get(p["symbol"]) or await asyncio.to_thread(K.price,p["symbol"]); pc[p["symbol"]]=px; DBX.mark(p["id"],px)
            if px<=p["stop_price"]:DBX.close(p["id"],px*(1-CFG.slippage_rate),"STOP",CFG.fee_rate)
            elif px>=p["target_price"]:DBX.close(p["id"],px*(1-CFG.slippage_rate),"TARGET",CFG.fee_rate)
        except Exception as e:DBX.set("last_position_error",repr(e))

async def scan():
    st=time.time(); pc={}; errs=[]; scanned=raw=new=dups=opened=rejected=0; await manage(pc)
    for s in CFG.markets:
        for tf in CFG.timeframes:
            scanned+=1
            try:
                df=await asyncio.to_thread(K.ohlc,s,tf); bar=int(df.iloc[-1].time)
                for strat,atr,sm,tm in signals(df):
                    raw+=1
                    if DBX.exists(s,tf,strat,bar):dups+=1;continue
                    new+=1; px=pc.get(s) or await asyncio.to_thread(K.price,s); pc[s]=px
                    e=px*(1+CFG.slippage_rate); sl=e-sm*atr; tp=e+tm*atr; ok,why,q,risk,ec=gate(s,e,sl,tp)
                    if not ok:
                        rejected+=1;DBX.signal(s,tf,strat,bar,"REJECTED",json.dumps({"reason":why,"net_rr":ec["rr"],"target_net_bps":ec["bps"]},separators=(",",":")));continue
                    try:
                        DBX.open(s,tf,strat,e,q,sl,tp,risk,e*q*CFG.fee_rate,bar);opened+=1;DBX.signal(s,tf,strat,bar,"OPENED",json.dumps({"entry":e,"qty":q,"stop":sl,"target":tp,"net_rr":ec["rr"]},separators=(",",":")))
                    except Exception as ex:rejected+=1;DBX.signal(s,tf,strat,bar,"REJECTED",f"open_error:{ex}")
            except Exception as e:errs.append(f"{s}/{tf}: {e}")
    eq,cash,gross=equity(pc); snap={"state":"idle","finished":iso(),"boot_id":BOOT_ID,"seconds":round(time.time()-st,2),"markets":len(CFG.markets),"scans":scanned,"setups_detected":raw,"new_setups":new,"duplicates":dups,"opened":opened,"rejected":rejected,"open_positions":len(DBX.positions()),"equity":eq,"errors":errs[-10:]}
    DBX.set("last_scan",snap);return snap

STOP=asyncio.Event();TASK=None
async def loop():
    while not STOP.is_set():
        try:await scan()
        except Exception as e:DBX.set("engine_error",repr(e))
        try:await asyncio.wait_for(STOP.wait(),timeout=CFG.scan_seconds)
        except asyncio.TimeoutError:pass

app=FastAPI(title=APP_VERSION)
@app.on_event("startup")
async def startup():
    global TASK
    if TASK is None or TASK.done():TASK=asyncio.create_task(loop())
@app.on_event("shutdown")
async def shutdown():STOP.set()

def status():
    eq,cash,gross=equity();a=DBX.account();ps={"db_path":CFG.db_path,"persistent_state_id":DBX.get("persistent_state_id"),"boot_id":BOOT_ID,"positions_rows":DBX.count("positions"),"trades_rows":DBX.count("trades"),"signals_rows":DBX.count("signals")}
    return {"version":APP_VERSION,"mode":"PAPER ONLY","markets":CFG.markets,"timeframes":CFG.timeframes,"scan_seconds":CFG.scan_seconds,"equity":eq,"cash":cash,"gross_exposure":gross,"realized_pnl_all_time":a["realized_pnl"],"open_positions":len(DBX.positions()),"last_scan":DBX.get("last_scan",{}),"performance":perf(),"risk":{"risk_per_trade_pct":CFG.risk_per_trade*100,"max_open_positions":CFG.max_positions,"fee_bps_each_side":CFG.fee_rate*10000,"slippage_bps_each_side":CFG.slippage_rate*10000,"min_net_rr":CFG.min_net_rr,"leverage":1.0,"shorts":False},"persistence":ps}
@app.get("/health")
def health():return {"ok":True,"version":APP_VERSION,"boot_id":BOOT_ID}
@app.get("/api/status")
def api_status():return JSONResponse(status())
@app.get("/api/positions")
def api_positions():return JSONResponse(DBX.positions())
@app.get("/api/trades")
def api_trades(limit:int=100):return JSONResponse(DBX.trades(min(max(limit,1),500)))
@app.get("/api/signals")
def api_signals(limit:int=200):return JSONResponse(DBX.signals(min(max(limit,1),1000)))
@app.get("/api/performance")
def api_performance():return JSONResponse(perf())

HTML=r"""<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Alpha v1.2</title>
<style>body{font-family:system-ui;margin:20px;background:#0d1117;color:#e6edf3}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.c{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px}.b{font-size:25px;font-weight:700}.m{color:#8b949e}table{width:100%;border-collapse:collapse;background:#161b22;margin-top:10px}th,td{padding:8px;border-bottom:1px solid #30363d;font-size:12px;text-align:left}</style></head><body>
<h1>Alpha v1.2 — Active Intraday 24/7 Paper Engine</h1><p class="m">25 crypto markets · 3 timeframes · 6 intraday families · paper only.</p><div id="cards" class="grid"></div><h2>Performance</h2><div id="perf"></div><h2>Strategy performance</h2><div id="strat"></div><h2>Open positions</h2><div id="pos"></div><h2>Recent trades</h2><div id="trades"></div><h2>Recent signals</h2><div id="sig"></div>
<script>
const n=(x,d=2)=>Number(x||0).toFixed(d);function tbl(r,c){if(!r.length)return'<div class="m">None</div>';return'<table><tr>'+c.map(x=>'<th>'+x+'</th>').join('')+'</tr>'+r.map(a=>'<tr>'+c.map(x=>'<td>'+String(a[x]??'')+'</td>').join('')+'</tr>').join('')+'</table>'}
async function go(){let[s,p,t,g]=await Promise.all([fetch('/api/status').then(r=>r.json()),fetch('/api/positions').then(r=>r.json()),fetch('/api/trades?limit=30').then(r=>r.json()),fetch('/api/signals?limit=40').then(r=>r.json())]);let l=s.last_scan||{},f=s.performance||{};document.getElementById('cards').innerHTML=[['Equity','$'+n(s.equity)],['Net P&L','$'+n(f.net_pnl)],['Closed trades',f.closed_trades||0],['Win rate',n(f.win_rate_pct,1)+'%'],['Open positions',s.open_positions],['Markets',(s.markets||[]).length],['Raw setups',l.setups_detected||0],['New setups',l.new_setups||0],['Opened',l.opened||0],['Rejected',l.rejected||0]].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b">'+x[1]+'</div></div>').join('');document.getElementById('perf').innerHTML='<div class="grid">'+[['Profit factor',f.profit_factor==null?'—':n(f.profit_factor)],['Avg R',f.avg_r==null?'—':n(f.avg_r)],['Fees paid','$'+n(f.fees_paid)],['Trades today',f.trades_today||0]].map(x=>'<div class="c"><div class="m">'+x[0]+'</div><div class="b">'+x[1]+'</div></div>').join('')+'</div>';document.getElementById('strat').innerHTML=tbl(f.by_strategy||[],['strategy','trades','net_pnl','win_rate_pct','profit_factor','avg_r']);document.getElementById('pos').innerHTML=tbl(p,['symbol','timeframe','strategy','entry_price','stop_price','target_price','last_price']);document.getElementById('trades').innerHTML=tbl(t,['exit_time','symbol','strategy','net_pnl','r_multiple','exit_reason']);document.getElementById('sig').innerHTML=tbl(g,['ts','symbol','timeframe','strategy','status','detail'])}go();setInterval(go,10000);
</script></body></html>"""
@app.get("/",response_class=HTMLResponse)
def dash():return HTMLResponse(HTML)
