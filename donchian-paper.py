"""Paper trading Donchian 1H. Python 3, sola libreria standard.
Controllo rete: python3 donchian-paper.py --check
Simulazione: python3 donchian-paper.py
Ctrl+C per fermare. Stato e operazioni in donchian-paper-state.json.
Commissioni ipotizzate: 0.05% taker e 0.02% maker simulato.
Funding, profondità e priorità in coda esclusi. Un solo processo per file di stato.
"""
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
import json
import os
import time

SYMBOLS = [a+'USDT' for a in 'AAVE ADA AVAX BCH BNB BTC DOGE DOT ETH HBAR LINK LTC NEAR SOL SUI TRX UNI XLM XRP'.split()]
BASE = 'https://fapi.binance.com/fapi/v1/'
TAKER = .0005
MAKER = .0002
DATA = Path(os.environ.get('DATA_DIR', '/data' if os.path.isdir('/data') else '.'))
STATE = DATA / 'donchian-paper-state.json'
DIAGNOSTICS = DATA / 'donchian-paper-diagnosi.jsonl'


def save_scan(hour, diagnostics, candidates):
    report = {
        'registrato_utc': datetime.now(timezone.utc).isoformat(),
        'candela_chiusa_utc': datetime.fromtimestamp(hour,timezone.utc).isoformat(),
        'segnali': len(candidates), 'coppie': diagnostics}
    with DIAGNOSTICS.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(report,ensure_ascii=False)+'\n')
    for side in ('long','short'):
        passed = {name: sum(row[side][name] for row in diagnostics)
                  for name in ('breakout','ADX','direzione','volume','candela','volatilita')}
        print(side.upper()+' | '+ ' | '.join(
            f'{name} {count}/{len(diagnostics)}' for name,count in passed.items()),flush=True)
    ranked=sorted(
        ((sum(row[side].values()),row,side) for row in diagnostics for side in ('long','short')),
        key=lambda item: item[0], reverse=True)
    for count,row,side in ranked[:3]:
        excluded=', '.join(name for name,passed in row[side].items() if not passed) or 'nessuno'
        print(f"{row['simbolo']} {side.upper()} | filtri superati {count}/6 | "
              f"esclusioni: {excluded} | volume {row['volume_relativo']}x | ADX {row['adx']}",
              flush=True)


def scan(group, diagnostics=None):
    signals=defaultdict(list)
    for a,bars in group.items():
        ts=sorted(bars)
        c=bars[ts[0]][3]
        atr=bars[ts[0]][1]-bars[ts[0]][2]
        pos=neg=adx=0.
        seq=[bars[t] for t in ts]
        for i,t in enumerate(ts[1:],1):
            op,hi,lo,close,_=bars[t]
            atr+=(max(hi-lo,abs(hi-c),abs(lo-c))-atr)/min(i+1,14)
            c=close
            _,prev_hi,prev_lo,_,_=seq[i-1]
            up,down=hi-prev_hi,prev_lo-lo
            pos+=((up if up>max(down,0) else 0)-pos)/min(i,14)
            neg+=((down if down>max(up,0) else 0)-neg)/min(i,14)
            if i>=14:
                dx=100*abs(pos-neg)/(pos+neg) if pos+neg else 0.
                adx+=(dx-adx)/min(i-13,14)
            if i<200:
                continue
            mean=sum(x[4] for x in seq[i-19:i+1])/20
            upper=max(x[1] for x in seq[i-48:i])
            lower=min(x[2] for x in seq[i-48:i])
            long=hi>lo and seq[i][4]>2.2*mean and c>upper and pos>neg and (c-lo)/(hi-lo)>=.65
            short=hi>lo and seq[i][4]>2.6*mean and c<lower and neg>pos and (hi-c)/(hi-lo)>=.65
            volatility_ok=atr/c<=.035
            long=long and adx>25 and hi>lo and volatility_ok
            short=short and adx>25 and hi>lo and volatility_ok
            if diagnostics is not None and i==len(ts)-1:
                long_checks = {
                    'breakout': c>upper, 'ADX': adx>25, 'direzione': pos>neg,
                    'volume': seq[i][4]>2.2*mean,
                    'candela': hi>lo and (c-lo)/(hi-lo)>=.65,
                    'volatilita': volatility_ok}
                short_checks = {
                    'breakout': c<lower, 'ADX': adx>25, 'direzione': neg>pos,
                    'volume': seq[i][4]>2.6*mean,
                    'candela': hi>lo and (hi-c)/(hi-lo)>=.65,
                    'volatilita': volatility_ok}
                diagnostics.append({
                    'simbolo': a, 'volume_relativo': round(seq[i][4]/mean,3) if mean else 0,
                    'adx': round(adx,2), 'atr_percento': round(100*atr/c,3),
                    'long': long_checks, 'short': short_checks})
            if long or short:
                signals[t+3600].append((-seq[i][4]/mean,a,atr,1 if long else -1,upper if long else lower))
    return signals


def public(endpoint, **params):
    with urlopen(BASE + endpoint + ('?' + urlencode(params) if params else ''), timeout=60) as response:
        return json.load(response)


def bars(symbol, hour):
    rows = public('klines', symbol=symbol, interval='1h', limit=288, endTime=hour*1000-1)
    series = {int(row[0])//1000: tuple(map(float,row[1:6])) for row in rows}
    expected = set(range(hour-288*3600, hour, 3600))
    if series.keys() != expected:
        raise ValueError(f'Candele mancanti o non chiuse: {symbol}')
    return symbol, series


def quotes():
    rows = public('ticker/bookTicker')
    book = {row['symbol']: (float(row['bidPrice']), float(row['askPrice']))
            for row in rows if row['symbol'] in SYMBOLS}
    if any(s not in book or not 0 < book[s][0] <= book[s][1] for s in SYMBOLS):
        raise ValueError('Feed bid/ask incompleto')
    return book


def record(state, now, kind, symbol, **values):
    event = {'utc': datetime.fromtimestamp(now, timezone.utc).isoformat(),
             'evento': kind, 'simbolo': symbol, **values}
    state['log'].append(event)
    print(json.dumps(event, ensure_ascii=False), flush=True)


def update(state, book, now, hour, selected):
    pos = state['position']
    if pos:
        symbol, side, entry = pos['symbol'], pos['side'], pos['entry']
        atr, qty = pos['atr'], pos['qty']
        bid, ask = book[symbol]
        price = bid if side == 1 else ask
        if pos['trigger_hour'] is not None and hour > pos['trigger_hour']:
            pos['protected'] = True
        stop_distance = .2 if pos['protected'] else (2 if side == 1 else 1.8)
        stop = entry-side*stop_distance*atr
        target = entry+side*(5 if side == 1 else 4)*atr
        if side*(price-stop) <= 0 or side*(price-target) >= 0:
            pnl = side*qty*(price-entry)-qty*(entry*pos['fee']+price*TAKER)
            state['equity'] += pnl
            state['position'] = None
            record(state, now, 'uscita', symbol, prezzo=price, bid=bid, ask=ask,
                   motivo='stop' if side*(price-stop)<=0 else 'target',
                   pnl=round(pnl, 4), saldo=round(state['equity'], 4))
        elif pos['trigger_hour'] is None and side*(price-entry) >= 2*atr:
            pos['trigger_hour'] = hour
        return

    pending = state['pending']
    if pending and now >= pending['expires']:
        record(state, now, 'scaduto', pending['symbol'], livello=pending['level'])
        pending = state['pending'] = None
    if pending is None and selected:
        _, symbol, atr, side, level = selected
        pending = state['pending'] = dict(symbol=symbol, atr=atr, side=side,
                                          level=level, expires=hour+7200, new=True)
        record(state, now, 'segnale', symbol, lato=side, livello=level)
    if pending is None:
        return

    symbol, side, level = pending['symbol'], pending['side'], pending['level']
    bid, ask = book[symbol]
    price = ask if side == 1 else bid
    marketable = side*(price-level) <= 0
    resting = not pending.pop('new', False)
    market = state['mode'] == 'market'
    if not market and not (marketable if not resting else side*(price-level)<0):
        return
    fee = MAKER if resting and not market else TAKER
    entry = level if resting and not market else price
    qty = state['equity']*.33/(entry*(1+fee))
    state['position'] = dict(symbol=symbol, side=side, entry=entry, qty=qty,
                             atr=pending['atr'], fee=fee, protected=False, trigger_hour=None)
    state['pending'] = None
    record(state, now, 'ingresso', symbol, lato=side, prezzo=entry, bid=bid, ask=ask,
           commissione=fee, esecuzione='maker simulato' if fee==MAKER else 'taker simulato')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        bid, ask = quotes()['BTCUSDT']
        print(f'Quotazioni ricevute per {len(SYMBOLS)} coppie. BTCUSDT bid={bid} ask={ask}')
        return
    if STATE.exists():
        data = json.loads(STATE.read_text())
    else:
        data = {'hour': None, 'accounts': [
            {'mode': mode, 'equity': 300., 'position': None, 'pending': None, 'log': []}
            for mode in ('market', 'limit')]}
    print('Paper attivo. 300 USDT virtuali iniziali per strategia. Ctrl+C per fermare.', flush=True)
    print('Versione diagnosi. Controllo iniziale anche se questa ora era gia stata analizzata.', flush=True)
    print('Storico diagnosi: '+str(DIAGNOSTICS.resolve()), flush=True)
    first_diagnostic=True
    last_heartbeat=0
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            try:
                now = time.time()
                hour = int(now-5)//3600*3600
                signal = None
                if data['hour'] != hour or first_diagnostic:
                    print('Scarico le candele delle 19 coppie. Attendere...',flush=True)
                    market_data = dict(pool.map(lambda s: bars(s,hour), SYMBOLS))
                    diagnostics=[]
                    candidates = scan(market_data,diagnostics).get(hour, [])
                    if data['hour'] != hour:
                        signal = min(candidates, default=None)
                    save_scan(hour,diagnostics,candidates)
                    first_diagnostic=False
                    print(f'{datetime.fromtimestamp(hour, timezone.utc):%Y-%m-%d %H:%M} UTC | '
                          f'{len(candidates)} segnali', flush=True)
                book = quotes()
                for account in data['accounts']:
                    update(account, book, time.time(), hour, signal)
                data['hour'] = hour
                temp = STATE.with_suffix('.tmp')
                temp.write_text(json.dumps(data, indent=2))
                temp.replace(STATE)
                if time.time()-last_heartbeat>=300:
                    print('Connessione OK | '+datetime.now(timezone.utc).strftime('%H:%M:%S UTC')+
                          ' | market '+str(round(data['accounts'][0]['equity'],2))+
                          ' | limit '+str(round(data['accounts'][1]['equity'],2)),flush=True)
                    last_heartbeat=time.time()
            except HTTPError as exc:
                if exc.code == 451:
                    raise SystemExit('Binance risponde HTTP 451 (area geografica bloccata). Sospendo i tentativi.')
                print(f'Feed o salvataggio: {exc}. Riprovo fra 15 secondi.', flush=True)
            except (OSError, ValueError, KeyError) as exc:
                print(f'Feed o salvataggio: {exc}. Riprovo fra 15 secondi.', flush=True)
            time.sleep(15)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Paper fermato.')
    except OSError as exc:
        raise SystemExit(f'Feed non disponibile: {exc}')
