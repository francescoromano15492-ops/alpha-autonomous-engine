"""Ricostruzione in sola lettura dei segnali extra con ADX 20 (assenti con ADX 25).
Usa SOLO candele Binance USDT perpetual (fapi). Se i dati futures non sono
raggiungibili stampa "NON VERIFICATA" e non usa alcuna fonte alternativa.
Non scrive file e non tocca stato, posizioni o ordini del simulatore.
Regole copiate da donchian-paper.py: stop 2 ATR (long) / 1.8 ATR (short), target 5 / 4 ATR,
protezione a 0.2 ATR dall'ora successiva al movimento di 2 ATR, scadenza ordine 2 ore,
commissioni 0.05% taker e 0.02% maker, 33% del saldo di 300 USDT.
"""
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen
import json
import time

BASE = 'https://fapi.binance.com/fapi/v1/'
TAKER, MAKER, EQUITY = .0005, .0002, 300.
OTHERS = ('breakout', 'direzione', 'volume', 'candela', 'volatilita')


def _get(endpoint, **params):
    with urlopen(BASE + endpoint + '?' + urlencode(params), timeout=20) as r:
        return json.load(r)


def _utc(ts):
    return f'{datetime.fromtimestamp(ts, timezone.utc):%m-%d %H:%M}'


def extra_signals(path, thr=20, base=25):
    """Primo record registrato per ora (valori originali) con segnale a thr e non a base."""
    first, bad = {}, 0
    for line in open(path, encoding='utf-8', errors='replace'):
        try:
            rec = json.loads(line)
            h = int(datetime.fromisoformat(rec['candela_chiusa_utc']).timestamp())
            first.setdefault(h, rec)
        except (ValueError, KeyError, TypeError):
            bad += 1
    out = []
    for h, rec in sorted(first.items()):
        for p in rec['coppie']:
            for side in ('long', 'short'):
                try:
                    if all(p[side][k] for k in OTHERS) and thr < p['adx'] <= base:
                        out.append((h, p, side))
                except (KeyError, TypeError):
                    pass
    return out


def indicators(rows):
    """ATR(14 Wilder come nel bot), minimo/massimo 48 barre precedenti, ultima barra."""
    seq = [tuple(map(float, r[1:6])) for r in rows]
    c = seq[0][3]
    atr = seq[0][1] - seq[0][2]
    for k in range(1, len(seq)):
        _, hi, lo, close, _ = seq[k]
        atr += (max(hi - lo, abs(hi - c), abs(lo - c)) - atr) / min(k + 1, 14)
        c = close
    i = len(seq) - 1
    return dict(atr=atr, upper=max(x[1] for x in seq[i-48:i]), lower=min(x[2] for x in seq[i-48:i]),
                close=seq[i][3])


def simulate(mode, side, level, atr, h0, minutes, now):
    """Un minuto alla volta, usando massimo e minimo come prezzo raggiungibile. Prima lo stop."""
    pos, ev = None, []
    for t, o, hi, lo, cl in minutes:
        if t + 60 > now:
            break                                   # minuto ancora in formazione
        hour = t // 3600 * 3600
        if pos is None:
            if t >= h0 + 7200:
                ev.append('SCADUTO alle ' + _utc(h0 + 7200) + ' senza ingresso'); return ev
            if mode == 'market':
                fee, entry = TAKER, o
            else:
                reach = hi >= level if side == -1 else lo <= level
                if not reach and not (t == h0 and (o >= level if side == -1 else o <= level)):
                    continue
                if t == h0 and (o >= level if side == -1 else o <= level):
                    fee, entry = TAKER, o
                else:
                    fee, entry = MAKER, level
            qty = EQUITY * .33 / (entry * (1 + fee))
            pos = dict(entry=entry, fee=fee, qty=qty, prot=False, trig=None)
            ev.append(f'INGRESSO {_utc(t)} prezzo {entry} commissione {fee*100:.2f}% qty {qty:.4f}')
            continue
        e = pos['entry']
        if pos['trig'] is not None and hour > pos['trig'] and not pos['prot']:
            pos['prot'] = True
            ev.append(f'PROTEZIONE attiva dalle {_utc(t)}: stop a {e - side*.2*atr:.4f}')
        stop = e - side * (.2 if pos['prot'] else (2 if side == 1 else 1.8)) * atr
        target = e + side * (5 if side == 1 else 4) * atr
        hs = hi >= stop if side == -1 else lo <= stop
        ht = lo <= target if side == -1 else hi >= target
        if hs or ht:
            price = stop if hs else target
            if hs and ht:
                ev.append('stop e target nello stesso minuto: assunto stop')
            pnl = side * pos['qty'] * (price - e) - pos['qty'] * (e * pos['fee'] + price * TAKER)
            ev.append(f"USCITA {_utc(t)} {'stop' if hs else 'target'} {price:.4f} | risultato dopo costi {pnl:+.4f} USDT")
            return ev
        if pos['trig'] is None and ((lo <= e - 2*atr) if side == -1 else (hi >= e + 2*atr)):
            pos['trig'] = hour
            ev.append(f'TRIGGER 2 ATR alle {_utc(t)}')
    if pos is None:
        ev.append('ORDINE ANCORA PENDENTE (non scaduto, non eseguito)'); return ev
    price = cl
    pnl = side * pos['qty'] * (price - e) - pos['qty'] * (e * pos['fee'] + price * TAKER)
    ev.append(f"POSIZIONE APERTA | stop {e - side*(.2 if pos['prot'] else (2 if side==1 else 1.8))*atr:.4f}"
              f" target {e + side*(5 if side==1 else 4)*atr:.4f} protetta={pos['prot']}"
              f" | provvisorio a {price} (chiusura {_utc(t-60)}): {pnl:+.4f} USDT dopo costi di uscita")
    return ev


def report(path, emit=print):
    emit('=== RICOSTRUZIONE FUTURES (Binance USDT perpetual) ===')
    sigs = extra_signals(path)
    emit(f'Segnali extra con ADX 20 e non con ADX 25: {len(sigs)}')
    for h0, p, side in sigs[:5]:
        sym, s = p['simbolo'], (1 if side == 'long' else -1)
        emit(f'--- {sym} {side.upper()} ora {_utc(h0)} UTC | valori registrati: ADX {p["adx"]}'
             f' volume {p["volume_relativo"]}x ATR% {p["atr_percento"]}')
        try:
            rows = _get('klines', symbol=sym, interval='1h', limit=288, endTime=h0 * 1000 - 1)
            if [int(r[0]) // 1000 for r in rows] != list(range(h0 - 288 * 3600, h0, 3600)):
                raise ValueError('candele orarie incomplete')
            ind = indicators(rows)
            minutes = [(int(r[0]) // 1000,) + tuple(map(float, r[1:5]))
                       for r in _get('klines', symbol=sym, interval='1m', startTime=h0 * 1000, limit=1500)]
        except Exception as exc:
            emit(f'RICOSTRUZIONE NON VERIFICATA: dati futures non recuperati ({type(exc).__name__}: {exc}). '
                 'Nessun dato spot usato.')
            continue
        atr_pct = round(100 * ind['atr'] / ind['close'], 3)
        emit(f'Dati futures: ATR {ind["atr"]:.4f} | ATR% ricalcolato {atr_pct} vs registrato {p["atr_percento"]}'
             f' | livello {ind["lower"] if s == -1 else ind["upper"]} | chiusura segnale {ind["close"]}')
        if abs(atr_pct - p['atr_percento']) > .002:
            emit('ATTENZIONE: ATR% ricalcolato diverso dal registrato, ricostruzione non verificata')
            continue
        level = ind['lower'] if s == -1 else ind['upper']
        now = time.time()
        emit(f'Minuti futures: {len(minutes)} da {_utc(minutes[0][0])} a {_utc(minutes[-1][0])} UTC')
        for mode in ('market', 'limit'):
            emit(f'  [{mode}]')
            for line in simulate(mode, s, level, ind['atr'], h0, minutes, now):
                emit('    ' + line)
    emit('Prezzi simulati su massimi/minimi a 1 minuto; bid e ask storici non disponibili. ADX operativo: 25.')
    emit('=== FINE RICOSTRUZIONE ===')
