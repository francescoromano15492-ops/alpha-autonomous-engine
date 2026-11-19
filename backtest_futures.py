"""Studio in sola lettura: verifica filtri e confronto ADX 25/22/20 su storico Binance USDT perpetual.
Solo dati futures (fapi). Se non raggiungibili: NON VERIFICATO, nessuna fonte alternativa.
Non scrive nello stato del simulatore; replica le regole di donchian-paper.py senza modificarle.
"""
from collections import OrderedDict
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
import json
import os
import time

BASE = os.environ.get('BT_BASE', 'https://fapi.binance.com/fapi/v1/')
TAKER, MAKER, START_EQ = .0005, .0002, 300.
FILTERS = ('breakout', 'ADX', 'direzione', 'volume', 'candela', 'volatilita')
THRESHOLDS = (25, 22, 20)
SLEEP = float(os.environ.get('BT_SLEEP', '0.5'))
WARM = 300  # barre di riscaldamento: l'effetto della finestra da 288 del bot e' ormai trascurabile


def utc(ts, fmt='%Y-%m-%d %H:%M'):
    return datetime.fromtimestamp(ts, timezone.utc).strftime(fmt)


def get(endpoint, **params):
    for attempt in range(6):
        try:
            time.sleep(SLEEP)
            with urlopen(BASE + endpoint + '?' + urlencode(params), timeout=30) as r:
                return json.load(r)
        except HTTPError as exc:
            if exc.code in (418, 429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(10 * (attempt + 1)); continue
            raise
        except OSError:
            if attempt == 5:
                raise
            time.sleep(5)


def hourly(symbol, start, end):
    """Barre 1h chiuse con apertura in [start, end)."""
    out, cur = [], start
    while cur < end:
        rows = get('klines', symbol=symbol, interval='1h', limit=1500,
                   startTime=cur * 1000, endTime=end * 1000 - 1)
        if not rows:
            break
        out += [(int(r[0]) // 1000,) + tuple(map(float, r[1:6])) for r in rows]
        cur = out[-1][0] + 3600
        if len(rows) < 2:
            break
    return out


def analyse_symbol(sym, bars, acc, cands, probe=None):
    """Indicatori ricorsivi come in scan(); filtri sull'ultima barra chiusa di ogni indice k."""
    n = len(bars)
    if n <= WARM:
        return
    t = [b[0] for b in bars]; o = [b[1] for b in bars]; h = [b[2] for b in bars]
    l = [b[3] for b in bars]; c = [b[4] for b in bars]; v = [b[5] for b in bars]
    atr = h[0] - l[0]; pos = neg = adx = 0.; prev_c = c[0]
    for i in range(1, n):
        atr += (max(h[i] - l[i], abs(h[i] - prev_c), abs(l[i] - prev_c)) - atr) / min(i + 1, 14)
        prev_c = c[i]
        up, down = h[i] - h[i-1], l[i-1] - l[i]
        pos += ((up if up > max(down, 0) else 0) - pos) / min(i, 14)
        neg += ((down if down > max(up, 0) else 0) - neg) / min(i, 14)
        if i >= 14:
            dx = 100 * abs(pos - neg) / (pos + neg) if pos + neg else 0.
            adx += (dx - adx) / min(i - 13, 14)
        if i < WARM:
            continue
        mean = sum(v[i-19:i+1]) / 20
        upper = max(h[i-48:i]); lower = min(l[i-48:i])
        rng = h[i] - l[i]
        vol_ok = atr / c[i] <= .035
        sig_t = t[i] + 3600
        if probe is not None:
            probe.append((t[i], adx, atr, mean, upper, lower))
        for side, name in ((1, 'long'), (-1, 'short')):
            if side == 1:
                chk = (c[i] > upper, adx > 25, pos > neg, v[i] > 2.2 * mean,
                       rng > 0 and (c[i] - l[i]) / rng >= .65, vol_ok)
                level = upper
            else:
                chk = (c[i] < lower, adx > 25, neg > pos, v[i] > 2.6 * mean,
                       rng > 0 and (h[i] - c[i]) / rng >= .65, vol_ok)
                level = lower
            acc.add(sig_t, name, chk)
            others = [chk[j] for j in range(6) if j != 1]
            if all(others):
                cands.setdefault(sig_t, []).append((-v[i] / mean, sym, atr, side, level, adx))


class Blockers:
    def __init__(self):
        self.n = {'long': 0, 'short': 0}
        self.alone = {s: [0] * 6 for s in self.n}
        self.blocked = {s: [0] * 6 for s in self.n}
        self.all6 = {s: 0 for s in self.n}

    def add(self, _t, side, chk):
        self.n[side] += 1
        fails = [j for j in range(6) if not chk[j]]
        for j in range(6):
            self.alone[side][j] += chk[j]
        if len(fails) == 1:
            self.blocked[side][fails[0]] += 1
        elif not fails:
            self.all6[side] += 1

    def lines(self, title):
        out = [f'{title}: combinazioni coppia x ora x lato = {self.n["long"] + self.n["short"]}']
        for side in ('long', 'short'):
            out.append(f'  {side.upper()} | n {self.n[side]} | tutti e 6 i filtri (ADX 25): {self.all6[side]}')
            out.append('    blocca da solo (gli altri 5 ok): ' + ', '.join(
                f'{FILTERS[j]} {self.blocked[side][j]}' for j in range(6)))
            out.append('    supera il filtro (da solo, quota): ' + ', '.join(
                f'{FILTERS[j]} {100*self.alone[side][j]/max(self.n[side],1):.1f}%' for j in range(6)))
        return out


def diag_blockers(path, emit):
    """Stessa conta sullo storico diagnostico salvato dal bot (prima riga registrata per ogni ora)."""
    acc, first, bad = Blockers(), {}, 0
    try:
        for line in open(path, encoding='utf-8', errors='replace'):
            try:
                rec = json.loads(line)
                first.setdefault(rec['candela_chiusa_utc'], rec)
            except (ValueError, KeyError, TypeError):
                bad += 1
    except OSError as exc:
        emit(f'BT | storico diagnostico non leggibile: {exc}'); return
    for key, rec in sorted(first.items()):
        for p in rec['coppie']:
            for side in ('long', 'short'):
                try:
                    acc.add(key, side, tuple(bool(p[side][f]) for f in FILTERS))
                except (KeyError, TypeError):
                    bad += 1
    for ln in acc.lines(f'BT | STORICO DIAGNOSTICO ({len(first)} ore, righe scartate {bad})'):
        emit(ln)


class Minutes:
    def __init__(self, end, max_chunks=300):
        self.cache, self.end, self.max = OrderedDict(), end, max_chunks

    def iterate(self, sym, start, cap_days=60):
        cur, stop = start, min(self.end, start + cap_days * 86400)
        while cur < stop:
            key = (sym, cur)
            if key not in self.cache:
                rows = get('klines', symbol=sym, interval='1m', limit=1500,
                           startTime=cur * 1000, endTime=self.end * 1000 - 1)
                self.cache[key] = [(int(r[0]) // 1000,) + tuple(map(float, r[1:5])) for r in rows]
                while len(self.cache) > self.max:
                    self.cache.popitem(last=False)
            rows = self.cache[key]
            if not rows:
                return
            yield from rows
            cur = rows[-1][0] + 60


def trade(mode, sig, t0, equity, minutes):
    """Ritorna (stato, ora_uscita, pnl, costi, entrata). Stato: chiusa, scaduta, aperta."""
    _, sym, atr, side, level, _a = sig
    pos, last = None, None
    for mt, op, hi, lo, cl in minutes.iterate(sym, t0):
        last = (mt, cl)
        hour = mt // 3600 * 3600
        if pos is None:
            if mt >= t0 + 7200:
                return 'scaduta', t0 + 7200, 0., 0., None
            if mode == 'market':
                fee, entry = TAKER, op
            else:
                at_open = op <= level if side == 1 else op >= level
                if mt == t0 and at_open:
                    fee, entry = TAKER, op
                elif (lo < level) if side == 1 else (hi > level):
                    fee, entry = MAKER, level
                else:
                    continue
            qty = equity * .33 / (entry * (1 + fee))
            pos = dict(entry=entry, fee=fee, qty=qty, prot=False, trig=None, t=mt)
            continue
        e = pos['entry']
        if pos['trig'] is not None and hour > pos['trig']:
            pos['prot'] = True
        stop = e - side * (.2 if pos['prot'] else (2 if side == 1 else 1.8)) * atr
        target = e + side * (5 if side == 1 else 4) * atr
        hs = lo <= stop if side == 1 else hi >= stop
        ht = hi >= target if side == 1 else lo <= target
        if hs or ht:
            price = stop if hs else target
            costs = pos['qty'] * (e * pos['fee'] + price * TAKER)
            return 'chiusa', mt, side * pos['qty'] * (price - e) - costs, costs, pos['t']
        if pos['trig'] is None and ((hi >= e + 2 * atr) if side == 1 else (lo <= e - 2 * atr)):
            pos['trig'] = hour
    if pos is None:
        return 'scaduta' if last and last[0] >= t0 + 7200 else 'aperta', (last[0] if last else t0), 0., 0., None
    price = last[1]
    costs = pos['qty'] * (pos['entry'] * pos['fee'] + price * TAKER)
    return 'aperta', last[0], side * pos['qty'] * (price - pos['entry']) - costs, costs, pos['t']


def simulate(mode, thr, cands, p_start, p_end, minutes):
    t, equity, trades = p_start, START_EQ, []
    while t < p_end:
        pool = [c for c in cands.get(t, []) if c[5] > thr]
        if not pool:
            t += 3600; continue
        status, tx, pnl, costs, tin = trade(mode, min(pool), t, equity, minutes)
        if status == 'scaduta':
            t = max(t + 7200, t + 3600); continue
        if status == 'aperta':
            trades.append((status, tin, tx, pnl, costs, min(pool)))
            break
        equity += pnl
        trades.append((status, tin, tx, pnl, costs, min(pool)))
        t = tx // 3600 * 3600 + 3600
    return trades, equity


def stats(trades, days):
    closed = [x for x in trades if x[0] == 'chiusa']
    eq, peak, mdd = START_EQ, START_EQ, 0.
    for x in closed:
        eq += x[3]; peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak)
    openp = [x for x in trades if x[0] == 'aperta']
    return dict(n=len(closed), per_day=len(closed) / days if days else 0,
                net=sum(x[3] for x in closed), mdd=100 * mdd, costs=sum(x[4] for x in closed),
                wins=sum(1 for x in closed if x[3] > 0), open=len(openp),
                open_pnl=sum(x[3] for x in openp))


def line(label, s):
    return (f'{label} | operazioni {s["n"]} ({s["per_day"]:.3f}/giorno) | netto {s["net"]:+.2f} USDT'
            f' ({100*s["net"]/START_EQ:+.2f}%) | perdita max capitale {s["mdd"]:.2f}% | costi {s["costs"]:.2f}'
            f' | vincenti {s["wins"]}' + (f' | aperte {s["open"]} (provvisorio {s["open_pnl"]:+.2f})' if s['open'] else ''))


def run(symbols, emit, days=int(os.environ.get('BT_DAYS', '730')), test_days=int(os.environ.get('BT_TEST_DAYS', '240'))):
    emit('BT | === STUDIO FILTRI E VARIANTI ADX (Binance USDT perpetual) ===')
    end = int(time.time() - 5) // 3600 * 3600
    start = end - days * 86400
    acc, cands, got = Blockers(), {}, []
    for sym in symbols:
        bars = hourly(sym, start, end)
        got.append((sym, len(bars)))
        analyse_symbol(sym, bars, acc, cands)
    emit('BT | barre 1h per coppia: ' + ', '.join(f'{s} {n}' for s, n in got))
    emit(f'BT | periodo dati {utc(start)} - {utc(end)} UTC, riscaldamento {WARM} barre per coppia')
    for ln in acc.lines('BT | STORICO FUTURES AMPIO'):
        emit(ln)
    first_sig = min((t for t in cands), default=start)
    p0 = start + WARM * 3600
    split = end - test_days * 86400
    periods = {'SELEZIONE': (p0, split), 'VERIFICA': (split, end)}
    emit(f'BT | selezione {utc(p0)} - {utc(split)} ({(split-p0)/86400:.0f} gg) | verifica {utc(split)} - {utc(end)} ({test_days} gg)')
    emit('BT | regola di scelta, fissata prima dei risultati: miglior netto market+limit nella SELEZIONE, minimo 10 operazioni chiuse per modalita')
    minutes = Minutes(end)
    emit('BT | simulazione in corso (richiede molti minuti per le candele a 1 minuto)')
    res = {}
    for thr in THRESHOLDS:
        for pname, (a, b) in periods.items():
            for mode in ('market', 'limit'):
                trades, _eq = simulate(mode, thr, cands, a, b, minutes)
                s = stats(trades, (b - a) / 86400)
                res[(thr, pname, mode)] = s
                emit(line(f'BT | ADX {thr} | {pname} | {mode}', s))
    ok = {thr: sum(res[(thr, 'SELEZIONE', m)]['net'] for m in ('market', 'limit'))
          for thr in THRESHOLDS if all(res[(thr, 'SELEZIONE', m)]['n'] >= 10 for m in ('market', 'limit'))}
    if ok:
        best = max(ok, key=ok.get)
        emit(f'BT | variante scelta in SELEZIONE: ADX {best} (netto combinato {ok[best]:+.2f}); verifica fuori campione sopra')
    else:
        emit('BT | nessuna variante ha almeno 10 operazioni chiuse per modalita nella SELEZIONE: scelta non significativa')
    emit('BT | COSTI INCLUSI: taker 0.05% in entrata market e in uscita, maker 0.02% per ingresso limit. '
         'ESCLUSI: funding, slippage, spread bid/ask, priorita in coda e fill parziali. '
         'Prezzi: massimi/minimi a 1 minuto, stop controllato prima del target nello stesso minuto, '
         'ingresso market all\'apertura del minuto del segnale. Perdita max sulle sole operazioni chiuse (non intra-operazione).')
    emit('BT | === FINE STUDIO ===')
