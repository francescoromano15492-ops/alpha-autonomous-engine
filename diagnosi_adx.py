"""Analisi in sola lettura dello storico diagnostico: confronto soglie ADX.
Ricalcola il solo filtro ADX (adx > soglia) lasciando invariati gli altri 5 filtri
salvati in ogni riga. Non scrive file e non tocca stato, posizioni o ordini.
Uso: python diagnosi_adx.py [percorso.jsonl]
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys

THRESHOLDS = (25, 22, 20)
OTHERS = ('breakout', 'direzione', 'volume', 'candela', 'volatilita')


def load(path):
    """Ritorna (scansioni, righe_incomplete). Mai eccezioni per righe malformate."""
    scans, bad = [], []
    try:
        lines = Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError as exc:
        return [], [f'file non leggibile: {exc}']
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            hour = datetime.fromisoformat(rec['candela_chiusa_utc']).astimezone(timezone.utc)
            pairs = [p for p in rec['coppie']
                     if isinstance(p, dict) and 'simbolo' in p and 'adx' in p
                     and all(side in p and all(k in p[side] for k in OTHERS) for side in ('long', 'short'))]
            if not pairs:
                raise ValueError('nessuna coppia valida')
            scans.append({'n': n, 'hour': hour, 'rec': rec, 'pairs': pairs,
                          'partial': len(pairs) != len(rec['coppie'])})
        except (ValueError, KeyError, TypeError) as exc:
            bad.append(f'riga {n}: {type(exc).__name__} {exc}')
    return scans, bad


def fmt(h):
    return f'{h:%m-%d %H}'


def ranges(hours):
    out, start, prev = [], None, None
    for h in sorted(hours):
        if start is None:
            start = prev = h
        elif h - prev == timedelta(hours=1):
            prev = h
        else:
            out.append((start, prev)); start = prev = h
    if start is not None:
        out.append((start, prev))
    return ', '.join(fmt(a) if a == b else f'{fmt(a)}..{fmt(b)}' for a, b in out) or 'nessuna'


def signals(scan, threshold):
    found = []
    for p in scan['pairs']:
        for side in ('long', 'short'):
            if all(p[side][k] for k in OTHERS) and p['adx'] > threshold:
                found.append((p['simbolo'], side.upper(), p['adx'], p.get('volume_relativo')))
    return found


def report(path, emit=print):
    emit(f'=== ANALISI ADX | {path} ===')
    scans, bad = load(path)
    emit(f'Righe valide {len(scans)} | incomplete o illeggibili {len(bad)}'
         f' | con coppie parziali {sum(s["partial"] for s in scans)}')
    for b in bad[:10]:
        emit('  scartata: ' + b)
    if not scans:
        emit('=== FINE ANALISI ADX (nessun dato) ==='); return
    count = Counter(s['hour'] for s in scans)
    hours = sorted(count)
    full = {hours[0] + timedelta(hours=i) for i in range(int((hours[-1] - hours[0]).total_seconds() // 3600) + 1)}
    missing = sorted(full - set(hours))
    dups = {h: c for h, c in count.items() if c > 1}
    emit(f'Ore UTC presenti ({len(hours)}): {ranges(hours)}')
    emit(f'Ore mancanti nel periodo ({len(missing)}): {ranges(missing)}')
    emit('Duplicati: ' + (', '.join(f'{fmt(h)} x{c}' for h, c in sorted(dups.items())) or 'nessuno')
         + ' (contati una sola volta, vale l\'ultima riga di ogni ora)')
    unique = {}
    for s in scans:
        unique[s['hour']] = s
    ordered = [unique[h] for h in hours]
    mism = [fmt(s['hour']) for s in ordered
            if s['rec'].get('segnali') is not None and s['rec']['segnali'] != len(signals(s, 25))]
    emit('Controllo ADX 25 ricalcolato vs segnali registrati dal bot: '
         + ('coerente su tutte le ore' if not mism else 'differenze alle ore ' + ', '.join(mism)))
    base = {(fmt(s['hour']), a, b) for s in ordered for a, b, _, _ in signals(s, 25)}
    for t in THRESHOLDS:
        rows = [(s, x) for s in ordered for x in signals(s, t)]
        hrs = sorted({fmt(s['hour']) for s, _ in rows})
        emit(f'--- ADX > {t}: {len(rows)} segnali su {len(ordered)} scansioni | ore: {", ".join(hrs) or "nessuna"}')
        for s, (sym, side, adx, vol) in rows:
            extra = '' if (fmt(s['hour']), sym, side) in base else ' | AGGIUNTIVO rispetto ad ADX 25'
            emit(f'    {s["hour"]:%Y-%m-%d %H:%M} UTC {sym} {side} | ADX {adx} | volume {vol}x'
                 f' | filtri mancanti: nessuno{extra}')
    emit('--- Quasi segnali (5 filtri su 6, ADX mancante a 25) con soglia che li recupera')
    for s in ordered:
        for p in s['pairs']:
            for side in ('long', 'short'):
                if all(p[side][k] for k in OTHERS) and p['adx'] <= 25:
                    rec = [str(t) for t in THRESHOLDS if p['adx'] > t] or ['nessuna']
                    emit(f'    {s["hour"]:%m-%d %H:%M} {p["simbolo"]} {side.upper()} | ADX {p["adx"]}'
                         f' | mancava solo ADX | passa con soglia: {", ".join(rec)}')
    emit('=== FINE ANALISI ADX ===')


if __name__ == '__main__':
    report(sys.argv[1] if len(sys.argv) > 1 else '/data/donchian-paper-diagnosi.jsonl')
