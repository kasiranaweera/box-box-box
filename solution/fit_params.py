#!/usr/bin/env python3
"""
Box Box Box — Global Parameter Fitter
=======================================
Finds the best global compound params across all 30k historical races.

Key findings:
- pit_lane_time and base_lap_time are PER-RACE (from race_config) — they cancel
  in pairwise comparisons within a race.
- Only the 12 compound params are global.
- Times are rounded to 4dp before sorting (handles fp near-ties).
- pit_stop semantics: lap N = complete N laps on old tyre (Mode B).
"""

import json, glob, os, sys, argparse
import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from race_simulator import (parse_stints, driver_time, simulate_race,
                            DEFAULT_PARAMS, T_REF, TIME_ROUND)

NAMES = ['off_SOFT','off_MEDIUM','off_HARD',
         'deg_SOFT','deg_MEDIUM','deg_HARD',
         'dq_SOFT', 'dq_MEDIUM', 'dq_HARD',
         'ts_SOFT', 'ts_MEDIUM', 'ts_HARD']

BOUNDS = [
    (-5.0, 0.0),   # off_SOFT
    (-1.0, 3.0),   # off_MEDIUM
    ( 0.0, 6.0),   # off_HARD
    ( 0.0, 1.5),   # deg_SOFT
    ( 0.0, 0.5),   # deg_MEDIUM
    ( 0.0, 0.5),   # deg_HARD
    ( 0.0, 0.05),  # dq_SOFT
    ( 0.0, 0.03),  # dq_MEDIUM
    ( 0.0, 0.02),  # dq_HARD
    (-0.5, 0.5),   # ts_SOFT
    (-0.3, 0.3),   # ts_MEDIUM
    (-0.3, 0.3),   # ts_HARD
]

INIT = np.array([-4.0, 2.0, 4.0,
                  0.498, 0.0, 0.126,
                  0.0,   0.0, 0.0,
                 -0.3,   0.2, 0.2])

# ── Data loading ──────────────────────────────────────────────────

def load_all(data_dir, max_races=30000):
    files = sorted(glob.glob(os.path.join(data_dir, 'historical_races', '*.json')))
    races = []
    for f in files:
        if len(races) >= max_races: break
        try:
            with open(f) as fh: raw = json.load(fh)
            races.extend(raw if isinstance(raw, list) else [raw])
        except: pass
    print(f"Loaded {len(races)} races", file=sys.stderr)
    return races[:max_races]

# ── Feature vectors (base_lt and pit_time cancel in same-race pairs) ──────────

def _sa(n):  return n*(n-1)/2
def _sa2(n): return n*(n-1)*(2*n-1)/6

def fv(stints, dT):
    """12-element feature vector for linear constraint building."""
    ci = {'SOFT':0,'MEDIUM':1,'HARD':2}
    v = np.zeros(12)
    for c, n in stints:
        if c not in ci: c = 'MEDIUM'
        i = ci[c]
        v[i]   += n
        v[3+i] += _sa(n)
        v[6+i] += _sa2(n)
        v[9+i] += n * dT
    return v

def parse_race_fast(race):
    """Parse a race into feature vectors for LP/optimization."""
    cfg        = race.get('race_config', {})
    total_laps = int(cfg.get('total_laps', 57))
    base_lt    = float(cfg.get('base_lap_time', 90.0))
    pit_time   = float(cfg.get('pit_lane_time', 22.0))
    track_temp = float(cfg.get('track_temp', 30.0))
    dT         = track_temp - T_REF
    result     = [str(r) for r in race.get('finishing_positions', [])]

    drivers = {}
    for pos_key, strat in race.get('strategies', {}).items():
        did   = strat.get('driver_id', pos_key)
        start = strat.get('starting_tire', 'MEDIUM')
        pits  = strat.get('pit_stops', [])
        try: gp = int(str(pos_key).replace('pos',''))
        except: gp = 99
        stints = parse_stints(start, pits, total_laps)
        drivers[did] = {
            'stints': stints,
            'fv':     fv(stints, dT),
            'grid':   gp,
        }

    return {
        'total_laps': total_laps,
        'base_lt':    base_lt,
        'pit_time':   pit_time,
        'track_temp': track_temp,
        'dT':         dT,
        'result':     result,
        'drivers':    drivers,
    }

# ── Stint signature for tiebreaker detection ──────────────────────

def stint_sig(stints):
    return tuple((c, n) for c, n in stints)

# ── LP: exact parameter recovery ─────────────────────────────────

def build_constraints(parsed_races, nearby=5, max_constraints=200000):
    rows = []
    for r in parsed_races:
        result  = r['result']
        drivers = r['drivers']
        sigs    = {d: stint_sig(dd['stints']) for d, dd in drivers.items()}

        for i in range(len(result)):
            for j in range(i+1, min(i+nearby+1, len(result))):
                wi, li = result[i], result[j]
                if wi not in drivers or li not in drivers: continue
                if sigs.get(wi) == sigs.get(li): continue  # tiebreaker
                rows.append(drivers[wi]['fv'] - drivers[li]['fv'])
                if len(rows) >= max_constraints: break
            if len(rows) >= max_constraints: break
        if len(rows) >= max_constraints: break

    if not rows: return np.zeros((0, 12))
    A = np.array(rows)
    # Deduplicate
    A = np.unique(A, axis=0)
    print(f"  {len(A)} unique constraints", file=sys.stderr)
    return A

def lp_solve(A):
    """Maximize margin eps: A·theta + eps <= 0."""
    if len(A) == 0: return None, -9999
    A_aug = np.hstack([A, np.ones((len(A), 1))])
    b     = np.zeros(len(A))
    c_obj = np.zeros(13); c_obj[-1] = -1.0
    bounds = list(BOUNDS) + [(None, None)]  # eps unbounded

    res = linprog(c_obj, A_ub=A_aug, b_ub=b, bounds=bounds,
                  method='highs', options={'time_limit': 120, 'disp': False})
    if not res.success:
        return None, -9999
    eps = float(-res.fun)
    return res.x[:12], eps

# ── Pairwise ranking loss (for gradient-based polish) ─────────────

def pairwise_loss(x, parsed_races, margin=0.01, nearby=5):
    total = 0.0; n = 0
    for r in parsed_races:
        result  = r['result']
        drivers = r['drivers']
        sigs    = {d: stint_sig(dd['stints']) for d, dd in drivers.items()}
        # dot-product times (base_lt and pit cancel within a race)
        times   = {d: float(np.dot(dd['fv'], x)) for d, dd in drivers.items()}

        for i in range(len(result)):
            for j in range(i+1, min(i+nearby+1, len(result))):
                wi, li = result[i], result[j]
                if wi not in times or li not in times: continue
                if sigs.get(wi) == sigs.get(li): continue
                total += max(0.0, times[wi] - times[li] + margin) ** 2
                n += 1
    return total / max(1, n)

# ── Evaluation ────────────────────────────────────────────────────

def sim_parsed(r, params):
    """Simulate a pre-parsed race."""
    times = {}
    for did, d in r['drivers'].items():
        t = round(driver_time(d['stints'], r['base_lt'], r['track_temp'],
                              r['pit_time'], params), TIME_ROUND)
        times[did] = (t, d['grid'])
    return [d for d, _ in sorted(times.items(), key=lambda z: (z[1][0], z[1][1]))]

def evaluate(parsed, params, n=500, label=''):
    exact = pos = total = 0
    for r in parsed[:n]:
        pred = sim_parsed(r, params)
        exp  = r['result']
        if pred == exp: exact += 1
        pos   += sum(1 for p, e in zip(pred, exp) if p == e)
        total += len(exp)
    nr = min(n, len(parsed))
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Exact {exact}/{nr} ({100*exact/nr:.1f}%)  "
          f"Pos {100*pos/total:.1f}%", file=sys.stderr)
    return exact / nr

def vec_to_params(x, base_params=None):
    p = dict(base_params or DEFAULT_PARAMS)
    p.update({n: float(v) for n, v in zip(NAMES, x)})
    return p

# ── Main pipeline ─────────────────────────────────────────────────

def fit(data_dir='data', max_races=30000, output='solution/params.json'):
    print("=" * 65, file=sys.stderr)
    print("Box Box Box — Global Parameter Fitter", file=sys.stderr)
    print("=" * 65, file=sys.stderr)

    # Load & parse
    races_raw = load_all(data_dir, max_races)
    print("Parsing races...", file=sys.stderr)
    parsed = []
    for race in races_raw:
        try: parsed.append(parse_race_fast(race))
        except: pass
    print(f"Parsed {len(parsed)} races successfully", file=sys.stderr)

    # ── Phase 1: LP ───────────────────────────────────────────────
    print("\n── Phase 1: LP ──────────────────────────────────────────", file=sys.stderr)
    A = build_constraints(parsed, nearby=5, max_constraints=300000)
    x_lp, eps_lp = lp_solve(A)
    print(f"  LP eps = {eps_lp:.4f}", file=sys.stderr)

    if x_lp is not None:
        params_lp = vec_to_params(x_lp)
        evaluate(parsed, params_lp, n=min(1000, len(parsed)), label='LP')
        x_best = x_lp
    else:
        print("  LP failed, using research defaults", file=sys.stderr)
        x_best = INIT.copy()

    # ── Phase 2: Differential Evolution (global search) ───────────
    print("\n── Phase 2: Differential Evolution ─────────────────────", file=sys.stderr)
    sample_size = min(3000, len(parsed))
    idx = np.random.choice(len(parsed), sample_size, replace=False)
    sample = [parsed[i] for i in idx]

    de = differential_evolution(
        pairwise_loss,
        BOUNDS,
        args=(sample, 0.02, 5),
        maxiter=500,
        popsize=15,
        tol=1e-10,
        seed=42,
        disp=False,
        workers=1,
        x0=x_best,
        init='sobol',
        mutation=(0.5, 1.5),
        recombination=0.9,
    )
    print(f"  DE loss = {de.fun:.8f}", file=sys.stderr)
    params_de = vec_to_params(de.x)
    acc_de = evaluate(parsed, params_de, n=min(1000, len(parsed)), label='DE')
    params_lp_acc = evaluate(parsed, vec_to_params(x_best), n=min(1000, len(parsed)), label='LP') if x_lp is not None else 0

    x_best = de.x if acc_de >= params_lp_acc else x_best

    # ── Phase 3: Nelder-Mead polish ───────────────────────────────
    print("\n── Phase 3: Nelder-Mead polish ──────────────────────────", file=sys.stderr)
    nm = minimize(
        pairwise_loss,
        x_best,
        args=(sample, 0.005, 5),
        method='Nelder-Mead',
        options={'maxiter': 500000, 'xatol': 1e-12, 'fatol': 1e-12, 'disp': False}
    )
    print(f"  NM loss = {nm.fun:.10f}", file=sys.stderr)
    params_nm = vec_to_params(nm.x)
    acc_nm = evaluate(parsed, params_nm, n=min(1000, len(parsed)), label='NM')

    if acc_nm > evaluate(parsed, vec_to_params(x_best), n=50, label='prev'):
        x_best = nm.x

    # ── Phase 4: Full dataset evaluation ─────────────────────────
    print("\n── Final Evaluation ─────────────────────────────────────", file=sys.stderr)
    params_final = vec_to_params(x_best)
    evaluate(parsed, params_final, n=len(parsed), label='FINAL')

    print("\nLearned parameters:", file=sys.stderr)
    for n, v in zip(NAMES, x_best):
        print(f"  {n:20s} = {v:.8f}", file=sys.stderr)

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as f:
        json.dump(params_final, f, indent=2)
    print(f"\nSaved to {output}", file=sys.stderr)
    return params_final


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--max-races', type=int, default=30000)
    ap.add_argument('--output', default='solution/params.json')
    args = ap.parse_args()
    fit(args.data_dir, args.max_races, args.output)