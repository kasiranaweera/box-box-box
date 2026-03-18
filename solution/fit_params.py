#!/usr/bin/env python3
"""
Box Box Box — Fast Chunked Parameter Fitter
=============================================
Processes 30k races in chunks, saves progress after each chunk.
Much faster than full DE — uses LP per chunk then averages.

Usage:
  python solution/fit_params.py --data-dir data --chunk-size 1000
  python solution/fit_params.py --data-dir data --quick   # just 2000 races, ~2 min
  python solution/fit_params.py --data-dir data --lp-only # LP only, fastest
"""
import json, glob, os, sys, argparse, time
import numpy as np
from scipy.optimize import linprog, minimize
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from race_simulator import parse_stints, driver_time, simulate_race, DEFAULT_PARAMS, T_REF, TIME_ROUND, _c

NAMES  = ['off_SOFT','off_MEDIUM','off_HARD',
          'deg_SOFT','deg_MEDIUM','deg_HARD',
          'dq_SOFT','dq_MEDIUM','dq_HARD',
          'ts_SOFT','ts_MEDIUM','ts_HARD']

BOUNDS = [(-5,0),(-2,3),(0,5),
          (0,1),(0,.5),(0,.3),
          (0,.06),(0,.05),(0,.04),
          (-.4,.4),(-.3,.3),(-.3,.3)]

# ── helpers ───────────────────────────────────────────────────────────────────

def sa(n):  return n*(n-1)/2
def sa2(n): return n*(n-1)*(2*n-1)/6

def make_fv(stints, dT):
    ci = {'SOFT':0,'MEDIUM':1,'HARD':2}
    v = np.zeros(12)
    for c,n in stints:
        c = _c(c); i = ci.get(c,1)
        v[i]+=n; v[3+i]+=sa(n); v[6+i]+=sa2(n); v[9+i]+=n*dT
    return v

def parse_race(race):
    cfg        = race.get('race_config', {})
    total_laps = int(cfg.get('total_laps', 57))
    base_lt    = float(cfg.get('base_lap_time', 90.0))
    pit_time   = float(cfg.get('pit_lane_time', 22.0))
    track_temp = float(cfg.get('track_temp', 30.0))
    dT         = track_temp - T_REF
    result     = [str(r) for r in race.get('finishing_positions', [])]
    if not result: return None

    drivers = {}
    for pos_key, strat in race.get('strategies', {}).items():
        did   = strat.get('driver_id', pos_key)
        start = strat.get('starting_tire', 'MEDIUM')
        pits  = strat.get('pit_stops', [])
        try: gp = int(str(pos_key).replace('pos',''))
        except: gp = 99
        stints = parse_stints(start, pits, total_laps)
        drivers[did] = {'stints':stints,'fv':make_fv(stints,dT),'grid':gp}

    return {'base_lt':base_lt,'pit_time':pit_time,'track_temp':track_temp,
            'dT':dT,'result':result,'drivers':drivers}

def load_races(data_dir, max_races=30000):
    files = sorted(glob.glob(os.path.join(data_dir,'historical_races','*.json')))
    races = []
    for f in files:
        if len(races)>=max_races: break
        try:
            with open(f) as fh: raw=json.load(fh)
            races.extend(raw if isinstance(raw,list) else [raw])
        except: pass
    return races[:max_races]

# ── constraint builder ────────────────────────────────────────────────────────

def build_constraints(parsed, nearby=3):
    """Extract ordering constraints from parsed races."""
    rows = []
    for r in parsed:
        result, drivers = r['result'], r['drivers']
        sigs = {d:tuple(dd['stints']) for d,dd in drivers.items()}
        for i in range(len(result)):
            for j in range(i+1, min(i+nearby+1, len(result))):
                wi,li = result[i],result[j]
                if wi not in drivers or li not in drivers: continue
                if sigs.get(wi)==sigs.get(li): continue
                rows.append(drivers[wi]['fv'] - drivers[li]['fv'])
    return np.array(rows) if rows else np.zeros((0,12))

# ── LP solver ─────────────────────────────────────────────────────────────────

def run_lp(A, time_limit=30):
    if len(A)==0: return None, -9999
    # Deduplicate
    A = np.unique(np.round(A,6), axis=0)
    A_aug = np.hstack([A, np.ones((len(A),1))])
    b = np.zeros(len(A))
    c_obj = np.zeros(13); c_obj[-1]=-1.0
    bounds = list(BOUNDS)+[(None,None)]
    try:
        res = linprog(c_obj, A_ub=A_aug, b_ub=b, bounds=bounds, method='highs',
                      options={'time_limit':time_limit,'primal_feasibility_tolerance':1e-8})
        eps = -res.fun if res.success else -9999
        return (res.x[:12] if res.success else None), eps
    except: return None, -9999

# ── local NM polish ───────────────────────────────────────────────────────────

def pairwise_loss(x, A, margin=0.01):
    """Fast loss using precomputed constraint matrix."""
    if len(A)==0: return 0.0
    scores = A @ x  # shape (n_constraints,)
    violations = np.maximum(0.0, scores + margin)
    return float(np.mean(violations**2))

def nm_polish(x0, A, margin=0.005, maxiter=20000):
    res = minimize(pairwise_loss, x0, args=(A, margin), method='Nelder-Mead',
                   options={'maxiter':maxiter,'xatol':1e-10,'fatol':1e-10,'adaptive':True})
    return res.x, res.fun

# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(parsed, x, n_eval=500):
    params = dict(zip(NAMES, x))
    exact = pos_ok = total_pos = 0
    for r in parsed[:n_eval]:
        result,drivers = r['result'],r['drivers']
        tl = [(round(driver_time(d['stints'],r['base_lt'],r['track_temp'],
                                  r['pit_time'],params), TIME_ROUND),
               d['grid'], did) for did,d in drivers.items()]
        tl.sort(key=lambda z:(z[0],z[1]))
        pred = [z[2] for z in tl]
        if pred==result: exact+=1
        pos_ok    += sum(1 for p,e in zip(pred,result) if p==e)
        total_pos += len(result)
    n = min(n_eval, len(parsed))
    pct = 100*exact/n
    pa  = 100*pos_ok/total_pos if total_pos else 0
    return exact, n, pct, pa

# ── CHUNKED FITTER ────────────────────────────────────────────────────────────

def fit_chunked(data_dir, chunk_size=1000, max_races=30000,
                output='solution/params.json', lp_only=False):

    print("="*60)
    print("Box Box Box — Chunked Parameter Fitter")
    print("="*60)

    # Load & parse all races
    print(f"\nLoading races...", flush=True)
    raw = load_races(data_dir, max_races)
    print(f"  Loaded {len(raw)} raw races")

    print(f"  Parsing...", flush=True)
    parsed = []
    for race in raw:
        try:
            p = parse_race(race)
            if p: parsed.append(p)
        except: pass
    print(f"  Parsed {len(parsed)} races")

    # Load existing best params as starting point
    best_x = np.array([DEFAULT_PARAMS.get(n,0.0) for n in NAMES])
    best_params_path = output
    if os.path.exists(best_params_path):
        try:
            with open(best_params_path) as f:
                saved = json.load(f)
            best_x = np.array([saved.get(n, DEFAULT_PARAMS.get(n,0.0)) for n in NAMES])
            print(f"\n  Loaded existing params from {best_params_path}")
        except: pass

    # Evaluate starting params
    exact,n,pct,pa = evaluate(parsed, best_x, n_eval=min(500, len(parsed)))
    print(f"\n  Starting params: {exact}/{n} exact ({pct:.1f}%)  pos={pa:.1f}%")

    # ── Chunk processing ──────────────────────────────────────────────
    n_chunks = (len(parsed) + chunk_size - 1) // chunk_size
    print(f"\n  Processing {n_chunks} chunks of ~{chunk_size} races each")
    print(f"  Mode: {'LP only' if lp_only else 'LP + NM polish'}")
    print("="*60)

    all_constraints = []
    chunk_x_list = []

    for chunk_idx in range(n_chunks):
        t0 = time.time()
        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, len(parsed))
        chunk = parsed[start:end]

        # Build constraints for this chunk
        A_chunk = build_constraints(chunk, nearby=3)
        all_constraints.append(A_chunk)

        # LP on this chunk
        x_lp, eps = run_lp(A_chunk, time_limit=15)

        if x_lp is not None and eps > -50:
            if lp_only:
                x_chunk = x_lp
            else:
                # Quick NM polish on chunk constraints
                x_chunk, loss = nm_polish(x_lp, A_chunk, margin=0.002, maxiter=5000)
            chunk_x_list.append(x_chunk)
            status = f"eps={eps:.2f}"
        else:
            # LP failed on this chunk — use current best
            chunk_x_list.append(best_x)
            status = "LP failed, using current best"

        elapsed = time.time()-t0
        print(f"  Chunk {chunk_idx+1:3d}/{n_chunks}  races {start+1}-{end:5d}  "
              f"{status}  [{elapsed:.1f}s]", flush=True)

        # Every 5 chunks, compute running average and evaluate
        if (chunk_idx+1) % 5 == 0 or chunk_idx == n_chunks-1:
            # Weighted average of chunk solutions
            if chunk_x_list:
                x_avg = np.mean(chunk_x_list, axis=0)
                # Clip to bounds
                for i,(lo,hi) in enumerate(BOUNDS):
                    if lo is not None: x_avg[i] = max(lo, x_avg[i])
                    if hi is not None: x_avg[i] = min(hi, x_avg[i])

                exact,n,pct,pa = evaluate(parsed, x_avg, n_eval=min(500,len(parsed)))
                print(f"  ── Checkpoint: {exact}/{n} ({pct:.1f}%)  pos={pa:.1f}%")

                # Save if better
                if pct >= (100*evaluate(parsed, best_x, n_eval=min(100,len(parsed)))[0]/
                           min(100,len(parsed))) - 1:
                    best_x = x_avg
                    save_params(best_x, output)
                    print(f"  ── Saved checkpoint to {output}")

    # ── Global LP on ALL constraints ──────────────────────────────────
    print(f"\n{'='*60}")
    print("Global LP on all constraints...", flush=True)
    A_all = np.vstack([a for a in all_constraints if len(a)>0])
    print(f"  Total constraints: {len(A_all)}")

    x_global, eps_global = run_lp(A_all, time_limit=60)
    print(f"  Global LP: eps={eps_global:.4f}")

    if x_global is not None and eps_global > -100:
        exact,n,pct,pa = evaluate(parsed, x_global, n_eval=min(1000,len(parsed)))
        print(f"  Global LP accuracy: {exact}/{n} ({pct:.1f}%)  pos={pa:.1f}%")

        # Final NM polish on ALL constraints (capped at 200k)
        if not lp_only:
            print("Final NM polish on global constraints...", flush=True)
            A_cap = A_all[:200000]
            x_final, loss = nm_polish(x_global, A_cap, margin=0.001, maxiter=50000)
            exact2,n2,pct2,pa2 = evaluate(parsed, x_final, n_eval=min(1000,len(parsed)))
            print(f"  After NM: {exact2}/{n2} ({pct2:.1f}%)  pos={pa2:.1f}%")
            if pct2 >= pct:
                x_global = x_final

        exact,n,pct,pa = evaluate(parsed, x_global, n_eval=min(1000,len(parsed)))
        if pct >= (100*evaluate(parsed, best_x, n_eval=min(100,len(parsed)))[0]/
                   min(100,len(parsed))) - 1:
            best_x = x_global

    # ── Final save ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    exact,n,pct,pa = evaluate(parsed, best_x, n_eval=len(parsed))
    print(f"FINAL: {exact}/{n} ({pct:.1f}%)  pos={pa:.1f}%")

    save_params(best_x, output)
    print(f"Saved final params to {output}")

    print("\nFinal parameter values:")
    for name, val in zip(NAMES, best_x):
        print(f"  {name:20s} = {val:.8f}")

    return best_x

def save_params(x, path):
    params = dict(DEFAULT_PARAMS)
    params.update({n:float(v) for n,v in zip(NAMES,x)})
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path,'w') as f: json.dump(params,f,indent=2)

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir',    default='data')
    ap.add_argument('--chunk-size',  type=int, default=1000)
    ap.add_argument('--max-races',   type=int, default=30000)
    ap.add_argument('--output',      default='solution/params.json')
    ap.add_argument('--quick',       action='store_true',
                    help='Fast mode: 2000 races only (~2 min)')
    ap.add_argument('--lp-only',     action='store_true',
                    help='LP only, no NM polish (fastest)')
    args = ap.parse_args()

    if args.quick:
        args.max_races  = 2000
        args.chunk_size = 500

    fit_chunked(args.data_dir, args.chunk_size, args.max_races,
                args.output, args.lp_only)
