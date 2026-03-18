#!/usr/bin/env python3
"""
Box Box Box — Data Explorer & Formula Detector
Robustly handles any JSON schema variant.

Usage:
  python solution/explore_data.py --data-dir data --n 100
  python solution/explore_data.py --single data/historical_races/race_0001.json
  python solution/explore_data.py --data-dir data --lp
"""

import json, os, sys, glob, argparse
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))

# ─── ROBUST LOADER ────────────────────────────────────────────────────────────

def load_races(data_dir, n=500):
    """
    Load races from data/historical_races/*.json
    Each file may be a single race dict OR a list of race dicts.
    Returns flat list of race dicts, up to n races.
    """
    pattern = os.path.join(data_dir, 'historical_races', '*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(data_dir, '*.json')
        files = sorted(glob.glob(pattern))

    races = []
    files_read = 0
    for fp in files:
        if len(races) >= n:
            break
        try:
            with open(fp) as f:
                raw = json.load(f)
            files_read += 1
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        races.append(item)
            elif isinstance(raw, dict):
                races.append(raw)
        except Exception as e:
            print(f"  [skip] {fp}: {e}", file=sys.stderr)

    print(f"Loaded {len(races)} races from {files_read} files")
    return races[:n]

# ─── SCHEMA DETECTION ─────────────────────────────────────────────────────────

def detect_schema(race):
    """Deep print of one race dict's structure."""
    if isinstance(race, list):
        print(f"NOTE: got list of {len(race)} — showing [0]")
        race = race[0]
    if not isinstance(race, dict):
        print(f"ERROR: expected dict, got {type(race).__name__}: {str(race)[:200]}")
        return

    print(f"\nTop-level keys: {list(race.keys())}\n")
    for k, v in race.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            print(f"  {k:30s} = {repr(v)}")
        elif isinstance(v, list):
            print(f"  {k:30s} = list[{len(v)}]")
            if v and isinstance(v[0], dict):
                print(f"    item keys: {list(v[0].keys())}")
                for ik, iv in list(v[0].items())[:8]:
                    print(f"      {ik:26s} = {repr(iv)[:70]}")
            elif v:
                print(f"    items: {str(v[:3])[:120]}")
        elif isinstance(v, dict):
            print(f"  {k:30s} = dict [{len(v)} entries]  keys={list(v.keys())[:5]}")
            # Sample one value
            sk = list(v.keys())[0]
            sv = v[sk]
            if isinstance(sv, dict):
                print(f"    [{repr(sk)}] keys: {list(sv.keys())}")
                for ik, iv in list(sv.items())[:8]:
                    if ik == 'stints' and isinstance(iv, list) and iv:
                        print(f"      stints[0]: {iv[0]}")
                    elif isinstance(iv, (str, int, float, bool)):
                        print(f"      {ik:24s} = {repr(iv)}")
            elif isinstance(sv, list):
                print(f"    [{repr(sk)}] = list[{len(sv)}]  first={str(sv[0])[:80] if sv else ''}")

    result = race.get('result', race.get('finishing_order', race.get('results', None)))
    if result is not None:
        print(f"\n  result: {str(result[:5])[:100]}")

# ─── STATISTICS ───────────────────────────────────────────────────────────────

def get_drivers(race):
    """Return list of driver dicts from any schema variant."""
    raw = race.get('drivers', race.get('strategies', {}))
    if isinstance(raw, dict):
        return list(raw.values()), list(raw.keys())
    elif isinstance(raw, list):
        ids = [d.get('driver_id', d.get('id', str(i))) for i, d in enumerate(raw)]
        return raw, ids
    return [], []

def compound_stats(races):
    counts, lengths = Counter(), defaultdict(list)
    for race in races:
        drivers, _ = get_drivers(race)
        for d in drivers:
            for s in d.get('stints', []):
                c = s.get('compound', 'UNKNOWN').upper()
                laps = s.get('laps', s.get('lap_count', s.get('num_laps', None)))
                counts[c] += 1
                if laps is not None:
                    lengths[c].append(int(laps))

    print(f"\nCompound usage over {len(races)} races:")
    for c in ['SOFT', 'MEDIUM', 'HARD']:
        if c in counts:
            ll = lengths[c]
            if ll:
                print(f"  {c:8s}: {counts[c]:5d} stints | avg {np.mean(ll):.1f} laps "
                      f"| range [{min(ll)}, {max(ll)}]")
            else:
                print(f"  {c:8s}: {counts[c]:5d} stints | NO lap count field")
    other = [k for k in counts if k not in ('SOFT','MEDIUM','HARD')]
    if other:
        print(f"  Other compounds found: {other}")

def temperature_stats(races):
    temps = []
    for race in races:
        for key in ('track_temperature','temperature','track_temp','temp'):
            t = race.get(key)
            if t is not None:
                temps.append(float(t)); break
    if temps:
        print(f"\nTrack temp: min={min(temps):.1f}  max={max(temps):.1f}  "
              f"mean={np.mean(temps):.1f}  std={np.std(temps):.1f}  n={len(temps)}")
    else:
        print("\nNo temperature field found")

def scalar_field_stats(races, *field_names, label=''):
    vals = []
    for race in races:
        for k in field_names:
            v = race.get(k)
            if v is not None:
                vals.append(float(v)); break
    if vals:
        c = Counter(vals)
        print(f"\n{label or field_names[0]}: "
              f"min={min(vals):.2f}  max={max(vals):.2f}  mean={np.mean(vals):.2f}  "
              f"common={dict(c.most_common(4))}")
    else:
        print(f"\n{label or field_names[0]}: NOT found in race data")
    return vals

def strategy_diversity(races):
    strats = set()
    stint_dist = Counter()
    for race in races:
        drivers, _ = get_drivers(race)
        for d in drivers:
            stints = d.get('stints', [])
            strats.add(tuple(s.get('compound','?') for s in stints))
            stint_dist[len(stints)] += 1
    print(f"\nUnique compound sequences: {len(strats)}")
    print(f"Pit stop dist: {dict(sorted(stint_dist.items()))}")
    print(f"Sample strategies: {list(strats)[:8]}")

def tiebreaker_check(races):
    correct = total = 0
    for race in races:
        result = race.get('result', race.get('finishing_order', []))
        if not result: continue
        if isinstance(result[0], dict):
            result = [r.get('driver_id', r.get('id', str(r))) for r in result]
        result = [str(r) for r in result]

        raw_d = race.get('drivers', race.get('strategies', {}))
        if isinstance(raw_d, list):
            drivers = {str(d.get('driver_id', d.get('id', str(i)))): d
                       for i, d in enumerate(raw_d)}
        else:
            drivers = {str(k): v for k, v in raw_d.items()}

        # Group by identical strategy signature
        groups = defaultdict(list)
        for did in result:
            d = drivers.get(did, {})
            stints = d.get('stints', [])
            key = tuple((s.get('compound','?'),
                         s.get('laps', s.get('lap_count', s.get('num_laps', 0))))
                        for s in stints)
            groups[key].append(did)

        for key, group in groups.items():
            if len(group) < 2: continue
            pos   = {did: result.index(did) for did in group if did in result}
            grids = {}
            for did in group:
                d = drivers.get(did, {})
                for gk in ('grid_position','starting_position','grid_pos','grid'):
                    g = d.get(gk)
                    if g is not None:
                        grids[did] = int(g); break

            pairs = [d for d in group if d in pos and d in grids]
            for i in range(len(pairs)):
                for j in range(i+1, len(pairs)):
                    a, b = pairs[i], pairs[j]
                    total += 1
                    if (pos[a] < pos[b]) == (grids[a] < grids[b]):
                        correct += 1

    if total > 0:
        pct = 100 * correct / total
        mark = "✅" if pct > 95 else "⚠️ "
        print(f"\nTiebreaker (grid_position): {correct}/{total} = {pct:.1f}%  {mark}")
    else:
        print("\nTiebreaker: no identical-strategy pairs found")
        print("  (stints may lack 'laps' field — check schema above)")

# ─── LP FEASIBILITY ───────────────────────────────────────────────────────────

def check_lp_feasibility(races, n=20):
    from scipy.optimize import linprog

    def feat(d, T):
        ci = {'SOFT':0,'MEDIUM':1,'HARD':2}
        v = np.zeros(13)
        stints = d.get('stints', [])
        dT = T - 30.0
        for s in stints:
            c = s.get('compound','MEDIUM').upper()
            if c not in ci: c = 'MEDIUM'
            idx = ci[c]
            laps = int(s.get('laps', s.get('lap_count', s.get('num_laps', 1))))
            ages = np.arange(laps)
            v[idx]     += laps
            v[3+idx]   += ages.sum()
            v[6+idx]   += (ages**2).sum()
            v[10+idx]  += laps * dT
        v[9] = len(stints) - 1  # n_pits
        return v

    constraints = []
    ties_skipped = 0
    for race in races[:n]:
        result = race.get('result', race.get('finishing_order', []))
        if not result: continue
        if isinstance(result[0], dict):
            result = [r.get('driver_id', r.get('id')) for r in result]
        result = [str(r) for r in result]

        raw_d = race.get('drivers', race.get('strategies', {}))
        if isinstance(raw_d, list):
            drivers = {str(d.get('driver_id', d.get('id', str(i)))): d
                       for i, d in enumerate(raw_d)}
        else:
            drivers = {str(k): v for k, v in raw_d.items()}

        T = float(race.get('track_temperature', race.get('temperature', 30.0)))

        feats, sigs = {}, {}
        for did in result:
            d = drivers.get(did, {})
            feats[did] = feat(d, T)
            stints = d.get('stints', [])
            sigs[did] = tuple((s.get('compound','?'),
                               s.get('laps', s.get('lap_count', 0)))
                              for s in stints)

        for i in range(len(result)):
            for j in range(i+1, len(result)):
                wi, li = result[i], result[j]
                if wi not in feats or li not in feats: continue
                if sigs.get(wi) == sigs.get(li):
                    ties_skipped += 1; continue
                constraints.append(feats[wi] - feats[li])

    print(f"\nLP: {len(constraints)} ordering constraints "
          f"({ties_skipped} tie-pairs skipped)")
    if not constraints:
        print("  No constraints — skipping LP")
        return

    A = np.array(constraints)
    A_aug = np.hstack([A, np.ones((len(A),1))])
    b = np.zeros(len(A))
    c_obj = np.zeros(14); c_obj[-1] = -1.0
    bounds = [(-3,0),(-1,1),(0,3),(0,.5),(0,.3),(0,.2),
              (0,.02),(0,.015),(0,.01),(15,35),
              (-.1,.1),(-.05,.05),(-.05,.05),(None,None)]
    res = linprog(c_obj, A_ub=A_aug, b_ub=b, bounds=bounds, method='highs')
    eps = -res.fun if res.success else None
    print(f"  Status: {res.message}")
    if res.success:
        print(f"  Max margin (eps): {eps:.4f}")
        if eps > 0:
            print("  ✅ FEASIBLE — formula confirmed!")
            names = ['off_S','off_M','off_H','deg_S','deg_M','deg_H',
                     'dq_S','dq_M','dq_H','pit_time','ts_S','ts_M','ts_H']
            for name, val in zip(names, res.x[:13]):
                print(f"    {name:20s} = {val:.6f}")
        else:
            print(f"  ❌ INFEASIBLE — formula needs more terms")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--single', help='Single file to inspect schema')
    parser.add_argument('--lp', action='store_true')
    args = parser.parse_args()

    if args.single:
        with open(args.single) as f:
            raw = json.load(f)
        print("── Schema Detection ─────────────────────────────────────")
        if isinstance(raw, list):
            print(f"File is a LIST of {len(raw)} entries — showing [0] and [1]")
            detect_schema(raw[0])
            if len(raw) > 1:
                print(f"\n── Entry [1] ────────────────────────────────────────")
                detect_schema(raw[1])
        else:
            detect_schema(raw)
        return

    races = load_races(args.data_dir, args.n)
    if not races:
        print(f"ERROR: No races found in {args.data_dir}/historical_races/")
        return

    print(f"\n{'='*60}")
    print(f"  SCHEMA  (first race entry)")
    print(f"{'='*60}")
    detect_schema(races[0])

    print(f"\n{'='*60}")
    print(f"  STATISTICS  ({len(races)} races)")
    print(f"{'='*60}")
    scalar_field_stats(races, 'total_laps','laps', label='total_laps')
    temperature_stats(races)
    scalar_field_stats(races, 'pit_lane_time','pit_stop_time','pit_time', label='pit_lane_time')
    scalar_field_stats(races, 'base_lap_time','base_time', label='base_lap_time')
    compound_stats(races)
    strategy_diversity(races)
    tiebreaker_check(races)

    if args.lp:
        print(f"\n{'='*60}")
        print(f"  LP FEASIBILITY CHECK")
        print(f"{'='*60}")
        check_lp_feasibility(races, min(30, args.n))

if __name__ == '__main__':
    main()