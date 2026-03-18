#!/usr/bin/env python3
"""
Box Box Box - Parameter Learner
================================
Reverse-engineers the race simulator parameters from 30,000 historical races.

Research basis:
- Tyre degradation is nonlinear (quadratic in age) per ScienceDirect / West 2021
- Temperature affects degradation rate (softer compounds more sensitive)
- Track temperature shifts optimal operating window per-compound
- Grid position is the tiebreaker for equal-time drivers (confirmed 100%)

Formula (per lap):
  lap_time(c, age, T) = base_lap_time
                       + compound_offset[c]
                       + deg_linear[c] * age
                       + deg_quad[c] * age^2
                       + temp_sensitivity[c] * (T - T_ref)

Total time = sum(lap_times over all stints) + n_pits * pit_lane_time

Finishing order: sort by total_time ASC, then grid_position ASC (tiebreaker)
"""

import json
import os
import glob
import sys
import numpy as np
from scipy.optimize import minimize, differential_evolution
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_historical_race(filepath):
    """Load one historical race JSON."""
    with open(filepath) as f:
        return json.load(f)

def load_all_historical(data_dir, max_races=None):
    """Load up to max_races historical races."""
    pattern = os.path.join(data_dir, 'historical_races', '*.json')
    files = sorted(glob.glob(pattern))
    if max_races:
        files = files[:max_races]
    races = []
    for fp in files:
        try:
            races.append(load_historical_race(fp))
        except Exception as e:
            print(f"  [skip] {fp}: {e}", file=sys.stderr)
    print(f"Loaded {len(races)} historical races", file=sys.stderr)
    return races

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
COMPOUND_IDX = {c: i for i, c in enumerate(COMPOUNDS)}

def extract_driver_features(driver_data, race_data):
    """
    Extract (compound, stints, pit_stops, grid_pos) from a driver entry.
    Returns dict with structured strategy info.
    """
    stints = driver_data.get('stints', [])
    grid_pos = driver_data.get('grid_position', driver_data.get('starting_position', 20))
    n_pits = len(stints) - 1  # pits = stints - 1

    # total laps
    total_laps = race_data.get('total_laps', 57)
    track_temp = race_data.get('track_temperature', race_data.get('temperature', 30.0))

    # Assign lap counts to stints
    stint_details = []
    remaining = total_laps
    for i, stint in enumerate(stints):
        compound = stint.get('compound', 'MEDIUM').upper()
        if 'laps' in stint:
            laps = stint['laps']
        elif i == len(stints) - 1:
            # last stint gets remaining laps
            laps = remaining
        else:
            laps = stint.get('lap_end', 0) - stint.get('lap_start', 0)

        remaining -= laps
        stint_details.append({
            'compound': compound,
            'laps': max(1, laps),
        })

    return {
        'grid_pos': grid_pos,
        'n_pits': n_pits,
        'stints': stint_details,
        'total_laps': total_laps,
        'track_temp': track_temp,
    }

def compute_total_time(features, params):
    """
    Compute total race time for a driver given params.

    params = [base_lap_time,           # not used in ranking (same for all)
              off_S, off_M, off_H,     # compound offsets (seconds vs base)
              deg_S, deg_M, deg_H,     # linear degradation rate (s/lap)
              dq_S, dq_M, dq_H,        # quadratic degradation (s/lap^2)
              ts_S, ts_M, ts_H,        # temperature sensitivity (s/degC)
              pit_lane_time]            # pit stop time penalty (s)

    Total Time = sum over stints of:
        sum over laps (age 0..n-1) of:
            base + offset[c] + deg[c]*age + dq[c]*age^2 + ts[c]*(T-T_ref)
    + n_pits * pit_lane_time
    """
    T_REF = 30.0  # reference temperature (degrees C)

    (base,
     off_S, off_M, off_H,
     deg_S, deg_M, deg_H,
     dq_S,  dq_M,  dq_H,
     ts_S,  ts_M,  ts_H,
     pit_lane_time) = params

    offsets = {'SOFT': off_S, 'MEDIUM': off_M, 'HARD': off_H}
    degs    = {'SOFT': deg_S, 'MEDIUM': deg_M, 'HARD': deg_H}
    quads   = {'SOFT': dq_S,  'MEDIUM': dq_M,  'HARD': dq_H}
    temps   = {'SOFT': ts_S,  'MEDIUM': ts_M,  'HARD': ts_H}

    T = features['track_temp']
    total = 0.0

    for stint in features['stints']:
        c = stint['compound']
        if c not in offsets:
            c = 'MEDIUM'  # fallback
        n = stint['laps']
        ages = np.arange(n)  # age 0, 1, ..., n-1

        stint_time = (
            n * (base + offsets[c] + temps[c] * (T - T_REF))
            + degs[c] * ages.sum()
            + quads[c] * (ages ** 2).sum()
        )
        total += stint_time

    total += features['n_pits'] * pit_lane_time
    return total

def rank_drivers(all_features, params):
    """
    Rank all drivers: sort by total_time ASC, then grid_pos ASC (tiebreaker).
    Returns list of driver_ids in finishing order.
    """
    times = []
    for driver_id, feat in all_features.items():
        t = compute_total_time(feat, params)
        times.append((t, feat['grid_pos'], driver_id))

    times.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in times]

# ─────────────────────────────────────────────
# OBJECTIVE: PAIRWISE RANKING ACCURACY
# ─────────────────────────────────────────────

def build_pairwise_constraints(race):
    """
    Extract all (winner_id, loser_id) pairs from a historical race result.
    Returns list of (winner_features, loser_features) pairs.
    """
    result = race.get('result', race.get('finishing_order', []))
    drivers_raw = race.get('drivers', race.get('strategies', {}))
    total_laps = race.get('total_laps', 57)
    track_temp = race.get('track_temperature', race.get('temperature', 30.0))

    # Build feature dict keyed by driver id
    feats = {}
    for d in result:
        driver_id = d if isinstance(d, str) else d.get('driver_id', d.get('id', str(d)))
        if isinstance(drivers_raw, dict):
            d_data = drivers_raw.get(driver_id, {})
        else:
            d_data = next((x for x in drivers_raw
                           if (x.get('driver_id') or x.get('id')) == driver_id), {})

        race_ctx = {'total_laps': total_laps, 'track_temperature': track_temp}
        feats[driver_id] = extract_driver_features(d_data, race_ctx)

    # Build ordered driver list from result
    ordered = []
    for d in result:
        driver_id = d if isinstance(d, str) else d.get('driver_id', d.get('id', str(d)))
        ordered.append(driver_id)

    # Pairwise: all pairs (i, j) where i finishes before j
    pairs = []
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            wi, li = ordered[i], ordered[j]
            if wi in feats and li in feats:
                # Only use pair if feature vectors differ (otherwise tiebreaker by grid)
                # Check if stints are identical
                fi, fj = feats[wi], feats[li]
                if _same_strategy(fi, fj):
                    # Tiebreaker: grid_pos lower = better finish
                    # This SHOULD be satisfied; skip from param fitting
                    continue
                pairs.append((feats[wi], feats[li]))
    return pairs

def _same_strategy(f1, f2):
    """Return True if two drivers have identical compound strategies."""
    s1 = [(s['compound'], s['laps']) for s in f1['stints']]
    s2 = [(s['compound'], s['laps']) for s in f2['stints']]
    return s1 == s2

def pairwise_loss(params, all_pairs, margin=0.01):
    """
    Hinge loss: for each (winner, loser), penalize if winner_time >= loser_time.
    Loss = sum max(0, winner_time - loser_time + margin)
    """
    loss = 0.0
    for (wf, lf) in all_pairs:
        wt = compute_total_time(wf, params)
        lt = compute_total_time(lf, params)
        loss += max(0.0, wt - lt + margin)
    return loss

# ─────────────────────────────────────────────
# SMART INITIALIZATION
# ─────────────────────────────────────────────

def get_smart_initial_params():
    """
    Research-backed initial parameter estimates:
    - Soft: fastest per lap (~0s offset), highest degradation
    - Medium: moderate offset, moderate degradation
    - Hard: slowest per lap (+0.7s offset), lowest degradation
    - Temperature sensitivity: softs most sensitive to heat (overheat faster)
    - Pit lane time: ~22s typical F1 pit stop
    - Quadratic term: nonlinear cliff at high tyre age

    Based on:
    - f1metrics.wordpress.com typical values
    - West 2021 tyre thermodynamic model
    - Sulsters 2016 F1 simulation paper
    """
    return np.array([
        90.0,     # base_lap_time (seconds, ~1:30/lap)
        -1.0,     # off_S (SOFT is fastest)
         0.0,     # off_M (MEDIUM is reference)
         0.7,     # off_H (HARD is slowest)
         0.08,    # deg_S (SOFT degrades fast: ~0.08s/lap)
         0.05,    # deg_M (MEDIUM moderate)
         0.03,    # deg_H (HARD slowest)
         0.002,   # dq_S (SOFT quadratic — cliff at high age)
         0.001,   # dq_M
         0.0005,  # dq_H
         0.02,    # ts_S (SOFT most sensitive to track temp)
         0.01,    # ts_M
         0.005,   # ts_H
        22.0,     # pit_lane_time
    ])

def get_param_bounds():
    """Bounds for optimization."""
    return [
        (60.0, 130.0),   # base_lap_time
        (-3.0, 0.0),     # off_S
        (-1.0, 1.0),     # off_M
        (0.0, 3.0),      # off_H
        (0.01, 0.5),     # deg_S
        (0.005, 0.3),    # deg_M
        (0.001, 0.2),    # deg_H
        (0.0, 0.02),     # dq_S
        (0.0, 0.015),    # dq_M
        (0.0, 0.01),     # dq_H
        (-0.1, 0.1),     # ts_S
        (-0.05, 0.05),   # ts_M
        (-0.03, 0.03),   # ts_H
        (15.0, 35.0),    # pit_lane_time
    ]

# ─────────────────────────────────────────────
# MAIN LEARNING PIPELINE
# ─────────────────────────────────────────────

def learn_parameters(data_dir, max_races=500, output_path='params.json'):
    print("=" * 60)
    print("Box Box Box — Parameter Learner")
    print("=" * 60)

    races = load_all_historical(data_dir, max_races=max_races)
    if not races:
        print("ERROR: No historical races found. Check data_dir.", file=sys.stderr)
        return None

    # Build pairwise constraints from all races
    print(f"Building pairwise constraints...", file=sys.stderr)
    all_pairs = []
    for race in races:
        try:
            pairs = build_pairwise_constraints(race)
            all_pairs.extend(pairs)
        except Exception as e:
            pass

    print(f"Total pairwise constraints: {len(all_pairs)}", file=sys.stderr)

    if len(all_pairs) == 0:
        print("WARNING: No pairs extracted. Using research-backed defaults.", file=sys.stderr)
        params = get_smart_initial_params()
    else:
        # Sample subset for efficiency
        if len(all_pairs) > 10000:
            idx = np.random.choice(len(all_pairs), 10000, replace=False)
            sample_pairs = [all_pairs[i] for i in idx]
        else:
            sample_pairs = all_pairs

        x0 = get_smart_initial_params()
        bounds = get_param_bounds()

        print("Running differential evolution (global optimizer)...", file=sys.stderr)
        result = differential_evolution(
            pairwise_loss,
            bounds,
            args=(sample_pairs, 0.01),
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            disp=False,
            workers=1,
        )
        params = result.x
        print(f"DE loss: {result.fun:.4f}", file=sys.stderr)

        # Fine-tune with L-BFGS-B
        print("Fine-tuning with L-BFGS-B...", file=sys.stderr)
        result2 = minimize(
            pairwise_loss,
            params,
            args=(all_pairs, 0.01),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        if result2.fun < result.fun:
            params = result2.x
            print(f"L-BFGS-B improved loss: {result2.fun:.4f}", file=sys.stderr)

    # Save params
    param_names = [
        'base_lap_time',
        'off_S', 'off_M', 'off_H',
        'deg_S', 'deg_M', 'deg_H',
        'dq_S',  'dq_M',  'dq_H',
        'ts_S',  'ts_M',  'ts_H',
        'pit_lane_time'
    ]
    param_dict = {name: float(v) for name, v in zip(param_names, params)}

    with open(output_path, 'w') as f:
        json.dump(param_dict, f, indent=2)

    print(f"\nLearned Parameters:", file=sys.stderr)
    for k, v in param_dict.items():
        print(f"  {k:20s} = {v:.6f}", file=sys.stderr)
    print(f"\nSaved to: {output_path}", file=sys.stderr)

    return param_dict

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='Path to data/ directory')
    parser.add_argument('--max-races', type=int, default=500)
    parser.add_argument('--output', default='solution/params.json')
    args = parser.parse_args()

    learn_parameters(args.data_dir, args.max_races, args.output)
