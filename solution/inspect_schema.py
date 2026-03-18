#!/usr/bin/env python3
"""
Run this to see the EXACT pit_stops structure and test input format.
  python solution/inspect_schema.py
"""
import json, glob, os, sys

def show(label, val, indent=2):
    prefix = " " * indent
    print(f"{prefix}{label}: {json.dumps(val, indent=indent+2) if isinstance(val, (dict,list)) else repr(val)}")

# ── Historical race - deep dive ───────────────────────────────────
hist_files = sorted(glob.glob('data/historical_races/*.json'))
if hist_files:
    with open(hist_files[0]) as f:
        raw = json.load(f)
    races = raw if isinstance(raw, list) else [raw]
    r = races[0]

    print("=" * 65)
    print("HISTORICAL RACE — race_config")
    print("=" * 65)
    print(json.dumps(r['race_config'], indent=2))

    print("\n" + "=" * 65)
    print("HISTORICAL RACE — strategies (pos1 and pos2 in full)")
    print("=" * 65)
    strats = r.get('strategies', {})
    for k in list(strats.keys())[:3]:
        print(f"\n--- {k} ---")
        print(json.dumps(strats[k], indent=2))

    print("\n" + "=" * 65)
    print("HISTORICAL RACE — finishing_positions")
    print("=" * 65)
    print(r.get('finishing_positions', []))

    # Find a driver WITH pit stops
    print("\n" + "=" * 65)
    print("STRATEGIES WITH PIT STOPS (first 3 found)")
    print("=" * 65)
    found = 0
    for k, v in strats.items():
        if v.get('pit_stops'):
            print(f"\n--- {k} ---")
            print(json.dumps(v, indent=2))
            found += 1
            if found >= 3:
                break
    if found == 0:
        print("No pit stops found in first race — checking race[1]...")
        if len(races) > 1:
            r2 = races[1]
            strats2 = r2.get('strategies', {})
            for k, v in strats2.items():
                if v.get('pit_stops'):
                    print(f"\n{k}: {json.dumps(v, indent=2)}")
                    found += 1
                    if found >= 3: break

# ── Test case input ───────────────────────────────────────────────
test_files = sorted(glob.glob('data/test_cases/inputs/*.json'))
if test_files:
    print("\n" + "=" * 65)
    print("TEST CASE INPUT (test_001)")
    print("=" * 65)
    with open(test_files[0]) as f:
        t = json.load(f)
    print(json.dumps(t, indent=2)[:3000])

    # Show strategies with pit stops
    strats = t.get('strategies', {})
    print("\n--- Strategies with pit_stops ---")
    for k, v in strats.items():
        if v.get('pit_stops'):
            print(f"{k}: {json.dumps(v)}")

test_exp = sorted(glob.glob('data/test_cases/expected_outputs/*.json'))
if test_exp:
    print("\n" + "=" * 65)
    print("TEST CASE EXPECTED OUTPUT (test_001)")
    print("=" * 65)
    with open(test_exp[0]) as f:
        e = json.load(f)
    print(json.dumps(e, indent=2))
