#!/usr/bin/env python3
"""Box Box Box — Local Test Runner"""
import json, os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from race_simulator import simulate_race, load_params

def run_all(test_dir, params=None, verbose=False, n=None):
    if params is None: params = load_params()
    inp_files = sorted(glob.glob(os.path.join(test_dir, 'inputs', '*.json')))
    if not inp_files: inp_files = sorted(glob.glob(os.path.join(test_dir, '*.json')))
    if n: inp_files = inp_files[:n]
    if not inp_files: print(f"No test files in {test_dir}"); return 0, 0

    passed = total = 0
    pos_sum = 0.0
    print(f"Running {len(inp_files)} tests...")
    print("=" * 60)

    for inp in inp_files:
        base    = os.path.basename(inp)
        test_id = base.replace('.json', '')
        exp     = None
        for c in [os.path.join(test_dir, 'expected_outputs', base),
                  os.path.join(test_dir, 'outputs', base),
                  inp.replace('/inputs/', '/expected_outputs/')]:
            if os.path.exists(c): exp = c; break
        if not exp: print(f"  [SKIP] {test_id}"); continue

        try:
            with open(inp) as f: race = json.load(f)
            if isinstance(race, list): race = race[0]
            with open(exp) as f: expected = json.load(f)
        except Exception as e:
            print(f"  [ERROR] {test_id}: {e}"); continue

        exp_order = (expected.get('finishing_positions') or
                     expected.get('finishing_order') or [])
        if exp_order and isinstance(exp_order[0], dict):
            exp_order = [r.get('driver_id', r.get('id')) for r in exp_order]
        exp_order = [str(r) for r in exp_order]

        try: pred = simulate_race(race, params)
        except Exception as e:
            print(f"  [ERROR] {test_id}: {e}"); continue

        ok   = (pred == exp_order)
        total += 1
        if ok: passed += 1
        pc = sum(1 for p, e in zip(pred, exp_order) if p == e) / max(len(exp_order), 1)
        pos_sum += pc

        status = "✅" if ok else "❌"
        if verbose or not ok:
            print(f"\n{status} {test_id}  pos={pc*100:.1f}%")
            if not ok:
                for i in range(max(len(pred), len(exp_order))):
                    p = pred[i] if i < len(pred) else '---'
                    e = exp_order[i] if i < len(exp_order) else '---'
                    m = "  " if p == e else "!!"
                    print(f"  {m} {i+1:2d}: pred={p:6s}  exp={e:6s}")
        else:
            print(f"  {status} {test_id}  ({pc*100:.1f}%)")

    print("\n" + "=" * 60)
    if total:
        print(f"SCORE: {passed}/{total} ({100*passed/total:.1f}%)")
        print(f"Avg pos accuracy: {100*pos_sum/total:.1f}%")
    print("=" * 60)
    return passed, total

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-dir', default='data/test_cases')
    ap.add_argument('--params', default=None)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--n', type=int, default=None)
    args = ap.parse_args()
    run_all(args.test_dir, load_params(args.params), args.verbose, args.n)