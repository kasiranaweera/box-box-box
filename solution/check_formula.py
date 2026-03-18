"""
CRITICAL CHECK: Verify our formula is even correct.
test_001 is feasible (eps>0) BUT the 20/20 result only works with 4dp rounding.
Maybe the formula itself is wrong and we need a different one.

Let's check: what if the formula uses a DIFFERENT lap counting convention?
Or what if pit_lane_time is NOT added N times but once total?
Or what if it's not additive at all?

Test: try ALL combinations on test_001:
- Mode A vs Mode B for pit laps
- pit_lane_time added per pit or as fixed overhead
- Check if integer arithmetic matches better
"""
import json, itertools, numpy as np
import sys; sys.path.insert(0,'solution')
from race_simulator import _c

with open('data/test_cases/inputs/test_001.json') as f: race=json.load(f)
with open('data/test_cases/expected_outputs/test_001.json') as f: expected=json.load(f)

cfg=race['race_config']
TOTAL=cfg['total_laps']; BASE=cfg['base_lap_time']
PIT_TIME=cfg['pit_lane_time']; TEMP=cfg['track_temp']
print(f"Config: laps={TOTAL}, base={BASE}, pit={PIT_TIME}, temp={TEMP}")

EXP=[str(r) for r in expected['finishing_positions']]
print(f"Expected winner: {EXP[0]}, runner-up: {EXP[1]}")

# Build strategies
strats={}
for pk,s in race['strategies'].items():
    did=s['driver_id']
    try: gp=int(pk.replace('pos',''))
    except: gp=99
    strats[did]={'start':s['starting_tire'].upper(),
                 'pits':s['pit_stops'],
                 'grid':gp}

# Try all pit lap modes
def get_stints_A(start, pits, total):
    """Mode A: pit_lap-1 laps on old tyre"""
    events=sorted((ps['lap'],ps['to_tire'].upper()) for ps in pits)
    stints=[]; cur=start; used=0
    for lap,new in events:
        n=lap-1-used
        if n>0: stints.append((cur,n))
        cur=new; used=lap-1
    remaining=total-used
    if remaining>0: stints.append((cur,remaining))
    return stints

def get_stints_B(start, pits, total):
    """Mode B: pit_lap laps on old tyre"""
    events=sorted((ps['lap'],ps['to_tire'].upper()) for ps in pits)
    stints=[]; cur=start; used=0
    for lap,new in events:
        n=lap-used
        if n>0: stints.append((cur,n))
        cur=new; used=lap
    remaining=total-used
    if remaining>0: stints.append((cur,remaining))
    return stints

def sa(n): return n*(n-1)/2
def sa2(n): return n*(n-1)*(2*n-1)/6

def race_time(stints, params, base, pit_time, temp, pit_mode='per_pit'):
    dT=temp-30
    off=params['off']; deg=params['deg']; dq=params['dq']; ts=params['ts']
    total=0.0
    for c,n in stints:
        total+=n*(base+off[c]+ts[c]*dT)+deg[c]*sa(n)+dq[c]*sa2(n)
    n_pits=len(stints)-1
    if pit_mode=='per_pit': total+=n_pits*pit_time
    elif pit_mode=='once':  total+=pit_time if n_pits>0 else 0
    elif pit_mode=='none':  pass
    return total

# The key question: what parameters make test_001 give the RIGHT order?
# We know test_001 LP params work. But why don't they generalize?
# 
# HYPOTHESIS: the formula actually uses a CONSTANT lap time (no degradation!)
# and the pit stop ordering is purely based on pit timing + compound speed.
# Let's check if a simpler model works.

# Simplest model: just compound offsets + pit time
# T = n_S * (base+off_S) + n_M * (base+off_M) + n_H * (base+off_H) + n_pits*pit_time
# This collapses to: n_total*base + n_S*off_S + n_M*off_M + n_H*off_H + n_pits*pit_time
# Since n_total is same for all drivers, just need: n_S*off_S + n_M*off_M + n_H*off_H + n_pits*pit

# Check: with this simple model, can we sort test_001?
# All have 1 pit stop, so n_pits=1 for all. Pit time cancels!
# T ∝ n_S*off_S + n_M*off_M + n_H*off_H

print("\n=== TESTING SIMPLEST MODEL (just compound offsets) ===")
for pit_fn in [get_stints_A, get_stints_B]:
    mode_name="A" if pit_fn==get_stints_A else "B"
    scores={}
    for did,s in strats.items():
        stints=pit_fn(s['start'],s['pits'],TOTAL)
        n_cmpd={'SOFT':0,'MEDIUM':0,'HARD':0}
        for c,n in stints: n_cmpd[c]+=n
        scores[did]=(n_cmpd, s['grid'], stints)
    
    # Try: what off_S, off_M, off_H sort test_001 correctly?
    # With just 3 params, use constraint analysis
    from scipy.optimize import linprog
    rows=[]
    for i in range(len(EXP)):
        for j in range(i+1,len(EXP)):
            wi,li=EXP[i],EXP[j]
            if scores[wi][0]==scores[li][0]: continue  # same compound counts
            si=scores[wi][0]; sj=scores[li][0]
            diff=[si['SOFT']-sj['SOFT'], si['MEDIUM']-sj['MEDIUM'], si['HARD']-sj['HARD']]
            rows.append(diff)
    
    if rows:
        A=np.array(rows,dtype=float)
        A_aug=np.hstack([A,np.ones((len(A),1))])
        b=np.zeros(len(A)); c_obj=np.zeros(4); c_obj[-1]=-1.0
        bounds=[(-5,0),(-2,3),(0,5),(None,None)]
        res=linprog(c_obj,A_ub=A_aug,b_ub=b,bounds=bounds,method='highs')
        eps=-res.fun if res.success else -9999
        print(f"  Mode {mode_name} simple model: eps={eps:.4f}")
        if res.success and eps>0:
            off_S,off_M,off_H=res.x[:3]
            print(f"    off_S={off_S:.3f} off_M={off_M:.3f} off_H={off_H:.3f}")

# Now check: DO ALL TEST CASES have 1 pit stop per driver?
print("\n=== PIT STOP COUNT DISTRIBUTION (test cases) ===")
pit_counts={}
for i in range(1,6):
    fn=f'data/test_cases/inputs/test_{i:03d}.json'
    with open(fn) as f: r=json.load(f)
    for pk,s in r['strategies'].items():
        n_pits=len(s.get('pit_stops',[]))
        pit_counts[n_pits]=pit_counts.get(n_pits,0)+1
print(f"Distribution across first 5 tests: {pit_counts}")

# Check all 100
pit_counts_all={}
for fn in sorted(glob('data/test_cases/inputs/*.json') if False else __import__('glob').glob('data/test_cases/inputs/*.json')):
    with open(fn) as f: r=json.load(f)
    for pk,s in r['strategies'].items():
        n=len(s.get('pit_stops',[]))
        pit_counts_all[n]=pit_counts_all.get(n,0)+1
print(f"Distribution across all 100 tests: {pit_counts_all}")
