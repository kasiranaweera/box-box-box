"""
Find the EXACT contradiction in test cases.
TEST_001 is feasible (eps>0 individually but eps=0 globally).
We need to find which pairs of tests CONTRADICT each other.
"""
import json, glob, numpy as np
from scipy.optimize import linprog
import sys
sys.path.insert(0, 'solution')
from race_simulator import parse_stints, _c, driver_time, TIME_ROUND

def sa(n): return n*(n-1)/2
def sa2(n): return n*(n-1)*(2*n-1)/6
def make_fv(stints, dT):
    ci={'SOFT':0,'MEDIUM':1,'HARD':2}; v=np.zeros(12)
    for c,n in stints:
        c=_c(c); i=ci.get(c,1)
        v[i]+=n; v[3+i]+=sa(n); v[6+i]+=sa2(n); v[9+i]+=n*dT
    return v

NAMES=['off_SOFT','off_MEDIUM','off_HARD','deg_SOFT','deg_MEDIUM','deg_HARD',
       'dq_SOFT','dq_MEDIUM','dq_HARD','ts_SOFT','ts_MEDIUM','ts_HARD']
BOUNDS=[(-5,0),(-2,3),(0,5),(0,1),(0,.5),(0,.3),(0,.06),(0,.05),(0,.04),(-.4,.4),(-.3,.3),(-.3,.3)]

def parse_test(inf, expf):
    with open(inf) as f: race=json.load(f)
    with open(expf) as f: expected=json.load(f)
    cfg=race.get('race_config',{})
    tl=int(cfg.get('total_laps',57)); base=float(cfg.get('base_lap_time',90.0))
    pit=float(cfg.get('pit_lane_time',22.0)); temp=float(cfg.get('track_temp',30.0))
    dT=temp-30.0
    exp_order=[str(r) for r in expected.get('finishing_positions',[])]
    drivers={}
    for pk,strat in race.get('strategies',{}).items():
        did=strat.get('driver_id',pk); st=strat.get('starting_tire','MEDIUM')
        pits=strat.get('pit_stops',[])
        try: gp=int(str(pk).replace('pos',''))
        except: gp=99
        stints=parse_stints(st,pits,tl)
        drivers[did]={'stints':stints,'fv':make_fv(stints,dT),'grid':gp}
    return {'race_id':race.get('race_id','?'),'base':base,'pit':pit,'temp':temp,
            'dT':dT,'result':exp_order,'drivers':drivers,'total_laps':tl}

def get_constraints(r):
    result,drivers=r['result'],r['drivers']
    sigs={d:tuple(dd['stints']) for d,dd in drivers.items()}
    rows=[]
    for i in range(len(result)):
        for j in range(i+1,len(result)):
            wi,li=result[i],result[j]
            if wi not in drivers or li not in drivers: continue
            if sigs.get(wi)==sigs.get(li): continue
            rows.append(drivers[wi]['fv']-drivers[li]['fv'])
    return np.array(rows) if rows else np.zeros((0,12))

def lp_eps(A):
    if len(A)==0: return 0.0
    A=np.unique(np.round(A,8),axis=0)
    A_aug=np.hstack([A,np.ones((len(A),1))]); b=np.zeros(len(A))
    c_obj=np.zeros(13); c_obj[-1]=-1.0
    bounds=list(BOUNDS)+[(None,None)]
    try:
        res=linprog(c_obj,A_ub=A_aug,b_ub=b,bounds=bounds,method='highs',
                    options={'primal_feasibility_tolerance':1e-10,'time_limit':10})
        return -res.fun if res.success else -9999
    except: return -9999

inps=sorted(glob.glob('data/test_cases/inputs/*.json'))
exps=sorted(glob.glob('data/test_cases/expected_outputs/*.json'))
tests=[parse_test(i,e) for i,e in zip(inps,exps)]
constraints=[get_constraints(r) for r in tests]

print("Finding first contradicting pair...")
# Binary search: find smallest subset that is infeasible
# Start: check test_001 alone, then 001+002, etc.
cumulative=constraints[0].copy()
for i in range(1,len(tests)):
    A_new=np.vstack([cumulative,constraints[i]]) if len(constraints[i])>0 else cumulative
    eps=lp_eps(A_new)
    if eps<-0.001:
        print(f"\n*** Adding test_{i+1:03d} makes it INFEASIBLE (eps={eps:.4f}) ***")
        print(f"Test {i+1}: {tests[i]['race_id']}  temp={tests[i]['temp']}  laps={tests[i]['total_laps']}")
        # Find the specific contradicting constraint
        print(f"\nLooking for contradiction...")
        # Try: which constraint in test_{i+1} contradicts test_001..test_i?
        for j,row in enumerate(constraints[i]):
            A_try=np.vstack([cumulative,[row]])
            e=lp_eps(A_try)
            if e<-0.001:
                # Find which race this row came from
                r=tests[i]; result=r['result']; drivers=r['drivers']
                sigs={d:tuple(dd['stints']) for d,dd in drivers.items()}
                pair_idx=0
                for ii in range(len(result)):
                    for jj in range(ii+1,len(result)):
                        wi,li=result[ii],result[jj]
                        if wi not in drivers or li not in drivers: continue
                        if sigs.get(wi)==sigs.get(li): continue
                        if pair_idx==j:
                            wd=drivers[wi]; ld=drivers[li]
                            print(f"\nContradicting constraint: {wi}(grid{wd['grid']}) < {li}(grid{ld['grid']})")
                            print(f"  {wi} stints: {wd['stints']}")
                            print(f"  {li} stints: {ld['stints']}")
                            print(f"  Feature diff: {row}")
                            # Which earlier constraint contradicts it?
                            for k,prev_row in enumerate(cumulative):
                                if np.dot(row,prev_row)<0 and abs(np.dot(row,prev_row))>0.1:
                                    # These two rows point in opposite directions
                                    print(f"\n  Contradicts constraint {k} from earlier tests")
                            break
                        pair_idx+=1
                break
        break
    cumulative=A_new if len(A_new)>0 else cumulative
    print(f"  After test_{i+1:03d}: eps={eps:.4f}  ✅",flush=True)
else:
    print("All tests are jointly feasible!")
    
print("\nDone")
