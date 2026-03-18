"""
BRUTE FORCE: Try every possible formula variant on the first 5 test cases.
Variants:
1. Pit lap mode: A (lap-1 laps old) vs B (lap laps old)  
2. Formula: linear only / with quadratic / with temperature
3. Parameter signs
4. Whether pit_time is per-pit or total

Key: find which variant gives eps>0 on tests 1-5 SIMULTANEOUSLY.
"""
import json, glob, numpy as np
from scipy.optimize import linprog
import sys; sys.path.insert(0,'solution')
from race_simulator import _c

def sa(n): return n*(n-1)/2
def sa2(n): return n*(n-1)*(2*n-1)/6

def parse_race_generic(inf, expf, pit_mode='B', dT_mult=1.0):
    """Build feature vectors for a test case."""
    with open(inf) as f: race=json.load(f)
    with open(expf) as f: expected=json.load(f)
    cfg=race['race_config']
    TOTAL=int(cfg['total_laps']); TEMP=float(cfg['track_temp'])
    dT=TEMP-30.0
    exp=[str(r) for r in expected['finishing_positions']]
    
    drivers={}
    for pk,s in race['strategies'].items():
        did=s['driver_id']; start=s['starting_tire'].upper()
        pits=s.get('pit_stops',[])
        try: gp=int(pk.replace('pos',''))
        except: gp=99
        
        # Build stints
        events=sorted((int(ps['lap']),ps['to_tire'].upper()) for ps in pits)
        stints=[]; cur=start; used=0
        for lap,new in events:
            n = lap-used if pit_mode=='B' else lap-1-used
            if n>0: stints.append((cur,n))
            cur=new; used = lap if pit_mode=='B' else lap-1
        rem=TOTAL-used
        if rem>0: stints.append((cur,rem))
        
        ci={'SOFT':0,'MEDIUM':1,'HARD':2}
        v=np.zeros(12)
        for c,n in stints:
            c=_c(c); i=ci.get(c,1)
            v[i]+=n; v[3+i]+=sa(n); v[6+i]+=sa2(n); v[9+i]+=n*dT*dT_mult
        n_pits=len(stints)-1
        drivers[did]={'fv':v,'grid':gp,'stints':stints,'n_pits':n_pits}
    
    return exp, drivers

BOUNDS_12=[(-5,0),(-2,3),(0,5),(0,1),(0,.5),(0,.3),(0,.06),(0,.05),(0,.04),(-.4,.4),(-.3,.3),(-.3,.3)]

def lp_test(A, bounds=BOUNDS_12):
    if len(A)==0: return 0.0,None
    A=np.unique(np.round(A,8),axis=0)
    A_aug=np.hstack([A,np.ones((len(A),1))]); b=np.zeros(len(A))
    c_obj=np.zeros(len(bounds)+1); c_obj[-1]=-1.0
    bds=list(bounds)+[(None,None)]
    try:
        res=linprog(c_obj,A_ub=A_aug,b_ub=b,bounds=bds,method='highs',
                    options={'time_limit':5,'primal_feasibility_tolerance':1e-9})
        if res.success: return -res.fun,res.x[:len(bounds)]
    except: pass
    return -9999,None

inps=sorted(glob.glob('data/test_cases/inputs/*.json'))
exps=sorted(glob.glob('data/test_cases/expected_outputs/*.json'))

print("Testing formula variants on ALL 100 test cases:")
print(f"{'Variant':30s} {'Tests feasible':15s} {'Global eps':12s}")
print("-"*60)

best_variant=None; best_eps=-9999; best_n=0

for pit_mode in ['A','B']:
    for dT_mult in [0.0, 1.0, -1.0, 2.0]:
        # Build all constraints
        all_rows=[]
        per_test_eps=[]
        for inf,expf in zip(inps,exps):
            exp,drivers=parse_race_generic(inf,expf,pit_mode,dT_mult)
            sigs={d:tuple(dd['stints']) for d,dd in drivers.items()}
            rows=[]
            for i in range(len(exp)):
                for j in range(i+1,len(exp)):
                    wi,li=exp[i],exp[j]
                    if wi not in drivers or li not in drivers: continue
                    if sigs.get(wi)==sigs.get(li): continue
                    rows.append(drivers[wi]['fv']-drivers[li]['fv'])
            if rows:
                A_t=np.array(rows)
                eps_t,_=lp_test(A_t)
                per_test_eps.append(eps_t>0)
                all_rows.extend(rows)
        
        n_feasible=sum(per_test_eps)
        if not all_rows: continue
        A_all=np.array(all_rows)
        global_eps,theta=lp_test(A_all)
        
        label=f"pit={pit_mode} dT_mult={dT_mult:+.1f}"
        print(f"  {label:30s} {n_feasible:3d}/100 feasible   eps={global_eps:+.4f}")
        
        if global_eps>best_eps or (global_eps==best_eps and n_feasible>best_n):
            best_eps=global_eps; best_n=n_feasible
            best_variant=(pit_mode,dT_mult,theta)

print(f"\nBest variant: pit_mode={best_variant[0]}, dT_mult={best_variant[1]}")
print(f"  eps={best_eps:.4f}  feasible_tests={best_n}")
if best_variant[2] is not None:
    for n,v in zip(['off_S','off_M','off_H','deg_S','deg_M','deg_H',
                    'dq_S','dq_M','dq_H','ts_S','ts_M','ts_H'],best_variant[2]):
        print(f"  {n}: {v:+.6f}")
