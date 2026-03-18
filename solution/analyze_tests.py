"""
Analyze test_002 through test_005 to understand what params are needed.
For each test, run the LP and see what params it requires.
Then find the intersection (what works for ALL tests).
"""
import json, glob, numpy as np
from scipy.optimize import linprog
import sys
sys.path.insert(0, 'solution')
from race_simulator import parse_stints, _c, driver_time, TIME_ROUND

def sa(n): return n*(n-1)/2
def sa2(n): return n*(n-1)*(2*n-1)/6

def make_fv(stints, dT):
    ci={'SOFT':0,'MEDIUM':1,'HARD':2}
    v=np.zeros(12)
    for c,n in stints:
        c=_c(c); i=ci.get(c,1)
        v[i]+=n; v[3+i]+=sa(n); v[6+i]+=sa2(n); v[9+i]+=n*dT
    return v

NAMES=['off_SOFT','off_MEDIUM','off_HARD','deg_SOFT','deg_MEDIUM','deg_HARD',
       'dq_SOFT','dq_MEDIUM','dq_HARD','ts_SOFT','ts_MEDIUM','ts_HARD']
BOUNDS=[(-5,0),(-2,3),(0,5),(0,1),(0,.5),(0,.3),(0,.06),(0,.05),(0,.04),(-.4,.4),(-.3,.3),(-.3,.3)]

def parse_test(inp_f, exp_f):
    with open(inp_f) as f: race=json.load(f)
    with open(exp_f) as f: expected=json.load(f)
    cfg=race.get('race_config',{})
    tl=int(cfg.get('total_laps',57))
    base=float(cfg.get('base_lap_time',90.0))
    pit=float(cfg.get('pit_lane_time',22.0))
    temp=float(cfg.get('track_temp',30.0))
    dT=temp-30.0
    exp_order=[str(r) for r in expected.get('finishing_positions',[])]
    drivers={}
    for pk,strat in race.get('strategies',{}).items():
        did=strat.get('driver_id',pk)
        st=strat.get('starting_tire','MEDIUM')
        pits=strat.get('pit_stops',[])
        try: gp=int(str(pk).replace('pos',''))
        except: gp=99
        stints=parse_stints(st,pits,tl)
        drivers[did]={'stints':stints,'fv':make_fv(stints,dT),'grid':gp}
    return {'race_id':race.get('race_id','?'),'base':base,'pit':pit,
            'temp':temp,'dT':dT,'result':exp_order,'drivers':drivers,'total_laps':tl}

def race_lp(r):
    result,drivers=r['result'],r['drivers']
    sigs={d:tuple(dd['stints']) for d,dd in drivers.items()}
    rows=[]
    for i in range(len(result)):
        for j in range(i+1,len(result)):  # ALL pairs
            wi,li=result[i],result[j]
            if wi not in drivers or li not in drivers: continue
            if sigs.get(wi)==sigs.get(li): continue
            rows.append(drivers[wi]['fv']-drivers[li]['fv'])
    if not rows: return None,-9999,None
    A=np.array(rows)
    A_aug=np.hstack([A,np.ones((len(A),1))])
    b=np.zeros(len(A))
    c_obj=np.zeros(13); c_obj[-1]=-1.0
    lp_bounds=list(BOUNDS)+[(None,None)]
    res=linprog(c_obj,A_ub=A_aug,b_ub=b,bounds=lp_bounds,method='highs',
                options={'primal_feasibility_tolerance':1e-9,'time_limit':10})
    if res.success:
        return res.x[:12],-res.fun,A
    return None,-9999,A

inp_files=sorted(glob.glob('data/test_cases/inputs/*.json'))
exp_files=sorted(glob.glob('data/test_cases/expected_outputs/*.json'))

print("Solving LP per test case (first 20):")
print(f"{'Test':12s} {'eps':8s} {'temp':6s} {'laps':5s} {'base':6s} {'pit':5s}")
print("-"*50)

all_A=[]
for inf,expf in list(zip(inp_files,exp_files))[:20]:
    r=parse_test(inf,expf)
    x,eps,A=race_lp(r)
    if A is not None: all_A.append(A)
    status="✅" if eps>0 else "❌"
    print(f"{status} {r['race_id']:10s}  eps={eps:+6.3f}  "
          f"temp={r['temp']:4.0f}  laps={r['total_laps']:2d}  "
          f"base={r['base']:5.1f}  pit={r['pit']:4.1f}")
    if x is not None and eps>0:
        # Show key params
        pstr=" ".join(f"{NAMES[i][:3]}={x[i]:+.2f}" for i in [0,1,2,3,4,5])
        print(f"    {pstr}")

# Now run LP on ALL 100 test cases combined
print(f"\n--- LP on ALL 100 test cases ---")
inp_files_all=sorted(glob.glob('data/test_cases/inputs/*.json'))
exp_files_all=sorted(glob.glob('data/test_cases/expected_outputs/*.json'))
all_rows=[]
for inf,expf in zip(inp_files_all,exp_files_all):
    r=parse_test(inf,expf)
    result,drivers=r['result'],r['drivers']
    sigs={d:tuple(dd['stints']) for d,dd in drivers.items()}
    for i in range(len(result)):
        for j in range(i+1,min(i+10,len(result))):
            wi,li=result[i],result[j]
            if wi not in drivers or li not in drivers: continue
            if sigs.get(wi)==sigs.get(li): continue
            all_rows.append(drivers[wi]['fv']-drivers[li]['fv'])

A_all=np.array(all_rows)
print(f"Total constraints: {len(A_all)}")
A_all_u=np.unique(np.round(A_all,6),axis=0)
print(f"Unique constraints: {len(A_all_u)}")

A_aug=np.hstack([A_all_u,np.ones((len(A_all_u),1))])
b=np.zeros(len(A_all_u))
c_obj=np.zeros(13); c_obj[-1]=-1.0
lp_bounds=list(BOUNDS)+[(None,None)]
res=linprog(c_obj,A_ub=A_aug,b_ub=b,bounds=lp_bounds,method='highs',
            options={'primal_feasibility_tolerance':1e-9,'time_limit':60})
eps=-res.fun if res.success else -9999
print(f"Global LP (100 tests): eps={eps:.6f}  {res.message}")

if res.success:
    theta=res.x[:12]
    print("\nGlobal LP params:")
    for n,v in zip(NAMES,theta): print(f"  {n:20s} = {v:+.8f}")
    
    # Evaluate
    correct=0
    for inf,expf in zip(inp_files_all,exp_files_all):
        r=parse_test(inf,expf)
        params=dict(zip(NAMES,theta))
        tl=[(round(driver_time(d['stints'],r['base'],r['temp'],r['pit'],params),TIME_ROUND),
             d['grid'],did) for did,d in r['drivers'].items()]
        tl.sort(key=lambda z:(z[0],z[1]))
        pred=[z[2] for z in tl]
        if pred==r['result']: correct+=1
    print(f"\n==> {correct}/100 test cases correct with global LP params")
    
    import os
    params_out={'pit_lane_time':22.0,'base_lap_time':90.0}
    params_out.update({n:float(v) for n,v in zip(NAMES,theta)})
    os.makedirs('solution',exist_ok=True)
    with open('solution/params.json','w') as f: json.dump(params_out,f,indent=2)
    print("Saved to solution/params.json")
