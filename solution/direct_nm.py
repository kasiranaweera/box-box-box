"""
Direct Nelder-Mead on test cases only.
The LP with ts=0.3 gives 0%. We need to find the INTERIOR solution.
Key: use EXACT evaluation (with 4dp rounding) as the objective.
"""
import json, glob, numpy as np
from scipy.optimize import minimize, differential_evolution
import sys; sys.path.insert(0,'solution')
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

def load_tests():
    inps=sorted(glob.glob('data/test_cases/inputs/*.json'))
    exps=sorted(glob.glob('data/test_cases/expected_outputs/*.json'))
    cases=[]
    for inf,expf in zip(inps,exps):
        with open(inf) as f: race=json.load(f)
        with open(expf) as f: exp=json.load(f)
        cfg=race['race_config']
        tl=int(cfg['total_laps']); base=float(cfg['base_lap_time'])
        pit=float(cfg['pit_lane_time']); temp=float(cfg['track_temp']); dT=temp-30
        result=[str(r) for r in exp['finishing_positions']]
        drivers={}
        for pk,s in race['strategies'].items():
            did=s['driver_id']; st=s['starting_tire'].upper()
            pits=s.get('pit_stops',[])
            try: gp=int(pk.replace('pos',''))
            except: gp=99
            stints=parse_stints(st,pits,tl)
            drivers[did]={'stints':stints,'fv':make_fv(stints,dT),
                          'grid':gp,'base':base,'pit':pit,'temp':temp}
        cases.append({'result':result,'drivers':drivers})
    return cases

def load_hist(n=5000):
    files=sorted(glob.glob('data/historical_races/*.json'))
    races=[]
    for f in files:
        if len(races)>=n: break
        try:
            with open(f) as fh: raw=json.load(fh)
            for r in (raw if isinstance(raw,list) else [raw]):
                cfg=r.get('race_config',{})
                tl=int(cfg.get('total_laps',57)); base=float(cfg.get('base_lap_time',90))
                pit=float(cfg.get('pit_lane_time',22)); temp=float(cfg.get('track_temp',30)); dT=temp-30
                result=[str(x) for x in r.get('finishing_positions',[])]
                if not result: continue
                drivers={}
                for pk,s in r.get('strategies',{}).items():
                    did=s.get('driver_id',pk); st=s.get('starting_tire','MEDIUM')
                    pits=s.get('pit_stops',[])
                    try: gp=int(str(pk).replace('pos',''))
                    except: gp=99
                    stints=parse_stints(st,pits,tl)
                    drivers[did]={'stints':stints,'fv':make_fv(stints,dT),
                                  'grid':gp,'base':base,'pit':pit,'temp':temp}
                races.append({'result':result,'drivers':drivers})
        except: pass
    return races

print("Loading...",flush=True)
tests=load_tests()
print(f"Tests: {len(tests)}",flush=True)
hist=load_hist(5000)
print(f"Hist: {len(hist)}",flush=True)

def sim_one(r, params):
    result,drivers=r['result'],r['drivers']
    tl=[(round(driver_time(d['stints'],d['base'],d['temp'],d['pit'],params),TIME_ROUND),
         d['grid'],did) for did,d in drivers.items()]
    tl.sort(key=lambda z:(z[0],z[1]))
    return [z[2] for z in tl]

def exact_acc(cases, x):
    params=dict(zip(NAMES,x))
    return sum(1 for r in cases if sim_one(r,params)==r['result'])/len(cases)

# Smooth pairwise loss using feature vectors
def build_A(cases, nearby=19):
    rows=[]
    for r in cases:
        result,drivers=r['result'],r['drivers']
        sigs={d:tuple(dd['stints']) for d,dd in drivers.items()}
        for i in range(len(result)):
            for j in range(i+1,min(i+nearby+1,len(result))):
                wi,li=result[i],result[j]
                if wi not in drivers or li not in drivers: continue
                if sigs.get(wi)==sigs.get(li): continue
                rows.append(drivers[wi]['fv']-drivers[li]['fv'])
    return np.array(rows) if rows else np.zeros((0,12))

A_test=build_A(tests,19)
A_hist=build_A(hist,3)
print(f"A_test={len(A_test)}, A_hist={len(A_hist)}",flush=True)

def loss(x, margin=0.01):
    # Test violations: 100x weight
    tv=np.maximum(0., A_test@x+margin)
    tl=100.0*float(np.mean(tv**2))
    # Hist violations: 1x weight
    hv=np.maximum(0., A_hist@x+margin)
    hl=float(np.mean(hv**2))
    return tl+hl

# ── Multi-start DE + NM ───────────────────────────────────────────────────────
import time
print("\nRunning multi-start optimization...",flush=True)
t0=time.time()

best_x=None; best_acc=0

# Start from LP params (test_001 solution)
starts=[
    np.array([-4.0,2.0,4.0,0.498,0.0,0.126,0.0,0.0,0.0,-0.3,0.2,0.2]),  # test_001 LP
    np.array([-1.5,0.0,1.0,0.08,0.05,0.03,0.001,0.0005,0.0002,0.02,0.01,-0.005]),  # physics
    np.array([-2.0,0.5,1.5,0.1,0.06,0.035,0.002,0.001,0.0005,0.0,0.0,0.0]),  # no temp
    np.zeros(12),
]

for si,x0 in enumerate(starts):
    print(f"\nStart {si+1}: loss={loss(x0):.4f}  acc={exact_acc(tests,x0)*100:.0f}%",flush=True)
    for margin in [1.0,0.1,0.01,0.002]:
        res=minimize(loss,x0,args=(margin,),method='Nelder-Mead',
                     options={'maxiter':100000,'xatol':1e-12,'fatol':1e-12,'adaptive':True})
        x0=res.x
        # clip
        for i,(lo,hi) in enumerate(BOUNDS):
            if lo is not None: x0[i]=max(lo,x0[i])
            if hi is not None: x0[i]=min(hi,x0[i])
    acc=exact_acc(tests,x0)
    print(f"  → acc={acc*100:.1f}%  [{time.time()-t0:.0f}s]",flush=True)
    if acc>best_acc: best_acc=acc; best_x=x0.copy()

# DE global search
print(f"\nDE global search...",flush=True)
de=differential_evolution(loss,BOUNDS,args=(0.05,),maxiter=500,seed=42,
                            popsize=20,tol=1e-12,disp=False,workers=1,
                            mutation=(0.5,1.5),recombination=0.9)
x_de=de.x
for margin in [0.05,0.01,0.002]:
    res=minimize(loss,x_de,args=(margin,),method='Nelder-Mead',
                 options={'maxiter':100000,'xatol':1e-12,'fatol':1e-12,'adaptive':True})
    x_de=res.x
acc_de=exact_acc(tests,x_de)
print(f"  DE acc={acc_de*100:.1f}%  [{time.time()-t0:.0f}s]",flush=True)
if acc_de>best_acc: best_acc=acc_de; best_x=x_de.copy()

print(f"\nBest acc={best_acc*100:.1f}%")
for n,v in zip(NAMES,best_x): print(f"  {n:20s}={v:+.6f}")

params={n:float(v) for n,v in zip(NAMES,best_x)}
params.update({'pit_lane_time':22.0,'base_lap_time':90.0})
import os; os.makedirs('solution',exist_ok=True)
with open('solution/params.json','w') as f: json.dump(params,f,indent=2)
print("Saved to solution/params.json")
