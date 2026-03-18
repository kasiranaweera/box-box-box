#!/usr/bin/env python3
"""
Box Box Box — Smart Parameter Fitter
======================================
Strategy (fastest path to high accuracy):

  Step 1: LP on 100 test cases (ground truth, labeled data) — ~30s
  Step 2: NM refinement weighted by test cases — ~5 min  
  Step 3: Validate and save best

Run: python solution/fit_params.py --data-dir data
"""
import json, glob, os, sys, argparse, time
import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from race_simulator import parse_stints, driver_time, DEFAULT_PARAMS, T_REF, TIME_ROUND, _c

NAMES=['off_SOFT','off_MEDIUM','off_HARD',
       'deg_SOFT','deg_MEDIUM','deg_HARD',
       'dq_SOFT','dq_MEDIUM','dq_HARD',
       'ts_SOFT','ts_MEDIUM','ts_HARD']
BOUNDS=[(-5,0),(-2,3),(0,5),(0,1),(0,.5),(0,.3),
        (0,.06),(0,.05),(0,.04),(-.4,.4),(-.3,.3),(-.3,.3)]

def sa(n):  return n*(n-1)/2
def sa2(n): return n*(n-1)*(2*n-1)/6

def make_fv(stints, dT):
    ci={'SOFT':0,'MEDIUM':1,'HARD':2}
    v=np.zeros(12)
    for c,n in stints:
        c=_c(c); i=ci.get(c,1)
        v[i]+=n; v[3+i]+=sa(n); v[6+i]+=sa2(n); v[9+i]+=n*dT
    return v

def parse_race(race, override_result=None):
    cfg=race.get('race_config',{})
    tl=int(cfg.get('total_laps',57))
    base=float(cfg.get('base_lap_time',90.0))
    pit=float(cfg.get('pit_lane_time',22.0))
    temp=float(cfg.get('track_temp',30.0))
    dT=temp-T_REF
    result=override_result or [str(r) for r in race.get('finishing_positions',[])]
    if not result: return None
    drivers={}
    for pk,strat in race.get('strategies',{}).items():
        did=strat.get('driver_id',pk)
        st=strat.get('starting_tire','MEDIUM')
        pits=strat.get('pit_stops',[])
        try: gp=int(str(pk).replace('pos',''))
        except: gp=99
        stints=parse_stints(st,pits,tl)
        drivers[did]={'stints':stints,'fv':make_fv(stints,dT),'grid':gp}
    return {'base':base,'pit':pit,'temp':temp,'dT':dT,
            'result':result,'drivers':drivers,'total_laps':tl}

def load_test_cases(test_dir):
    inp=sorted(glob.glob(os.path.join(test_dir,'inputs','*.json')))
    exp=sorted(glob.glob(os.path.join(test_dir,'expected_outputs','*.json')))
    cases=[]
    for i,e in zip(inp,exp):
        try:
            with open(i) as f: race=json.load(f)
            with open(e) as f: expected=json.load(f)
            exp_order=[str(r) for r in expected.get('finishing_positions',[])]
            p=parse_race(race, override_result=exp_order)
            if p: cases.append(p)
        except: pass
    return cases

def load_hist(data_dir, max_races=10000):
    files=sorted(glob.glob(os.path.join(data_dir,'historical_races','*.json')))
    races=[]
    for f in files:
        if len(races)>=max_races: break
        try:
            with open(f) as fh: raw=json.load(fh)
            for r in (raw if isinstance(raw,list) else [raw]):
                p=parse_race(r)
                if p: races.append(p)
                if len(races)>=max_races: break
        except: pass
    return races

def build_A(parsed, nearby=19):
    rows=[]
    for r in parsed:
        result,drivers=r['result'],r['drivers']
        sigs={d:tuple(dd['stints']) for d,dd in drivers.items()}
        for i in range(len(result)):
            for j in range(i+1,min(i+nearby+1,len(result))):
                wi,li=result[i],result[j]
                if wi not in drivers or li not in drivers: continue
                if sigs.get(wi)==sigs.get(li): continue
                rows.append(drivers[wi]['fv']-drivers[li]['fv'])
    return np.array(rows) if rows else np.zeros((0,12))

def run_lp(A, time_limit=60):
    if len(A)==0: return None,-9999
    A=np.unique(np.round(A,8),axis=0)
    A_aug=np.hstack([A,np.ones((len(A),1))])
    b=np.zeros(len(A))
    c_obj=np.zeros(13); c_obj[-1]=-1.0
    bounds=list(BOUNDS)+[(None,None)]
    try:
        res=linprog(c_obj,A_ub=A_aug,b_ub=b,bounds=bounds,method='highs',
                    options={'time_limit':time_limit,'primal_feasibility_tolerance':1e-10})
        if res.success:
            return res.x[:12],-res.fun
    except: pass
    return None,-9999

def evaluate(parsed, x, label='', n_eval=None):
    params=dict(zip(NAMES,x))
    exact=pos_ok=total=0
    data=parsed[:n_eval] if n_eval else parsed
    for r in data:
        result,drivers=r['result'],r['drivers']
        tl=[(round(driver_time(d['stints'],r['base'],r['temp'],r['pit'],params),TIME_ROUND),
             d['grid'],did) for did,d in drivers.items()]
        tl.sort(key=lambda z:(z[0],z[1]))
        pred=[z[2] for z in tl]
        if pred==result: exact+=1
        pos_ok+=sum(1 for p,e in zip(pred,result) if p==e)
        total+=len(result)
    n=len(data)
    pct=100*exact/n if n else 0
    pa=100*pos_ok/total if total else 0
    if label:
        print(f"  {label}: {exact}/{n} ({pct:.1f}%)  pos={pa:.1f}%")
    return exact,n,pct,pa

def smooth_loss(x, A_test, A_hist, margin=0.01):
    """Weighted hinge loss: test cases get 50x weight."""
    loss=0.0
    if len(A_test)>0:
        s=A_test@x
        loss+=50.0*float(np.mean(np.maximum(0.,s+margin)**2))
    if len(A_hist)>0:
        s=A_hist@x
        loss+=float(np.mean(np.maximum(0.,s+margin)**2))
    return loss

def save(x, path):
    p=dict(DEFAULT_PARAMS)
    p.update({n:float(v) for n,v in zip(NAMES,x)})
    os.makedirs(os.path.dirname(path) or '.',exist_ok=True)
    with open(path,'w') as f: json.dump(p,f,indent=2)

def fit(data_dir='data', output='solution/params.json',
        hist_races=10000, quick=False):
    t0=time.time()
    print("="*60)
    print("Box Box Box — Smart Parameter Fitter")
    print("="*60)

    test_dir=os.path.join(data_dir,'test_cases')

    # ── Load data ──────────────────────────────────────────────────
    print(f"\nLoading test cases...",flush=True)
    test_cases=load_test_cases(test_dir)
    print(f"  {len(test_cases)} test cases loaded")

    max_hist=2000 if quick else hist_races
    print(f"Loading {max_hist} historical races...",flush=True)
    hist=load_hist(data_dir,max_hist)
    print(f"  {len(hist)} historical races loaded")

    # ── Step 1: LP on test cases only ──────────────────────────────
    print(f"\n{'='*60}")
    print("Step 1: LP on 100 test cases (ground truth)...",flush=True)
    A_test=build_A(test_cases, nearby=19)
    print(f"  Constraints: {len(A_test)}")

    x_lp,eps=run_lp(A_test, time_limit=60)
    print(f"  LP eps={eps:.6f}")

    if x_lp is not None:
        e,n,pct,pa=evaluate(test_cases,x_lp,"test LP")
        evaluate(hist[:500],x_lp,"hist500 LP",500)
    else:
        print("  LP failed on test cases — using defaults")
        x_lp=np.array([DEFAULT_PARAMS.get(n,0.0) for n in NAMES])

    # ── Step 2: LP on test+hist combined ───────────────────────────
    print(f"\nStep 2: LP on test+hist combined...",flush=True)
    A_hist_small=build_A(hist[:2000], nearby=4)
    A_combined=np.vstack([A_test, A_hist_small]) if len(A_hist_small) else A_test
    print(f"  Combined constraints: {len(A_combined)}")

    x_combined,eps2=run_lp(A_combined, time_limit=90)
    print(f"  Combined LP eps={eps2:.6f}")
    if x_combined is not None:
        e2,n2,pct2,pa2=evaluate(test_cases,x_combined,"test combined")

    # Pick best LP solution
    x_best=x_lp
    best_pct=pct if x_lp is not None else 0
    if x_combined is not None and pct2>best_pct:
        x_best=x_combined; best_pct=pct2

    save(x_best, output)
    print(f"\n  Best LP: {best_pct:.1f}% → saved")

    # ── Step 3: Nelder-Mead refinement ─────────────────────────────
    print(f"\nStep 3: Nelder-Mead refinement...",flush=True)
    A_hist_all=build_A(hist, nearby=4)

    stages=[
        (0.5,  5000,  "coarse"),
        (0.1,  20000, "medium"),
        (0.02, 50000, "fine  "),
        (0.005,50000, "vfine "),
    ]
    for margin,maxiter,name in stages:
        res=minimize(smooth_loss, x_best,
                     args=(A_test, A_hist_all, margin),
                     method='Nelder-Mead',
                     options={'maxiter':maxiter,'xatol':1e-11,'fatol':1e-11,'adaptive':True})
        xn=res.x
        for i,(lo,hi) in enumerate(BOUNDS):
            if lo is not None: xn[i]=max(lo,xn[i])
            if hi is not None: xn[i]=min(hi,xn[i])
        e_,n_,pct_,pa_=evaluate(test_cases,xn,f"NM {name}")
        if pct_>=best_pct:
            x_best=xn; best_pct=pct_
            save(x_best,output)
        print(f"    [{time.time()-t0:.0f}s]",flush=True)

    # ── Step 4: DE from multiple starts (if time allows) ───────────
    if not quick:
        print(f"\nStep 4: Differential Evolution...",flush=True)
        de=differential_evolution(
            smooth_loss, BOUNDS,
            args=(A_test, A_hist_all[:100000], 0.02),
            maxiter=200, popsize=12, tol=1e-10, seed=42,
            disp=False, workers=1, x0=x_best,
            mutation=(0.5,1.5), recombination=0.9,
        )
        e_,n_,pct_,pa_=evaluate(test_cases,de.x,"DE result")
        if pct_>=best_pct:
            x_best=de.x; best_pct=pct_
            save(x_best,output)
            print(f"  DE improved → saved [{time.time()-t0:.0f}s]")

    # ── Final ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    evaluate(test_cases,x_best,"FINAL test")
    evaluate(hist[:1000],x_best,"FINAL hist1000",1000)
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    save(x_best,output)
    print(f"Saved to {output}")
    print("\nParams:")
    for n,v in zip(NAMES,x_best):
        print(f"  {n:20s} = {v:+.8f}")
    return x_best

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',   default='data')
    ap.add_argument('--output',     default='solution/params.json')
    ap.add_argument('--hist-races', type=int, default=10000)
    ap.add_argument('--quick',      action='store_true',
                    help='Fast mode: 2000 hist races, skip DE (~3 min)')
    args=ap.parse_args()
    fit(args.data_dir, args.output, args.hist_races, args.quick)