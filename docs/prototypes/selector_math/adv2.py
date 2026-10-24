import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import *
Z=np.load(sys.argv[1]); rng=np.random.default_rng(2)
sh=(128,128)
print("== CONTINUOUS outliers (magnitude k*U(0.5,1.5), random sign) on N(0,1); NSR true | M1/true | winners true/M1")
print("   n_out  k   INT6    E2M3    E3M2    | INT  E2M3 E3M2 E4M1 | true/M1")
agree=tot=0
for n in (5,20,100,400,1500):
    for k in (10,20,50,100):
        res=[]
        for rep in range(3):
            W=rng.standard_normal(sh); W.flat[rng.choice(W.size,n,replace=False)]=rng.choice([-1,1],n)*k*rng.uniform(.5,1.5,n)
            t={x:true_nsr(W,x) for x in NAMES}; a={x:m1(W,x) for x in NAMES}; res.append((t,a))
        t={x:np.mean([r[0][x] for r in res]) for x in NAMES}; a={x:np.mean([r[1][x] for r in res]) for x in NAMES}
        wt,wa=min(t,key=t.get),min(a,key=a.get); tot+=1; agree+=wt==wa
        print(f"  {n:5d} {k:4d} {t['MXINT6']:.1e} {t['MXFP6_E2M3']:.1e} {t['MXFP6_E3M2']:.1e} | "+" ".join(f"{a[x]/t[x]:4.2f}" for x in NAMES)+f" | {wt[-5:]}/{wa[-5:]}"+('' if wt==wa else '  <--'))
print(f"decision agreement M1 vs exact: {agree}/{tot}")
# ---- output-aware (X) weighting, real datasets: predicted Y-NSR = sum err^2 H_jj / sum w^2 H_jj
X=Z['X_cal'].astype(np.float64); H=np.diag(X@X.T)
def m1w(W,name,block=32):
    kind,p,emax=spec(name); tot=0.0
    for sl in blocks(W.shape[1],block):
        B=np.abs(W[:,sl]); e=default_exp(kind,p,emax,B.max(1),'ocp'); s=(2.0**e)[:,None]; z=B/s
        if kind=='fp': step=2.0**(np.maximum(floor_log2(np.where(z>0,z,1.0)),1-p['bias'])-p['mbits']); vm=p['maxval']
        else: step=2.0**-p['frac_bits']; vm=p['kmax']*step
        err=np.where(z>vm,(z-vm)**2,np.where(z<step/2,z**2,step**2/12)); tot+=(np.where(B>0,err,0)*s**2*H[sl][None,:]).sum()
    return tot/((W**2)*H[None,:]).sum()
print("\n== Y-error (with X_cal, H=diag(XX^T) weights) real datasets: predicted/true NSR of Y")
for k in 'ABCDE':
    W=Z['W_'+k].astype(np.float64); Y=W@X; r=[]
    for x in NAMES:
        Wq,_=quantize(W,x,policy='ocp'); true=((Wq@X-Y)**2).sum()/(Y**2).sum(); r.append(m1w(W,x)/true)
    print(f" {k}: "+" ".join(f"{v:4.2f}" for v in r))
# ---- other bit widths: INT closed form rho*4^-(b-2)/12 vs true generic MXINT-b
def mxint(W,b,block=32):
    out=np.empty_like(W); kmax=2**(b-1)-1
    for sl in blocks(W.shape[1],block):
        B=W[:,sl]; mx=np.abs(B).max(1); F=floor_log2(np.where(mx>0,mx,1.0)); st=(2.0**(F-(b-2)))[:,None]
        out[:,sl]=np.clip(np.round(B/st),-kmax,kmax)*st
    return out
print("\n== other bit widths, MXINT-b on W_C and dataset B: rho formula / true")
for k in 'CB':
    W=Z['W_'+k].astype(np.float64)
    print(f" {k}: "+"  ".join(f"b={b}: {rho(W)*4.0**-(b-2)/12/(((W-mxint(W,b))**2).sum()/(W**2).sum()):.2f}" for b in (4,6,8)))
