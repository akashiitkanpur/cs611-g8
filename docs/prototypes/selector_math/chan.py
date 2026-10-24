import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import *
Z=np.load(sys.argv[1])
W0=Z['W_C'].astype(np.float64); r=np.random.default_rng(5)
print("W_C with outlier CHANNELS: c random columns multiplied by factor f (continuous values). NSR(W) true, M1/true, winners")
print("  c   f  |  INT6    E2M3    E3M2    E4M1 | M1/true INT E2M3 E3M2 | true/M1 | outlier-energy share")
for c in (1,2,4,8):
    for f in (10,30,100,300):
        t={x:0 for x in NAMES}; a={x:0 for x in NAMES}; sh_=0
        for rep in range(4):
            W=W0.copy(); cols=r.choice(100,c,replace=False); W[:,cols]*=f
            for x in NAMES: t[x]+=true_nsr(W,x)/4; a[x]+=m1(W,x)/4
            sh_+=(W[:,cols]**2).sum()/(W**2).sum()/4
        wt,wa=min(t,key=t.get),min(a,key=a.get)
        print(f" {c:2d} {f:4d} | "+" ".join(f"{t[x]:.1e}" for x in NAMES)+" | "+" ".join(f"{a[x]/t[x]:4.2f}" for x in NAMES[:3])+f" | {wt[-5:]}/{wa[-5:]}"+("" if wt==wa else " <--")+f" | {sh_:.2f}")
