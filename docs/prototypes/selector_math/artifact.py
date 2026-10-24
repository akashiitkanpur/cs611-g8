import numpy as np, sys
sys.path.insert(0,'.')
from formats import quantize
Z=np.load(sys.argv[1]); W0=Z['W_C'].astype(np.float64); sig=W0.std(); X=Z['X_cal'].astype(np.float64)
N=['MXINT6','MXFP6_E2M3','MXFP6_E3M2']
def err(W): return {n:100*np.linalg.norm((W-quantize(W,n)[0])@X)/np.linalg.norm(W@X) for n in N}
def run(n,k,jit,seeds=8):
    acc={x:0 for x in N}
    for s in range(seeds):
        r=np.random.default_rng(s); W=W0.copy(); idx=r.choice(W.size,n,replace=False)
        mag=k*sig*(r.uniform(1-jit,1+jit,n) if jit else 1.0); W.flat[idx]=r.choice([-1,1],n)*mag
        e=err(W)
        for x in N: acc[x]+=e[x]/seeds
    return acc
print("W_C, n outliers of magnitude k*sigma (Nikita's protocol: exact k*sigma) vs the same with +-30% jitter. Y err %, 8 seeds")
print(" n    k   |  exact: INT6  E2M3  E3M2 winner  |  jitter 30%: INT6  E2M3  E3M2 winner")
for n,k in ((20,20),(20,50),(100,10),(100,20),(100,50)):
    a=run(n,k,0.0); b=run(n,k,0.3)
    w=lambda d:min(d,key=d.get)[-5:]
    print(f"{n:4d} {k:3d} |  {a['MXINT6']:5.2f} {a['MXFP6_E2M3']:5.2f} {a['MXFP6_E3M2']:5.2f} {w(a):>6s}  |  {b['MXINT6']:5.2f} {b['MXFP6_E2M3']:5.2f} {b['MXFP6_E3M2']:5.2f} {w(b):>6s}")
