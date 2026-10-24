import numpy as np, sys
sys.path.insert(0,'.')  # run from the repo root; assignment_data.npz from the course zip must be in the cwd
from formats import quantize, blocks, floor_log2
Z=np.load('assignment_data.npz')
def nsr_true(W,name): Wq,_=quantize(W,name,policy='ocp'); return ((W-Wq)**2).sum()/(W**2).sum()
def rho(W,block=32):
    """block-peak statistic: sum over blocks of (#nonzeros * 4^floor(log2 max)) / total energy"""
    num=0.0
    for sl in blocks(W.shape[1],block):
        B=np.abs(W[:,sl]); m=B.max(1); ok=m>0
        F=floor_log2(np.where(ok,m,1.0)); nnz=(B>0).sum(1)
        num+=(nnz*4.0**F*ok).sum()
    return num/(W**2).sum()
def pred_int(W): return rho(W)*4.0**-4/12            # MXINT6: step = 2^(F-4), uniform noise step^2/12 on every nonzero
def pred_fp(m): return 4.0**-m/28                    # FP: step ~ 2^-m * binade, uniform density inside a binade
print("dataset  rho   | NSR true (INT E2M3 E3M2 E4M1) | closed-form (INT E2M3 E3M2 E4M1) | winner true / predicted")
sets={k:Z['W_'+k].astype(np.float64) for k in 'ABCDE'}
for k,W in sets.items():
    t={n:nsr_true(W,n) for n in ('MXINT6','MXFP6_E2M3','MXFP6_E3M2','MXFP6_E4M1')}
    p={'MXINT6':pred_int(W),'MXFP6_E2M3':pred_fp(3),'MXFP6_E3M2':pred_fp(2),'MXFP6_E4M1':pred_fp(1)}
    wt=min(t,key=t.get); wp=min(p,key=p.get)
    print(f"{k}  {rho(W):6.2f} | "+" ".join(f"{v:.2e}" for v in t.values())+" | "+" ".join(f"{v:.2e}" for v in p.values())+f" | {wt[-5:]} / {wp[-5:]}")
# C with sparsity sweep: does rho track the crossover?
rng=np.random.default_rng(0); W0=sets['C']
print("\nW_C with random zeros: rho, true NSR INT vs E2M3, closed-form INT vs E2M3")
for p in (0,25,50,60,70,75,80,90):
    W=W0.copy(); W.flat[rng.choice(W.size,int(W.size*p/100),replace=False)]=0
    ti,tf=nsr_true(W,'MXINT6'),nsr_true(W,'MXFP6_E2M3')
    print(f" {p:3d}%  rho {rho(W):5.2f}  true INT {ti:.2e} E2M3 {tf:.2e} (INT wins: {ti<tf})   formula INT {pred_int(W):.2e} E2M3 {pred_fp(3):.2e} (INT wins: {pred_int(W)<pred_fp(3)})")
