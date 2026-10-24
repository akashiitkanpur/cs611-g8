import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import *
Z=np.load(sys.argv[1])

from formats import elem_round
def m1h(W,name,block=32,top=1):
    """hybrid: elements within the top 'top' binades of their block are rounded EXACTLY; all others use the expected-value (step^2/12) model"""
    kind,p,emax=spec(name); tot=0.0; nex=0
    for sl in blocks(W.shape[1],block):
        B=np.abs(W[:,sl]); mx=B.max(1); e=default_exp(kind,p,emax,mx,'ocp'); s=(2.0**e)[:,None]; z=B/s
        if kind=='fp': step=2.0**(np.maximum(floor_log2(np.where(z>0,z,1.0)),1-p['bias'])-p['mbits']); vm=p['maxval']
        else: step=2.0**-p['frac_bits']; vm=p['kmax']*step
        exp_=np.where(z>vm,(z-vm)**2,np.where(z<step/2,z**2,step**2/12))
        F=floor_log2(np.where(mx>0,mx,1.0))[:,None]; topmask=(B>0)&(B>=2.0**(F-(top-1)))
        ex=(z-np.abs(elem_round(kind,p,z)))**2
        err=np.where(topmask,ex,exp_); tot+=(np.where(B>0,err,0)*s**2).sum(); nex+=topmask.sum()
    return tot/(W**2).sum(), nex/W.size

r=np.random.default_rng(21); sh=(128,128); N=1000; recs=[]
for it in range(N):
    base=r.choice(['gauss','laplace','t3','loguni'])
    if base=='gauss': W=r.standard_normal(sh)
    elif base=='laplace': W=r.laplace(size=sh)
    elif base=='t3': W=r.standard_t(3,sh)
    else: W=r.choice([-1,1],sh)*2.0**r.uniform(-r.uniform(2,14),0,sh)
    ps=r.choice([0,0,0.3,0.6,0.9]); W=W*(r.random(sh)>ps)
    po=10**r.uniform(-4,-1.3); k=10**r.uniform(0.5,3); mode=r.choice(['random','channel','block'])
    if mode=='random': m=r.random(sh)<po
    elif mode=='channel': m=np.zeros(sh,bool); m[:,r.choice(sh[1],max(1,int(po*sh[1]*3)),replace=False)]=True
    else: m=np.zeros(sh,bool); m[:,::32]=r.random((sh[0],4))<po*30
    W=np.where(m,W*k*r.uniform(.5,1.5,sh)+np.sign(W+1e-9)*k*0.5,W)
    W=W*10**r.uniform(-1,1,(sh[0],1))*(10**r.uniform(-1,1,(1,sh[1])) if r.random()<0.3 else 1)
    t={n:true_nsr(W,n) for n in NAMES}; a={n:m1(W,n) for n in NAMES}; s=simple(W); ah={top:{n:m1h(W,n,top=top)[0] for n in NAMES} for top in (1,2)}; frac=m1h(W,'MXINT6',top=1)[1]; recs.append((t,a,s,base,mode,ah,frac,ps,po,k,float(rho(W)),float(phi(W)),float(((np.abs(W)<np.abs(W).max(1,keepdims=True)/64)&(W!=0)).mean())))


def win(d): return min(d,key=d.get)
def summ(label,pick):
    ag=0;rg=[]
    for rec in recs:
        t=rec[0]; w=win(pick(rec)); ag+=w==win(t); rg.append(10*np.log10(t[w]/min(t.values())))
    rg=np.array(rg); print(f"{label:46s} top-1 {ag}/{len(recs)}  mean regret {rg.mean():.3f} dB  p95 {np.percentile(rg,95):.2f}  max {rg.max():.2f}  (>0.5 dB: {(rg>0.5).sum()})")
summ("expected-value model (M1)",lambda r:r[1])
summ("hybrid: exact top binade, expected rest",lambda r:r[5][1])
summ("hybrid: exact top 2 binades, expected rest",lambda r:r[5][2])
print("fraction of elements rounded exactly (top binade): mean %.3f"%np.mean([r[6] for r in recs]))
print("\n== CASCADE: accept the 2-statistic answer (rho,phi) if its best-vs-second-best margin >= tau dB, else escalate to hybrid")
print(" tau(dB) | decided by 2-stat | top-1 overall | mean regret | max regret | errors among 2-stat decisions")
for tau in (0,1,2,3,4,6,8,10):
    ag=0;rg=[];cov=0;err2=0
    for rec in recs:
        t=rec[0]; s=rec[2]; v=sorted(s.values()); marg=10*np.log10(v[1]/v[0])
        if marg>=tau: w=win(s); cov+=1; err2+= (w!=win(t))
        else: w=win(rec[5][1])
        ag+=w==win(t); rg.append(10*np.log10(t[w]/min(t.values())))
    rg=np.array(rg); print(f" {tau:6.0f}  | {100*cov/len(recs):6.1f}%          | {100*ag/len(recs):6.1f}%       | {rg.mean():.3f} dB   | {rg.max():.2f} dB    | {err2}/{cov}")
