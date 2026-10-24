import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import *
Z=np.load(sys.argv[1]); rng=np.random.default_rng(1)
sh=(128,128)
def gauss(): return rng.standard_normal(sh)
def lap(): return rng.laplace(size=sh)
def t3(): return rng.standard_t(3,sh)
def uni(): return rng.uniform(-1,1,sh)
def sparse(p): return lambda: rng.standard_normal(sh)*(rng.random(sh)>p)
def outl(n,k): 
    def f():
        W=rng.standard_normal(sh); W.flat[rng.choice(W.size,n,replace=False)]=rng.choice([-1,1],n)*k; return W
    return f
def loguni(b): return lambda: rng.choice([-1,1],sh)*2.0**rng.uniform(-b,0,sh)          # magnitudes spread over b binades
def pareto(a): return lambda: rng.choice([-1,1],sh)*(rng.pareto(a,sh)+1)
def lognorm(s): return lambda: rng.choice([-1,1],sh)*np.exp(s*rng.standard_normal(sh))
def ternary(): return rng.choice([-0.5,0,0.5],sh)
def binary(): return rng.choice([-1.0,1.0],sh)
def alpha(): return rng.choice([-3,-2,-1,1,2,3],sh)*0.25
def bimodal(): return rng.choice([-1,1],sh)*(1+0.1*rng.standard_normal(sh))
def smooth(): 
    t=np.arange(sh[1]); return np.sin(2*np.pi*t[None,:]/50+rng.uniform(0,6,(sh[0],1)))+0.05*rng.standard_normal(sh)
def onepb(): 
    W=rng.standard_normal(sh)*0.3; W[:,::32]=8.0*rng.choice([-1,1],(sh[0],1)); return W
def colspread(): return rng.standard_normal(sh)*np.repeat(10**rng.uniform(-2,2,16),8)[None,:]
def zeroblocks(): 
    W=rng.standard_normal(sh); W[rng.random(sh[0])<0.5,:]=0; return W
def pre8(): 
    W=rng.standard_normal(sh); return np.round(W/ (np.abs(W).max()/127))*(np.abs(W).max()/127)       # already on a 8-bit INT grid
fam={'gauss':gauss,'laplace':lap,'student-t3':t3,'uniform':uni,'sparse50':sparse(.5),'sparse80':sparse(.8),'sparse95':sparse(.95),
 'outl 20x20':outl(20,20),'outl 20x50':outl(20,50),'outl 200x20':outl(200,20),'outl 1/block x8':onepb,'loguni 4 binades':loguni(4),'loguni 8':loguni(8),'loguni 12':loguni(12),
 'pareto 1.5':pareto(1.5),'pareto 3':pareto(3),'lognormal s=1':lognorm(1),'lognormal s=2':lognorm(2),'ternary':ternary,'binary +-1':binary,'6-level alphabet':alpha,
 'bimodal +-1':bimodal,'smooth sinusoid':smooth,'col spread 1e4':colspread,'half rows zero':zeroblocks,'pre-quantized int8':pre8}
for k in 'ABCDE': fam['dataset '+k]=(lambda k=k:Z['W_'+k].astype(np.float64))
print("NSR true | ratio M1(per-element model)/true | ratio simple(rho,phi)/true   -- order: INT6, E2M3, E3M2, E4M1;  winners true / M1 / simple")
bad=[]
for name,g in fam.items():
    W=g(); t={n:true_nsr(W,n) for n in NAMES}; a={n:m1(W,n) for n in NAMES}; s=simple(W)
    w=lambda d:min(d,key=d.get)[-5:]
    r1=" ".join(f"{a[n]/t[n]:5.2f}" if t[n]>1e-12 else "  n/a" for n in NAMES); r2=" ".join(f"{s[n]/t[n]:5.2f}" if t[n]>1e-12 else "  n/a" for n in NAMES)
    tr=" ".join(f"{t[n]:.1e}" for n in NAMES[:3])
    flag='' if (w(t)==w(a)==w(s)) else '  <-- disagreement'
    print(f"{name:18s} {tr} | {r1} | {r2} | {w(t)}/{w(a)}/{w(s)}{flag}")
