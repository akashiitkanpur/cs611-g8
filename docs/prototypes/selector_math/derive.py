import numpy as np, sys
sys.path.insert(0,'.')
from formats import quantize, blocks, floor_log2, default_exp, spec
Z=np.load(sys.argv[1]); W=Z['W_C'].astype(np.float64)
# ---- worked example: ONE block of 8 numbers, MXINT6 ----
x=np.array([0.9,-0.4,0.15,0.0,2.3,-1.1,0.05,0.7])
m=np.abs(x).max(); F=np.floor(np.log2(m)); S=2.0**F; step=S*2**-4
q=np.clip(np.round(x/step),-31,31)*step
print("one block:",x)
print(f" max={m}, F=floor(log2 max)={F:.0f}, scale S=2^F={S}, MXINT6 step = S*2^-4 = {step}")
print(" quantized :",q); print(" error     :",np.round(q-x,5))
print(f" actual sum err^2 = {((q-x)**2).sum():.6f}; model: nonzeros({(x!=0).sum()}) * step^2/12 = {(x!=0).sum()*step**2/12:.6f}")
print(f" actual NSR={((q-x)**2).sum()/(x**2).sum():.3e}; model NSR = rho_block*4^-4/12 with rho_block = n_nz*4^F/sum x^2 = {(x!=0).sum()*4**F/(x**2).sum():.3f} -> {(x!=0).sum()*4**F/(x**2).sum()*4**-4/12:.3e}")
# ---- full W_C, step by step ----
num=0.0; nb=0; nnz_tot=0
for sl in blocks(100,32):
    B=np.abs(W[:,sl]); mx=B.max(1); Fb=floor_log2(np.where(mx>0,mx,1.0)); nz=(B>0).sum(1)
    num+=(nz*4.0**Fb).sum(); nb+=len(mx); nnz_tot+=nz.sum()
den=(W**2).sum()
print(f"\nW_C: blocks={nb}, nonzeros={nnz_tot}, sum(n_nz*4^F)={num:.1f}, sum w^2={den:.1f}")
rho=num/den; print(f" rho = {rho:.4f}")
nsr=rho*4.0**-4/12; print(f" NSR_INT6 = rho*4^-4/12 = {nsr:.4e} -> relative error sqrt = {100*np.sqrt(nsr):.3f}%")
Wq,_=quantize(W,'MXINT6',policy='ocp'); print(f" actual MXINT6 weight rel err = {100*np.linalg.norm(W-Wq)/np.linalg.norm(W):.3f}%")
# floating point: err^2 = (2^e * 2^-m)^2/12 per element, w = q*2^e, q in [1,2)
phi=(4.0**floor_log2(np.abs(W[W!=0]))).sum()/den
print(f"\n phi = sum 4^floor(log2|w|) / sum w^2 = {phi:.4f}  (= 1/<q^2>, energy-weighted mean of q^2 = {1/phi:.3f})")
for name,mb in (('MXFP6_E2M3',3),('MXFP6_E3M2',2),('MXFP6_E4M1',1)):
    pred=4.0**-mb/12*phi; Wq,_=quantize(W,name,policy='ocp'); true=((W-Wq)**2).sum()/den
    print(f" {name}: phi-formula 4^-m/12*phi = {pred:.3e}   true = {true:.3e}   true/formula = {true/pred:.2f}")
