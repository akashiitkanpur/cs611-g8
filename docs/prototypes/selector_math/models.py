"""Shared error models for the selector prototypes (v1 formats.py names; no data is loaded here).
true_nsr: exhaustive weight error-to-signal; m1: per-element expected-error model; rho / phi: block-peak and
element-binade statistics; simple: the two-statistic closed form built from rho and phi."""
import numpy as np, os, sys
sys.path.insert(0, '.')
from formats import quantize, blocks, floor_log2, default_exp, spec
NAMES=['MXINT6','MXFP6_E2M3','MXFP6_E3M2','MXFP6_E4M1']
def true_nsr(W,n): Wq,_=quantize(W,n,policy='ocp'); return ((W-Wq)**2).sum()/(W**2).sum()
def m1(W,name,block=32):
    """per-element model: error^2 = step^2/12 (uniform), flush z^2 if z<step/2, clip (z-vmax)^2; scale = class ocp rule"""
    kind,p,emax=spec(name); tot=0.0
    for sl in blocks(W.shape[1],block):
        B=np.abs(W[:,sl]); e=default_exp(kind,p,emax,B.max(1),'ocp'); s=(2.0**e)[:,None]; z=B/s
        if kind=='fp': step=2.0**(np.maximum(floor_log2(np.where(z>0,z,1.0)),1-p['bias'])-p['mbits']); vm=p['maxval']
        else: step=2.0**-p['frac_bits']; vm=p['kmax']*step
        err=np.where(z>vm,(z-vm)**2,np.where(z<step/2,z**2,step**2/12)); tot+=(np.where(B>0,err,0)*s**2).sum()
    return tot/(W**2).sum()
def rho(W):
    num=0.0
    for sl in blocks(W.shape[1],32):
        B=np.abs(W[:,sl]); mx=B.max(1); num+=((B>0).sum(1)*4.0**floor_log2(np.where(mx>0,mx,1.0))*(mx>0)).sum()
    return num/(W**2).sum()
def phi(W): a=np.abs(W[W!=0]); return (4.0**floor_log2(a)).sum()/(W**2).sum()
def simple(W):   # few-statistics version: INT via rho, FP via phi (no flush/clip terms)
    return {'MXINT6':rho(W)*4.0**-4/12,'MXFP6_E2M3':4.0**-3/12*phi(W),'MXFP6_E3M2':4.0**-2/12*phi(W),'MXFP6_E4M1':4.0**-1/12*phi(W)}
