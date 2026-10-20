"""Independent dense-energy and finite-difference checks for fixed pair weights."""
import importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build/test-results/pair-weights'
OUT.mkdir(parents=True, exist_ok=True)
spec=importlib.util.spec_from_file_location('sparse_pair_test',ROOT/'arm_clearance/solver_v2_contacts.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
pair_contacts=module.pair_contacts
rng=np.random.default_rng(56);n=3;p=7;a=3
points=rng.normal(0,.025,(n,2,p,3))
axes=rng.uniform(.009,.027,(n,2,p,3));radii=axes.max(-1)
covariance=np.eye(3)*axes[...,None,:]**2
mutual_weights=rng.uniform(.3,7.,(n,2,p));own_weights=rng.uniform(.2,9.,(n,2,p))
mask=rng.random((2,a,p-a))>.1; margin=.001

def energy(pts,cov):
    total=0.
    for sa,sb,ia,ib,weight,allowed in ((0,1,np.arange(p),np.arange(p),mutual_weights,None),
        (0,0,np.arange(a),np.arange(a,p),own_weights,mask[0]),
        (1,1,np.arange(a),np.arange(a,p),own_weights,mask[1])):
        diff=pts[:,sa,ia][:,:,None]-pts[:,sb,ib][:,None]
        length=np.linalg.norm(diff,axis=-1);direction=diff/np.maximum(length[...,None],1e-12)
        ca=np.einsum('npij,npqj->npqi',cov[:,sa,ia],direction)
        cb=np.einsum('nqij,npqj->npqi',cov[:,sb,ib],direction)
        ra=np.sqrt(np.maximum(np.sum(ca*direction,-1),1e-14))
        rb=np.sqrt(np.maximum(np.sum(cb*direction,-1),1e-14))
        depth=np.maximum(ra+rb+margin-length,0.)
        if allowed is not None:depth*=allowed[None]
        weights=np.maximum(weight[:,sa,ia][:,:,None],weight[:,sb,ib][:,None])
        total+=np.sum(depth*depth*weights)
    return total

observed=pair_contacts(points,covariance,radii,margin,mask,a,
    mutual_weights=mutual_weights,own_weights=own_weights)
assert abs(observed[0]-energy(points,covariance))<1e-14
epsilon=1e-7;force_errors=[];torque_errors=[]
for fi,side,point,axis in ((0,0,0,0),(1,0,5,1),(2,1,1,2),(1,1,6,0)):
    plus=points.copy();minus=points.copy()
    plus[fi,side,point,axis]+=epsilon;minus[fi,side,point,axis]-=epsilon
    numeric=(energy(plus,covariance)-energy(minus,covariance))/(2*epsilon)
    force_errors.append(abs(numeric-observed[1][fi,side,point,axis]))
    skew=np.zeros((3,3));k=(axis+1)%3;l=(axis+2)%3;skew[k,l]=-1.;skew[l,k]=1.
    rotation_plus=np.eye(3)+np.sin(epsilon)*skew+(1-np.cos(epsilon))*(skew@skew)
    rotation_minus=rotation_plus.T
    plus=covariance.copy();minus=covariance.copy();original=covariance[fi,side,point]
    plus[fi,side,point]=rotation_plus@original@rotation_plus.T
    minus[fi,side,point]=rotation_minus@original@rotation_minus.T
    numeric=(energy(points,plus)-energy(points,minus))/(2*epsilon)
    torque_errors.append(abs(numeric-observed[2][fi,side,point,axis]))
assert max(force_errors)<1e-8,force_errors
assert max(torque_errors)<1e-8,torque_errors
report={'dense_energy_absolute_error':abs(observed[0]-energy(points,covariance)),
    'force_finite_difference_errors':force_errors,'torque_finite_difference_errors':torque_errors,
    'fixed_weight_policy':'max endpoint weight','passed':True}
(OUT/'pair-weights.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(report)
