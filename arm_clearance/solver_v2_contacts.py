"""Exact sparse evaluation of mutually intersecting limb ellipsoids.

Every pair is screened with a conservative enclosing sphere.  Omitted pairs
have exactly zero penetration, energy and derivative in the existing solver.
The remaining ellipsoid support, forces and torques use the same formulae.
"""
import numpy as np


def pair_contacts(points, covariance, radii, margin, own_mask, arm_count,
                  need_gradient=True, chunk_size=32,
                  mutual_weights=None,own_weights=None):
    n,_,p,_=points.shape
    # Model radii are the largest local ellipsoid axes.  Exact rotations keep
    # those bounds, but imported pose/basis matrices can contain small shear.
    # A covariance Gershgorin bound also encloses that case without assuming
    # orthogonality or spending an eigendecomposition on every sample.
    covariance_bound=np.sqrt(np.max(np.sum(np.abs(covariance),axis=-1),axis=-1))
    radii=np.maximum(np.maximum(radii,covariance_bound),1e-7)
    force=np.zeros_like(points); torque=np.zeros_like(points)
    mutual_frames=np.zeros(n); own_frames=np.zeros(n);energy=0.
    candidates=0;positive=0

    def evaluate(start,end,side_a,side_b,ids_a,ids_b,mask,frames,weights):
        nonlocal energy,candidates,positive
        pa=points[start:end,side_a,ids_a];pb=points[start:end,side_b,ids_b]
        diff=pa[:,:,None]-pb[:,None]
        bound=(radii[start:end,side_a][:,ids_a,None]
               +radii[start:end,side_b][:,None,ids_b]+margin+1e-12)
        squared=np.sum(diff*diff,axis=-1)
        possible=squared<bound*bound
        if mask is not None:possible &= mask[None]
        fi,ai,bi=np.nonzero(possible)
        if not len(fi):return
        candidates+=len(fi)
        local_diff=diff[fi,ai,bi]
        length=np.sqrt(squared[fi,ai,bi]);inverse=1/np.maximum(length,1e-12)
        direction=local_diff*inverse[:,None]
        ca=np.einsum('nij,nj->ni',covariance[start+fi,side_a,ids_a[ai]],direction)
        cb=np.einsum('nij,nj->ni',covariance[start+fi,side_b,ids_b[bi]],direction)
        ra=np.sqrt(np.maximum(np.sum(ca*direction,-1),1e-14))
        rb=np.sqrt(np.maximum(np.sum(cb*direction,-1),1e-14))
        depth=np.maximum(ra+rb+margin-length,0.)
        pair_weights=(1. if weights is None else np.maximum(
            weights[start+fi,side_a,ids_a[ai]],weights[start+fi,side_b,ids_b[bi]]))
        energy+=float(np.sum(depth*depth*pair_weights));positive+=np.count_nonzero(depth)
        np.maximum.at(frames,start+fi,depth)
        if need_gradient:
            active=depth>0.
            weighted_depth=(depth*pair_weights)[active]
            fi,ai,bi,depth,direction,ca,cb,ra,rb,inverse=(
                value[active] for value in (fi,ai,bi,depth,direction,ca,cb,ra,rb,inverse))
            derivative=ca/ra[:,None]+cb/rb[:,None]
            derivative-=direction*np.sum(derivative*direction,axis=-1,keepdims=True)
            pair=2*weighted_depth[:,None]*(derivative*inverse[:,None]-direction)
            np.add.at(force,(start+fi,side_a,ids_a[ai]),pair)
            np.add.at(force,(start+fi,side_b,ids_b[bi]),-pair)
            np.add.at(torque,(start+fi,side_a,ids_a[ai]),2*weighted_depth[:,None]*np.cross(ca,direction)/ra[:,None])
            np.add.at(torque,(start+fi,side_b,ids_b[bi]),2*weighted_depth[:,None]*np.cross(cb,direction)/rb[:,None])

    all_ids=np.arange(p);arm_ids=np.arange(arm_count);hand_ids=np.arange(arm_count,p)
    for start in range(0,n,chunk_size):
        end=min(start+chunk_size,n)
        evaluate(start,end,0,1,all_ids,all_ids,None,mutual_frames,mutual_weights)
        for side in range(2):
            evaluate(start,end,side,side,arm_ids,hand_ids,own_mask[side],own_frames,own_weights)
    return energy,force,torque,{'mutual_max':float(mutual_frames.max()),
        'own_max':float(own_frames.max()),'mutual_frames':mutual_frames,
        'own_frames':own_frames,'pair_candidates':candidates,'positive_pairs':int(positive)}


def sparse_contacts(self,points,orientation,need_gradient=True):
    """Drop-in benchmark implementation; body terms are copied unchanged."""
    m=self.model;n=m.n;p=m.p
    body_margin=getattr(self,'body_margin',0.)
    body_weights=getattr(self,'body_weights',1.)
    if need_gradient:
        mask=np.ptp(m.axes,axis=-1)>1e-8
        distance,g,hessian=self.body.distances(points,hessian=True,hessian_mask=mask,
            hessian_distance_limit=m.radii+m.margin+body_margin+.001)
    else:
        distance,g=self.body.distances(points);hessian=None
    unit=g/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-12)
    covariance=(orientation*(m.axes[...,None,:]**2))@orientation.swapaxes(-1,-2)
    cn=np.einsum('nspij,nspj->nspi',covariance,unit)
    radius=np.sqrt(np.maximum(np.sum(cn*unit,axis=-1),1e-14))
    body=np.maximum(radius+m.margin+body_margin-distance,0.)
    weighted_body=body*body_weights
    force=-2*weighted_body[...,None]*g
    torque=np.zeros((n,2,p,3))
    if need_gradient:
        quotient=cn/radius[...,None]
        dg=(quotient-unit*radius[...,None])/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-12)
        force+=2*weighted_body[...,None]*np.einsum('nspij,nspj->nspi',hessian,dg)
        torque+=2*weighted_body[...,None]*np.cross(cn,unit)/radius[...,None]
    energy,other_force,other_torque,detail=pair_contacts(points,covariance,m.radii,
        m.margin,m.own_mask,m.arm_count,need_gradient,
        mutual_weights=getattr(self,'mutual_weights',None),
        own_weights=getattr(self,'own_weights',None))
    force+=other_force;torque+=other_torque
    detail['body']=body
    return float(np.sum(body*weighted_body))+energy,force,torque,detail
