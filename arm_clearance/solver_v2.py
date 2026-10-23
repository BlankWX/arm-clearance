"""Six-bone spacetime correction with tight arm sections and articulated hands."""
import math
import numpy as np
from .solver import ClipSolver,rotation_vectors,corrected_quaternions
from .arm_geometry import fit_arm_ellipses
from .hand_geometry import build_hand_geometry
from .body_field import TightBodyField
from .solver_v2_derivatives import analytical_fk_gradient
from .solver_v2_contacts import pair_contacts


class FullLimbModel:
    parameter_count=8
    def __init__(self,matrices,quaternions,location_bases,geometry,meta,hand=None,margin=.001):
        self.matrices=np.asarray(matrices,np.float64);self.n=len(matrices)
        self.meta=meta;self.target_ids=list(meta['target_ids']);self.arm_ids=list(meta['arm_target_ids'])
        self.wrist_ids=list(meta['wrist_ids']);self.quaternions=quaternions
        self.rot=np.take(self.matrices,self.target_ids,axis=1)[:,:,:3,:3].copy()
        self.rot/=np.maximum(np.linalg.norm(self.rot,axis=-2,keepdims=True),1e-12)
        positions=np.take(self.matrices,self.target_ids,axis=1)[:,:,:3,3]
        self.shoulders=positions[:,(0,3)];self.elbow_orig=positions[:,(1,4)];self.wrist_orig=positions[:,(2,5)]
        self.upper=self.elbow_orig-self.shoulders;self.lower=self.wrist_orig-self.elbow_orig
        self.length=float(np.mean(np.linalg.norm(self.upper,axis=-1)))
        self.location_bases=np.asarray(location_bases,np.float64)
        arm_meta=dict(meta,target_ids=self.arm_ids)
        self.arm_geometry=fit_arm_ellipses(geometry,arm_meta,axial_fraction=.5)
        arm_axes=np.asarray(self.arm_geometry['radii_axes']);self.basis=self.arm_geometry['basis_local']
        self.tu=np.linspace(.34,1.,10);self.tf=np.linspace(.04,.99,13);self.arm_count=23
        self.hand=hand or build_hand_geometry(geometry,meta,matrices)
        self.hand_radii=self.hand.per_frame_radii
        if self.hand_radii is None:self.hand_radii=np.broadcast_to(self.hand.radii,(self.n,)+self.hand.radii.shape)
        self.p=self.arm_count+self.hand.original_points.shape[2]
        hand_axes=getattr(self.hand,'ellipsoid_axes',None)
        if hand_axes is None:hand_axes=np.repeat(self.hand_radii[...,None],3,axis=-1)
        self.hand_basis=getattr(self.hand,'ellipsoid_basis_local',None)
        if self.hand_basis is None:self.hand_basis=np.broadcast_to(np.eye(3),(2,self.p-23,3,3))
        self.axes=np.concatenate((np.broadcast_to(arm_axes,(self.n,)+arm_axes.shape),hand_axes),axis=2)
        self.radii=self.axes.max(-1)
        self.margin=margin
        self.original_points,self.original_orientation,_,_=self.state(np.zeros((self.n,8,3)))
        own=np.ones((2,23,self.p-23),bool)
        own[:,10:]=self.hand.own_forearm_pairs(self.tf).swapaxes(1,2)
        self.own_mask=own

    def state(self,x):
        local=rotation_vectors(x[:,:6]);world=self.rot@local@self.rot.swapaxes(-1,-2)
        u=world[:,(0,3)];f=world[:,(1,4)];h=world[:,(2,5)];upstream=u@f
        upper=np.einsum('nsij,nsj->nsi',u,self.upper)
        fore=np.einsum('nsij,nsj->nsi',upstream,self.lower)
        elbow=self.shoulders+upper;wrist=elbow+fore
        translations=np.einsum('nsij,nsjk,nsk->nsi',upstream,self.location_bases,x[:,6:]*self.length)
        hand=self.hand.apply_corrections(wrist,upstream,h,wrist_translation_world=translations)
        points=np.concatenate((self.shoulders[:,:,None]+self.tu[None,None,:,None]*upper[:,:,None],
                               elbow[:,:,None]+self.tf[None,None,:,None]*fore[:,:,None],hand),axis=2)
        upper_rot=u@self.rot[:,(0,3)];fore_rot=upstream@self.rot[:,(1,4)]
        arm_orientation=np.concatenate((np.broadcast_to(upper_rot[:,:,None],(self.n,2,10,3,3)),
                                    np.broadcast_to(fore_rot[:,:,None],(self.n,2,13,3,3))),axis=2)@self.basis[None]
        wrist_rot=upstream@h@self.rot[:,(2,5)]
        hand_orientation=wrist_rot[:,:,None]@self.hand_basis[None]
        orientation=np.concatenate((arm_orientation,hand_orientation),axis=2)
        return points,orientation,elbow,wrist+translations

    def points(self,x):return self.state(x)[0]
    def quats(self,x):return corrected_quaternions(self.quaternions,x[:,:6])


class FullLimbSolver(ClipSolver):
    def __init__(self,model,collider,matrices,progress=None):
        super().__init__(model,collider,matrices,progress)
        self.body=TightBodyField(collider,matrices,model.meta,model.length,progress)
        self.penalty=18000.;self.velocity=10.;self.acceleration=3000.
        self.pose_weights=np.array([100.,2.,.4,100.,2.,.4,100.,100.])
        self.elbow_weight=80.;self.wrist_weight=1.0;self.position_weight=.006
        self.wrist_translation_limit=.010
        self.body_weights=np.ones((model.n,2,model.p))
        self.mutual_weights=np.ones((model.n,2,model.p))
        self.own_weights=np.ones((model.n,2,model.p))
        self.body_margin=np.zeros((model.n,2,model.p))

    def prepare(self):
        yield from self.body.prepare()
        self.body.narrow_band=float(self.model.radii.max()+self.model.margin+.006)
        if hasattr(self.body,'update_active_planes'):
            self.progress('planes', 0, 1)
            self.body.update_active_planes(self.model.original_points)
            self.progress('planes', 1, 1)

    def optimize_staged(self,iterations=600,initial=None,stage_iterations=150,free_frames=None):
        delta=np.zeros((self.model.n,8,3)) if initial is None else np.asarray(initial).copy()
        stages=range(0,iterations,stage_iterations)
        self.optimization_stage_count=len(stages)
        for stage_index,start in enumerate(stages,1):
            self.optimization_stage_index=stage_index
            self.progress('solve_stage', 0, min(stage_iterations,iterations-start))
            if hasattr(self.body,'update_active_planes'):
                self.body.update_active_planes(self.model.points(delta))
            for value in self.optimize(min(stage_iterations,iterations-start),initial=delta,free_frames=free_frames):
                delta=value
                yield value
            if hasattr(self.body,'restore_full_planes'):self.body.restore_full_planes()
        self.result=delta

    def contacts(self,points,orientation,need_gradient=True):
        m=self.model;n=m.n;a=m.arm_count;p=m.p
        covariance=(orientation*(m.axes[...,None,:]**2))@orientation.swapaxes(-1,-2)
        # Gershgorin also encloses imported bases with small shear. Include
        # the same support-radius floor used below for degenerate directions.
        support_bound=np.maximum(np.sqrt(np.max(np.sum(np.abs(covariance),axis=-1),axis=-1)),1e-7)
        contact_limit=support_bound+m.margin+self.body_margin
        if need_gradient:
            mask=np.ptp(m.axes,axis=-1)>1e-8
            distance,g,hessian=self.body.distances(points,hessian=True,hessian_mask=mask,
                hessian_distance_limit=m.radii+m.margin+self.body_margin+.001,
                contact_limit=contact_limit)
        else:
            distance,g=self.body.distances(points,contact_limit=contact_limit);hessian=None
        unit=g/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-12)
        cn=np.einsum('nspij,nspj->nspi',covariance,unit)
        radius=np.sqrt(np.maximum(np.sum(cn*unit,axis=-1),1e-14))
        body=np.maximum(radius+m.margin+self.body_margin-distance,0.)
        weighted_body=body*self.body_weights
        force=-2*weighted_body[...,None]*g
        torque=np.zeros((n,2,p,3))
        if need_gradient:
            quotient=cn/radius[...,None]
            dg=(quotient-unit*radius[...,None])/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-12)
            force+=2*weighted_body[...,None]*np.einsum('nspij,nspj->nspi',hessian,dg)
            torque+=2*weighted_body[...,None]*np.cross(cn,unit)/radius[...,None]
        energy,other_force,other_torque,detail=pair_contacts(points,covariance,m.radii,
            m.margin,m.own_mask,m.arm_count,need_gradient,
            mutual_weights=self.mutual_weights,own_weights=self.own_weights)
        force+=other_force;torque+=other_torque;detail['body']=body
        return float(np.sum(body*weighted_body))+energy,force,torque,detail

    def objective(self,flat):
        m=self.model;x=flat.reshape(m.n,8,3)
        points,orientation,elbows,wrists=m.state(x)
        collision,force,torque,detail=self.contacts(points,orientation)
        factor=self.penalty/(m.length*m.length)
        energy=factor*collision;force*=factor;torque*=factor
        displacement=points-m.original_points;weight=self.position_weight/(m.length*m.length)
        energy+=weight*np.sum(displacement**2);force+=2*weight*displacement
        de=elbows-m.elbow_orig;dw=wrists-m.wrist_orig
        ew=self.elbow_weight/(m.length*m.length);ww=self.wrist_weight/(m.length*m.length)
        energy+=ew*np.sum(de*de)+ww*np.sum(dw*dw)
        ef=2*ew*de;wf=2*ww*dw
        grad=analytical_fk_gradient(m,x,points,orientation,elbows,wrists,force,torque,ef,wf)
        energy+=np.sum(self.pose_weights[None,:,None]*x*x)
        grad+=2*self.pose_weights[None,:,None]*x
        d1=np.diff(x,axis=0);d2=np.diff(x,n=2,axis=0)
        energy+=self.velocity*np.sum(d1*d1)+self.acceleration*np.sum(d2*d2)
        grad[:-1]-=2*self.velocity*d1;grad[1:]+=2*self.velocity*d1
        grad[:-2]+=2*self.acceleration*d2;grad[1:-1]-=4*self.acceleration*d2;grad[2:]+=2*self.acceleration*d2
        angles=np.linalg.norm(x,axis=-1)
        # Only a second, confirmed upper-arm/body repair can relax the strong
        # 30-degree soft threshold. Its fixed time field is C2 and the original
        # pose/elbow fidelity costs still minimize the actual movement.
        repair=np.clip((self.body_weights[:,:,:len(m.tu)]-4.)/4.,0.,1.)
        repair=repair**3*(10.-15.*repair+6.*repair**2)
        upper_limit=math.radians(30)+math.radians(5)*(1.-np.prod(1.-repair,axis=2))
        for ids,limit,penalty in (([0,3],math.radians(15),500.),([0,3],upper_limit,30000.),([1,4],math.radians(75),1000.),([2,5],math.radians(65),400.),([6,7],self.wrist_translation_limit/m.length,30000.)):
            over=np.maximum(angles[:,ids]-limit,0.)
            energy+=penalty*np.sum(over*over)
            grad[:,ids]+=2*penalty*over[...,None]*x[:,ids]/np.maximum(angles[:,ids,None],1e-12)
        self.evaluations+=1
        self.last_metrics={'energy':float(energy),'body_max_mm':float(detail['body'].max()*1000),
                           'mutual_max_mm':detail['mutual_max']*1000,'own_max_mm':detail['own_max']*1000}
        return float(energy),grad.ravel()


def full_metrics(model,solver,delta):
    points,orientation,elbows,wrists=model.state(delta)
    _,_,_,contact=solver.contacts(points,orientation,False)
    angle=np.linalg.norm(delta[:,:6],axis=-1)*180/np.pi
    def stats(value):return {'mean':float(value.mean()),'p95':float(np.percentile(value,95)),'max':float(value.max())}
    body=contact['body'];worst=np.maximum(body.max((1,2)),np.maximum(contact['mutual_frames'],contact['own_frames']))
    return {'frames':model.n,'body_max_mm':float(body.max()*1000),'mutual_max_mm':contact['mutual_max']*1000,
        'own_hand_arm_max_mm':contact['own_max']*1000,'frames_over_1mm':int(np.sum(worst>.001)),
        'frames_over_3mm':int(np.sum(worst>.003)),
        'upper_arm_degrees':stats(angle[:,(0,3)]),'forearm_degrees':stats(angle[:,(1,4)]),'wrist_degrees':stats(angle[:,(2,5)]),
        'wrist_translation_mm':stats(np.linalg.norm(delta[:,6:]*model.length,axis=-1)*1000),
        'elbow_displacement_mm':stats(np.linalg.norm(elbows-model.elbow_orig,axis=-1)*1000),
        'rotation_acceleration_deg_per_frame2':stats(np.linalg.norm(np.diff(delta[:,:6],n=2,axis=0),axis=-1)*180/np.pi),
        'worst_frame_indices':np.argsort(worst)[-20:][::-1].tolist()}
