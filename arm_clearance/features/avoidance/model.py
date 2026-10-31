"""Six-bone FK. Prepared shapes enter through LimbGeometry."""
import numpy as np
from ...numerics import project_parameters, rotation_vectors, corrected_quaternions

class LimbModel:
    parameter_count=8
    fixed_parameter_indices=(6,7)
    def __init__(self,matrices,quaternions,location_bases,geometry,meta,shapes,margin=.001):
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
        self.arm_geometry=shapes.arm_geometry
        arm_axes=np.asarray(self.arm_geometry['radii_axes']);self.basis=self.arm_geometry['basis_local']
        self.tu=np.linspace(.34,1.,10);self.tf=np.linspace(.04,.99,13);self.arm_count=23
        self.hand=shapes.hand
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
        x=project_parameters(self,x)
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

    def replace_hand_envelopes(model, hand):
        """Freeze refitted shape axes without changing centres or articulation."""
        model.hand = hand
        radii = hand.per_frame_radii
        if radii is None:
            radii = np.broadcast_to(hand.radii, (model.n,) + hand.radii.shape)
        model.hand_radii = radii
        axes = hand.ellipsoid_axes
        if axes is None:
            axes = np.repeat(radii[..., None], 3, axis=-1)
        model.axes = np.concatenate((model.axes[:, :, :model.arm_count], axes), axis=2)
        model.radii = model.axes.max(-1)

