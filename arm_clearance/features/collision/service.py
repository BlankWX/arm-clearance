"""Proxy evaluation, geometry fitting and real verification ports."""
import numpy as np
from ...contracts import CollisionFields, LimbGeometry
from .body_field import TightBodyField
from .arm_geometry import fit_arm_ellipses
from .hand_geometry import build_hand_geometry, refit_corrected_hand_envelopes
from .pair_contacts import pair_contacts


def prepare_limb_geometry(geometry, meta, matrices, hand=None,time_partition=None):
    arm_meta = dict(meta, target_ids=list(meta['arm_target_ids']))
    arm_geometry = fit_arm_ellipses(geometry, arm_meta, axial_fraction=.5)
    return LimbGeometry(arm_geometry, hand or build_hand_geometry(geometry, meta, matrices,time_partition=time_partition))


class CollisionSystem:
    def __init__(self, model, collider, matrices, progress=None, surface_checker=None):
        self.model = model
        self.collider = collider
        self.progress = progress or (lambda *args: None)
        self.body = TightBodyField(collider, matrices, model.meta, model.length, self.progress)
        self.fields = CollisionFields.baseline(model.n, model.p)
        self.surface_checker = surface_checker

    def set_fields(self, fields):
        self.fields = fields

    def restore_full_planes(self):
        self.body.restore_full_planes()

    def body_report(self):
        return self.body.report

    def begin_stage(self, delta):
        if hasattr(self.body, 'update_active_planes'):
            self.body.update_active_planes(self.model.points(delta))

    def end_stage(self):
        if hasattr(self.body, 'restore_full_planes'):
            self.body.restore_full_planes()

    def update_repair_band(self):
        self.body.narrow_band = float(np.max(self.model.radii + self.model.margin + self.fields.body_margin) + .006)
        return self.body.narrow_band

    def refit(self, delta, clip):
        return refit_corrected_hand_envelopes(self.model.hand, self.model, delta, clip.geometry,
            meta=clip.meta, matrices=clip.matrices, progress=self.progress,time_partition=clip.time_partition)

    def verify(self, *args, **kwargs):
        if self.surface_checker is None:
            raise RuntimeError('No Blender surface checker was bound')
        return self.surface_checker(*args, **kwargs)

    def prepare(self):
        yield from self.body.prepare()
        self.body.narrow_band=float(self.model.radii.max()+self.model.margin+.006)
        if hasattr(self.body,'update_active_planes'):
            self.progress('planes', 0, 1)
            self.body.update_active_planes(self.model.original_points)
            self.progress('planes', 1, 1)

    def evaluate(self,points,orientation,need_gradient=True):
        m=self.model;n=m.n;a=m.arm_count;p=m.p
        covariance=(orientation*(m.axes[...,None,:]**2))@orientation.swapaxes(-1,-2)
        # Gershgorin also encloses imported bases with small shear. Include
        # the same support-radius floor used below for degenerate directions.
        support_bound=np.maximum(np.sqrt(np.max(np.sum(np.abs(covariance),axis=-1),axis=-1)),1e-7)
        contact_limit=support_bound+m.margin+self.fields.body_margin
        if need_gradient:
            mask=np.ptp(m.axes,axis=-1)>1e-8
            distance,g,hessian=self.body.distances(points,hessian=True,hessian_mask=mask,
                hessian_distance_limit=m.radii+m.margin+self.fields.body_margin+.001,
                contact_limit=contact_limit)
        else:
            distance,g=self.body.distances(points,contact_limit=contact_limit);hessian=None
        unit=g/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-12)
        cn=np.einsum('nspij,nspj->nspi',covariance,unit)
        radius=np.sqrt(np.maximum(np.sum(cn*unit,axis=-1),1e-14))
        body=np.maximum(radius+m.margin+self.fields.body_margin-distance,0.)
        weighted_body=body*self.fields.body_weights
        force=-2*weighted_body[...,None]*g
        torque=np.zeros((n,2,p,3))
        if need_gradient:
            quotient=cn/radius[...,None]
            dg=(quotient-unit*radius[...,None])/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-12)
            force+=2*weighted_body[...,None]*np.einsum('nspij,nspj->nspi',hessian,dg)
            torque+=2*weighted_body[...,None]*np.cross(cn,unit)/radius[...,None]
        energy,other_force,other_torque,detail=pair_contacts(points,covariance,m.radii,
            m.margin,m.own_mask,m.arm_count,need_gradient,
            mutual_weights=self.fields.mutual_weights,own_weights=self.fields.own_weights)
        force+=other_force;torque+=other_torque;detail['body']=body
        return float(np.sum(body*weighted_body))+energy,force,torque,detail
