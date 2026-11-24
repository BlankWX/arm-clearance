"""Proxy evaluation, geometry fitting and real verification ports."""
import numpy as np
from ...contracts import CollisionFields, LimbGeometry
from .body_field import TightBodyField
from .arm_geometry import fit_arm_ellipses
from .hand_geometry import build_hand_geometry, refit_corrected_hand_envelopes
from .pair_contacts import PAIR_CHUNK_SIZE, pair_contacts


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
        self._local_stage = False
        self._reference = None

    def set_fields(self, fields):
        self._reference = None
        self.fields = fields

    def restore_full_planes(self):
        self._reference = None
        self._local_stage = False
        self.body.restore_full_planes()

    def body_report(self):
        return self.body.report

    def begin_stage(self, delta, local=False):
        self._reference = None
        self._local_stage = local
        if hasattr(self.body, 'update_active_planes'):
            self.body.update_active_planes(self.model.points(delta))

    def end_stage(self):
        self._reference = None
        self._local_stage = False
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

    def evaluate(self, points, orientation, need_gradient=True, pose_parameters=None):
        if not self._local_stage or pose_parameters is None or not need_gradient:
            body_energy, energy, force, torque, detail = self._evaluate_rows(
                points, orientation, need_gradient)
            return float(np.sum(body_energy))+energy, force, torque, detail

        # All model inputs except pose are fixed until this optimization stage
        # ends. Compare the actual projected pose so stationary/control support
        # expansion and even tiny changes outside the requested mask are safe.
        if self._reference is None or self._reference[0] is not self.model.axes:
            stats = []
            body_energy, energy, force, torque, detail = self._evaluate_rows(
                points, orientation, True, stats=stats)
            stored_detail = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in detail.items()}
            self._reference = (self.model.axes, pose_parameters.copy(), body_energy, force.copy(),
                               torque.copy(), stored_detail, stats)
            return float(np.sum(body_energy))+energy, force, torque, detail

        _, initial, stored_body, stored_force, stored_torque, stored_detail, stored_stats = self._reference
        changed = np.any(pose_parameters != initial, axis=(1,2))
        # Pair energy uses 3 ordered reductions per original 32-frame chunk.
        # Recompute complete touched chunks and replay that exact sum order.
        starts = np.arange(0, len(changed), PAIR_CHUNK_SIZE)
        chunks = np.flatnonzero(np.logical_or.reduceat(changed, starts))
        if len(chunks) == len(starts):
            values, energy, force, torque, detail = self._evaluate_rows(points, orientation, True)
            return float(np.sum(values))+energy, force, torque, detail
        body_energy = stored_body.copy()
        force = stored_force.copy()
        torque = stored_torque.copy()
        detail = dict(stored_detail)
        for field in ('body', 'mutual_frames', 'own_frames'):
            detail[field] = stored_detail[field].copy()
        stats = list(stored_stats)
        if len(chunks):
            rows = np.flatnonzero(np.repeat(np.isin(np.arange(len(starts)), chunks),
                                            PAIR_CHUNK_SIZE)[:len(changed)])
            updated = []
            values, _, f, t, local = self._evaluate_rows(points[rows], orientation[rows],
                                                        True, rows=rows, stats=updated)
            body_energy[rows], force[rows], torque[rows] = values, f, t
            for field in ('body', 'mutual_frames', 'own_frames'):
                detail[field][rows] = local[field]
            for j, chunk in enumerate(chunks):
                stats[3*chunk:3*chunk+3] = updated[3*j:3*j+3]
        detail.update(mutual_max=float(detail['mutual_frames'].max()),
                      own_max=float(detail['own_frames'].max()),
                      pair_candidates=sum(item[1] for item in stats),
                      positive_pairs=sum(item[2] for item in stats))
        pair_energy = 0.
        for contribution, _, _ in stats:
            pair_energy += contribution
        return float(np.sum(body_energy))+pair_energy, force, torque, detail

    def _evaluate_rows(self,points,orientation,need_gradient,rows=None,stats=None):
        m=self.model
        n, _, p, _ = points.shape
        selection = slice(None) if rows is None else rows
        axes = m.axes[selection]
        radii = m.radii[selection]
        margin = self.fields.body_margin[selection]
        body_weights = self.fields.body_weights[selection]
        covariance=(orientation*(axes[...,None,:]**2))@orientation.swapaxes(-1,-2)
        # Gershgorin also encloses imported bases with small shear. Include
        # the same support-radius floor used below for degenerate directions.
        support_bound=np.maximum(np.sqrt(np.max(np.sum(np.abs(covariance),axis=-1),axis=-1)),1e-7)
        contact_limit=support_bound+m.margin+margin
        if need_gradient:
            mask=np.ptp(axes,axis=-1)>1e-8
            distance,g,hessian=self.body.distances(points,hessian=True,hessian_mask=mask,
                hessian_distance_limit=radii+m.margin+margin+.001,
                contact_limit=contact_limit,frame_rows=rows)
        else:
            distance,g=self.body.distances(points,contact_limit=contact_limit,frame_rows=rows);hessian=None
        unit=g/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-12)
        cn=np.einsum('nspij,nspj->nspi',covariance,unit)
        radius=np.sqrt(np.maximum(np.sum(cn*unit,axis=-1),1e-14))
        body=np.maximum(radius+m.margin+margin-distance,0.)
        weighted_body=body*body_weights
        force=-2*weighted_body[...,None]*g
        torque=np.zeros((n,2,p,3))
        if need_gradient:
            quotient=cn/radius[...,None]
            dg=(quotient-unit*radius[...,None])/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-12)
            force+=2*weighted_body[...,None]*np.einsum('nspij,nspj->nspi',hessian,dg)
            torque+=2*weighted_body[...,None]*np.cross(cn,unit)/radius[...,None]
        energy,other_force,other_torque,detail=pair_contacts(points,covariance,radii,
            m.margin,m.own_mask,m.arm_count,need_gradient,
            mutual_weights=self.fields.mutual_weights[selection],own_weights=self.fields.own_weights[selection],
            chunk_stats=stats)
        force+=other_force;torque+=other_torque;detail['body']=body
        return body*weighted_body,energy,force,torque,detail
