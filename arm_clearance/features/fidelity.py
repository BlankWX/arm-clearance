"""Source retention costs, with source wrist path and world palm orientation.

The variables remain the same six LOCAL joint corrections. World orientation
returns a matrix derivative; it never changes FK, hand-envelope coordinates or
the meaning of a wrist parameter.
"""
import math
import numpy as np


class FidelityTerms:
    def __init__(self):
        self.time_partition = None
        self.pose_weights = np.array([100.,80.,80.,100.,80.,80.,100.,100.])
        self.elbow_weight = 2000.
        self.wrist_weight = 2000.
        self.palm_orientation_weight = 120.0
        self.wrist_velocity_weight = 8000.
        self.elbow_velocity_weight = 4000.
        self.source_motion_scale = .04
        self._path_source_model = None
        self._path_source_scale = None
        self._path_source_partition = None
        self.position_weight = .006
        self.wrist_translation_limit = .010

    def add_geometry(self, m, points, elbows, wrists, energy, force):
        displacement=points-m.original_points;weight=self.position_weight/(m.length*m.length)
        energy+=weight*np.sum(displacement**2);force+=2*weight*displacement
        de=elbows-m.elbow_orig;dw=wrists-m.wrist_orig
        ew=self.elbow_weight/(m.length*m.length);ww=self.wrist_weight/(m.length*m.length)
        energy+=ew*np.sum(de*de)+ww*np.sum(dw*dw)
        ef=2*ew*de;wf=2*ww*dw
        if self.wrist_velocity_weight or self.elbow_velocity_weight:
            elbow_edges,wrist_edges=self._source_motion_weights(m)
            # diff(result-source) is exactly the discrepancy from source
            # endpoint velocity. Weights depend only on the immutable source:
            # strongest at a hold, continuously weaker at fast source changes.
            for displacement,point_gradient,edges,strength in (
                    (de,ef,elbow_edges,self.elbow_velocity_weight),
                    (dw,wf,wrist_edges,self.wrist_velocity_weight)):
                velocity=np.diff(displacement,axis=0)
                weight=strength/(m.length*m.length)
                energy+=weight*np.sum(edges[...,None]*velocity*velocity)
                change=2*weight*edges[...,None]*velocity
                point_gradient[:-1]-=change;point_gradient[1:]+=change
        return energy, force, ef, wf

    def _source_motion_weights(self, m):
        if self._path_source_model is not m or self._path_source_scale!=self.source_motion_scale or self._path_source_partition is not self.time_partition:
            scale=max(m.length*self.source_motion_scale,1e-12)
            self._elbow_edges=1./(1.+np.sum(np.diff(m.elbow_orig,axis=0)**2,axis=-1)/(scale*scale))
            self._wrist_edges=1./(1.+np.sum(np.diff(m.wrist_orig,axis=0)**2,axis=-1)/(scale*scale))
            if self.time_partition is not None and self.time_partition.cuts:
                edges=self.time_partition.first_mask()[:,None]
                self._elbow_edges*=edges;self._wrist_edges*=edges
            self._path_source_model=m
            self._path_source_scale=self.source_motion_scale
            self._path_source_partition=self.time_partition
        return self._elbow_edges,self._wrist_edges

    def add_pose(self, x, energy, grad):
        energy+=np.sum(self.pose_weights[None,:,None]*x*x)
        grad+=2*self.pose_weights[None,:,None]*x
        return energy, grad

    def add_orientation(self, m, orientation, energy):
        # Every hand cell shares the same wrist frame up to its fixed local
        # basis. A single cell per side avoids multiplying this cost by the
        # number of palm/finger proxies. Both operands include the same basis.
        if self.palm_orientation_weight==0.:
            return energy,None
        reference=m.original_orientation[:,:,m.arm_count]
        current=orientation[:,:,m.arm_count]
        difference=current-reference
        energy+=.5*self.palm_orientation_weight*np.sum(difference*difference)
        # The kinematic core pulls dE/dR back through the existing matrix chain.
        # Unlike a rigid-world torque shortcut this also differentiates the
        # imported, float32-captured source frames without assuming A.T@A==I.
        return energy,self.palm_orientation_weight*difference

    def settings_report(self):
        return {'profile':'source-retention1','local_pose_weights':self.pose_weights.tolist(),
            'elbow_position_weight':self.elbow_weight,'wrist_position_weight':self.wrist_weight,
            'palm_world_orientation_weight':self.palm_orientation_weight,
            'source_wrist_velocity_weight':self.wrist_velocity_weight,
            'source_elbow_velocity_weight':self.elbow_velocity_weight,
            'source_velocity_scale_limb_fraction':self.source_motion_scale,
            'point_position_weight':self.position_weight,
            'parameterization':'unchanged six local SO(3) corrections; translation slots fixed zero'}

    def deviation_metrics(self, m, points, orientation, elbows, wrists):
        def stats(value):
            return {'mean':float(value.mean()),'p95':float(np.percentile(value,95)),
                    'max':float(value.max())}
        reference=m.original_orientation[:,:,m.arm_count]
        current=orientation[:,:,m.arm_count]
        chord=np.linalg.norm(current-reference,axis=(-2,-1))
        angles=np.degrees(2*np.arcsin(np.clip(chord/(2*np.sqrt(2.)),0.,1.)))
        orthogonality=np.max(np.abs(reference.swapaxes(-1,-2)@reference-np.eye(3)))
        return {'palm_world_deviation_degrees':stats(angles),
            'wrist_displacement_mm':stats(np.linalg.norm(wrists-m.wrist_orig,axis=-1)*1000),
            'elbow_displacement_mm':stats(np.linalg.norm(elbows-m.elbow_orig,axis=-1)*1000),
            'position_units':'armature coordinates x 1000; apply object uniform scale for world mm',
            'palm_angle_definition':'rotation-frame chord angle; exact for orthonormal source frames',
            'reference_frame_orthogonality_max_abs':float(orthogonality)}

    def add_limits(self, m, x, energy, grad, body_weights):
        angles=np.linalg.norm(x,axis=-1)
        # Only a second, confirmed upper-arm/body repair can relax the strong
        # 30-degree soft threshold. Its fixed time field is C2 and the original
        # pose/elbow fidelity costs still minimize the actual movement.
        repair=np.clip((body_weights[:,:,:len(m.tu)]-4.)/4.,0.,1.)
        repair=repair**3*(10.-15.*repair+6.*repair**2)
        upper_limit=math.radians(30)+math.radians(5)*(1.-np.prod(1.-repair,axis=2))
        for ids,limit,penalty in (([0,3],math.radians(15),500.),([0,3],upper_limit,30000.),([1,4],math.radians(75),1000.),([2,5],math.radians(65),400.),([6,7],self.wrist_translation_limit/m.length,30000.)):
            over=np.maximum(angles[:,ids]-limit,0.)
            energy+=penalty*np.sum(over*over)
            grad[:,ids]+=2*penalty*over[...,None]*x[:,ids]/np.maximum(angles[:,ids,None],1e-12)
        return energy, grad
