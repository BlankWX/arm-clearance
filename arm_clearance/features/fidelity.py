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
        self.elbow_bend_weight = 30000.
        self._bend_model = None
        self._contour_planes = []
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
        self._prepare_bend_reference(m)
        # Near extension the elbow and wrist must swing together to clear
        # the body. Preserve their velocity costs throughout that swing.
        endpoint_scale = 1. - 0.92 * np.maximum(self._extension, self._contour_blend)
        ew = ew * endpoint_scale * (1. - .6*self._contour_blend)
        ww = ww * endpoint_scale
        energy+=np.sum(ew*de*de)+np.sum(ww*dw*dw)
        ef=2*ew*de;wf=2*ww*dw
        bend_cost, elbow_force, wrist_force = self._bend_cost(m, elbows, wrists)
        energy += bend_cost
        ef += elbow_force
        wf += wrist_force
        for axes, source_offset, weight in self._contour_planes:
            upper = elbows - m.shoulders
            reach = wrists - m.shoulders
            elbow_2d = np.einsum('nfk,nsk->nsf', axes, upper)
            wrist_2d = np.einsum('nfk,nsk->nsf', axes, reach)
            offset, fraction, inverse = self._offset(elbow_2d, wrist_2d, m.length)
            residual = offset - source_offset
            # If the candidate points at the viewer, the projected chord
            # loses its direction. Fade its cost, including the fade gradient.
            normalized_reach = np.linalg.norm(wrist_2d, axis=-1, keepdims=True) / (.4*m.length)
            t = np.clip(normalized_reach, 0., 1.)
            visible = t*t*(3.-2.*t)
            energy += np.sum(weight * visible * residual**2)
            force_2d = 2 * weight * visible * residual
            axial = np.sum(force_2d * wrist_2d, axis=-1, keepdims=True) * inverse
            ge = force_2d - axial * wrist_2d
            gw = -fraction * ge - axial * offset
            gw += (np.sum(residual**2, axis=-1, keepdims=True) * weight *
                   (6*(1-t)/(.4*m.length)**2) * wrist_2d)
            ef += np.einsum('nfk,nsf->nsk', axes, ge)
            wf += np.einsum('nfk,nsf->nsk', axes, gw)
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

    @staticmethod
    def _direction(segment):
        length = np.maximum(np.linalg.norm(segment, axis=-1, keepdims=True), 1e-12)
        return segment / length, length

    @staticmethod
    def _chord(vector):
        return np.sqrt(np.sum(vector * vector, axis=-1, keepdims=True) + 1e-12)

    def _prepare_bend_reference(self, model):
        if self._bend_model is model:
            return
        upper, _ = self._direction(model.upper)
        fore, _ = self._direction(model.lower)
        self._source_chord = self._chord(upper - fore)
        # Use the original bend on each arm, independent of the current iterate.
        # Blend out the extended-arm profile between 10 and 30 degrees.
        lo, hi = 2 * np.sin(np.radians([5., 15.]))
        t = np.clip((hi - self._source_chord) / (hi - lo), 0., 1.)
        self._extension = t*t*(3. - 2.*t)
        self._prepare_contour(model)
        self._bend_model = model

    @staticmethod
    def _ramp(value):
        t = np.clip(value, 0., 1.)
        return t*t*(3.-2.*t)

    @staticmethod
    def _offset(elbow, wrist, length):
        # Regularization prevents a singular derivative for a foreshortened arm.
        inverse = 1. / (np.sum(wrist*wrist, axis=-1, keepdims=True) + (.04*length)**2)
        fraction = np.sum(elbow*wrist, axis=-1, keepdims=True) * inverse
        return elbow - wrist*fraction, fraction, inverse

    def _prepare_contour(self, model):
        self._contour_planes = []
        self._contour_blend = np.zeros_like(self._extension)
        meta = getattr(model, 'meta', {})
        names = meta.get('bones', ())
        # Resolve the actual hip pivots; never substitute feet or IK controls.
        hips = next(((names.index(l), names.index(r)) for l, r in
                     [('足.L','足.R'), ('左足','右足'), ('thigh.L','thigh.R'),
                      ('LeftUpLeg','RightUpLeg'), ('mixamorig:LeftUpLeg','mixamorig:RightUpLeg')]
                     if l in names and r in names), None)
        if hips is None:
            return
        parents = meta.get('parents', ())
        if len(parents) != len(names) or parents[hips[0]] != parents[hips[1]]:
            return
        hip_center = .5 * (model.matrices[:, hips[0], :3, 3] + model.matrices[:, hips[1], :3, 3])
        lateral, width = self._direction(model.shoulders[:,0] - model.shoulders[:,1])
        spine = model.shoulders.mean(axis=1) - hip_center
        vertical, height = self._direction(spine - lateral*np.sum(spine*lateral, axis=-1, keepdims=True))
        depth = np.cross(vertical, lateral)
        confidence = self._ramp(np.minimum(width,height)/model.length/.3)
        chord = self._source_chord
        # The near-straight solver already preserves very small bends. Blend the
        # directional correction in for open arms, out again for folded arms.
        blend = self._ramp((chord-.1)/.1) * self._ramp((.85-chord)/.3)
        front_axes = np.stack((lateral, vertical), axis=1)
        upper = np.einsum('nfk,nsk->nsf', front_axes, model.upper)
        reach = np.einsum('nfk,nsk->nsf', front_axes, model.upper+model.lower)
        offset, _, _ = self._offset(upper, reach, model.length)
        straight = self._ramp((.18-np.linalg.norm(offset,axis=-1,keepdims=True)/model.length)/.12)
        self._contour_blend = blend * confidence[:,None,:] * straight
        for horizontal, strength in ((lateral, 68400.0), (depth, 34200.0)):
            axes = np.stack((horizontal, vertical), axis=1)
            e = np.einsum('nfk,nsk->nsf', axes, model.upper)
            w = np.einsum('nfk,nsk->nsf', axes, model.upper+model.lower)
            ref, _, _ = self._offset(e, w, model.length)
            visibility = self._ramp(np.linalg.norm(w,axis=-1,keepdims=True)/model.length/.45)
            weight = strength/model.length**2 * self._contour_blend * visibility
            self._contour_planes.append((axes, ref, weight))

    def _bend_cost(self, model, elbows, wrists):
        # The chord between unit directions is 2*sin(bend/2). It measures
        # bend independently of arm direction and avoids acos near extension.
        self._prepare_bend_reference(model)
        upper, upper_length = self._direction(elbows - model.shoulders)
        fore, fore_length = self._direction(wrists - elbows)
        delta = upper - fore
        magnitude = self._chord(delta)
        error = magnitude - self._source_chord
        weight = 100. + (self.elbow_bend_weight - 100.) * self._extension
        derivative = (2 * weight * error / magnitude) * delta

        def project(unit, length):
            return (derivative - unit * np.sum(unit * derivative, axis=-1, keepdims=True)) / length

        wrist_gradient = -project(fore, fore_length)
        elbow_gradient = project(upper, upper_length) - wrist_gradient
        return np.sum(weight * error * error), elbow_gradient, wrist_gradient

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
        self._prepare_bend_reference(m)
        weight = self.palm_orientation_weight * (1. + 11.5 * np.maximum(self._extension,self._contour_blend)[..., None])
        energy+=.5*np.sum(weight*difference*difference)
        # The kinematic core pulls dE/dR back through the existing matrix chain.
        # Unlike a rigid-world torque shortcut this also differentiates the
        # imported, float32-captured source frames without assuming A.T@A==I.
        return energy,weight*difference

    def settings_report(self):
        return {'profile':'source-retention1','local_pose_weights':self.pose_weights.tolist(),
            'elbow_position_weight':self.elbow_weight,'wrist_position_weight':self.wrist_weight,
            'elbow_bend_chord_weight':self.elbow_bend_weight,
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
