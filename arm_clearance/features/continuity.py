"""Exact-order temporal projection without quadratic frame-space matrices.

The frame derivative Hessian is five-diagonal. Project its nonzero diagonals
in source-frame order, then skip the zero rows in each spline column. This
keeps the original left-associated (B.T @ C) @ B evaluation order on the
generic NumPy BLAS used by Blender, including its tiny roundoff asymmetry.
"""
import numpy as np
from ..source_time import cubic_basis, join_bases


def temporal_matrix(basis, velocity, acceleration, cut_indices=()):
    """Return B.T @ (2*v*D1.T@D1 + 2*a*D2.T@D2) @ B.

    basis is the existing dense, compact-support cubic spline basis. No
    frame-by-frame square array is allocated. A single controls-by-frames
    workspace replaces the old dense finite-difference matrices. In
    particular, do not symmetrize the result: eigh historically consumes
    its lower triangle and roundoff is part of the existing solver path.
    """
    basis = np.asarray(basis, dtype=np.float64)
    if basis.ndim != 2 or not basis.shape[0] or not basis.shape[1]:
        raise ValueError('basis must be a nonempty two-dimensional array')
    frames, controls = basis.shape
    velocity = float(velocity)
    acceleration = float(acceleration)
    edge_mask=np.ones(max(frames-1,0),dtype=float)
    if cut_indices:edge_mask[np.asarray(cut_indices)-1]=0.
    triples=edge_mask[:-1]*edge_mask[1:]
    first_diagonal = np.zeros(frames, dtype=np.float64)
    first_diagonal[:-1] += edge_mask
    first_diagonal[1:] += edge_mask
    second_diagonal = np.zeros(frames, dtype=np.float64)
    second_diagonal[:-2] += triples
    second_diagonal[1:-1] += 4.*triples
    second_diagonal[2:] += triples
    second_adjacent = np.zeros(max(frames-1, 0), dtype=np.float64)
    second_adjacent[:-1] -= 2.*triples
    second_adjacent[1:] -= 2.*triples
    diagonal = 2*velocity*first_diagonal + 2*acceleration*second_diagonal
    adjacent = -2*velocity*edge_mask + 2*acceleration*second_adjacent
    outer = 2*acceleration*triples

    # For any output frame, source frames are added in ascending order.
    left = np.zeros((controls, frames), dtype=np.float64)
    left[:, 2:] = basis[:-2].T * outer
    left[:, 1:] += basis[:-1].T * adjacent
    left += basis.T * diagonal
    left[:, :-1] += basis[1:].T * adjacent
    left[:, :-2] += basis[2:].T * outer
    projected = np.empty((controls, controls), dtype=np.float64)
    for column in range(controls):
        active = np.flatnonzero(basis[:, column])
        if active.size:
            projected[:, column] = left[:, active] @ basis[active, column]
        else:
            projected[:, column] = 0.
    return projected


class ContinuityTerms:
    def __init__(self):
        self.time_partition = None
        self.velocity = 10.
        self.acceleration = 3000.
        self.elbow_acceleration_weight = 40000.
        self.wrist_acceleration_weight = 80000.
        self.palm_velocity_weight = 1200.
        self.palm_acceleration_weight = 18000.
        self.wrist_excess_speed_weight = 8000.
        self.source_position_scale = .04
        self.source_orientation_scale = .12
        self.wrist_speed_ratio = 1.15
        self.wrist_speed_slack = .005
        self._source_motion_model = None
        self._source_motion_key = None

    def basis(self, frames, spacing):
        if self.time_partition is not None and self.time_partition.cuts:
            if self.time_partition.frame_count!=frames:raise ValueError('Spline source frame count differs')
            return join_bases([cubic_basis(b-a,spacing) for a,b in self.time_partition.blocks])
        return cubic_basis(frames, spacing)

    def add_cost(self, x, energy, grad):
        d1=np.diff(x,axis=0);d2=np.diff(x,n=2,axis=0)
        if self.time_partition is not None and self.time_partition.cuts:
            d1*=self.time_partition.first_mask()[:,None,None]
            d2*=self.time_partition.second_mask()[:,None,None]
        energy+=self.velocity*np.sum(d1*d1)+self.acceleration*np.sum(d2*d2)
        grad[:-1]-=2*self.velocity*d1;grad[1:]+=2*self.velocity*d1
        grad[:-2]+=2*self.acceleration*d2;grad[1:-1]-=4*self.acceleration*d2;grad[2:]+=2*self.acceleration*d2
        return energy, grad

    def _source_motion(self, m):
        key=(float(self.source_position_scale),float(self.source_orientation_scale),
             float(self.wrist_speed_ratio),float(self.wrist_speed_slack))
        if self.time_partition is not None and self.time_partition.cuts:key+=(self.time_partition.cuts,)
        if self._source_motion_model is not m or self._source_motion_key != key:
            position_scale=max(m.length*self.source_position_scale,1e-12)
            elbow_velocity=np.diff(m.elbow_orig,axis=0)
            wrist_velocity=np.diff(m.wrist_orig,axis=0)
            def position_weights(velocity):
                edges=1./(1.+np.sum(velocity*velocity,axis=-1)/(position_scale*position_scale))
                return np.minimum(edges[:-1],edges[1:])
            reference=m.original_orientation[:,:,m.arm_count]
            reference_step=np.diff(reference,axis=0)
            orientation_scale=max(self.source_orientation_scale,1e-12)
            orientation_edges=1./(1.+np.sum(reference_step*reference_step,axis=(-2,-1))/(2*orientation_scale*orientation_scale))
            elbow_weights=position_weights(elbow_velocity);wrist_weights=position_weights(wrist_velocity)
            if self.time_partition is not None and self.time_partition.cuts:
                edge_mask=self.time_partition.first_mask();triple_mask=self.time_partition.second_mask()
                elbow_weights*=triple_mask[:,None];wrist_weights*=triple_mask[:,None]
                orientation_edges*=edge_mask[:,None]
            self._source_motion_data=(elbow_weights,wrist_weights,
                orientation_edges,np.minimum(orientation_edges[:-1],orientation_edges[1:]),
                self.wrist_speed_ratio*np.linalg.norm(wrist_velocity,axis=-1)+m.length*self.wrist_speed_slack)
            self._source_motion_key=key
            self._source_motion_model=m
        return self._source_motion_data

    def add_geometry(self, m, elbows, wrists, energy, elbow_gradient, wrist_gradient):
        if not (self.elbow_acceleration_weight or self.wrist_acceleration_weight or self.wrist_excess_speed_weight):
            return energy,elbow_gradient,wrist_gradient
        elbow_weights,wrist_weights,_,_,speed_limit=self._source_motion(m)
        length2=m.length*m.length
        for current,reference,gradient,weights,strength in (
                (elbows,m.elbow_orig,elbow_gradient,elbow_weights,self.elbow_acceleration_weight),
                (wrists,m.wrist_orig,wrist_gradient,wrist_weights,self.wrist_acceleration_weight)):
            if not strength:continue
            acceleration=np.diff(current-reference,n=2,axis=0)
            weight=strength/length2
            energy+=weight*np.sum(weights[...,None]*acceleration*acceleration)
            change=2*weight*weights[...,None]*acceleration
            gradient[:-2]+=change;gradient[1:-1]-=2*change;gradient[2:]+=change
        if self.wrist_excess_speed_weight:
            velocity=np.diff(wrists,axis=0)
            speed=np.linalg.norm(velocity,axis=-1)
            excess=np.maximum(speed-speed_limit,0.)
            if self.time_partition is not None and self.time_partition.cuts:
                excess*=self.time_partition.first_mask()[:,None]
            weight=self.wrist_excess_speed_weight/length2
            energy+=weight*np.sum(excess*excess)
            change=2*weight*(excess/np.maximum(speed,1e-12))[...,None]*velocity
            wrist_gradient[:-1]-=change;wrist_gradient[1:]+=change
        return energy,elbow_gradient,wrist_gradient

    def add_orientation(self, m, orientation, energy, palm_gradient):
        if not (self.palm_velocity_weight or self.palm_acceleration_weight):
            return energy,palm_gradient
        _,_,edges,triples,_=self._source_motion(m)
        reference=m.original_orientation[:,:,m.arm_count]
        current=orientation[:,:,m.arm_count]
        # World correction relative to the moving source frame. Subtraction
        # before multiplication makes the source exactly zero even when its
        # imported float32 matrices are not exactly orthonormal. This never
        # penalizes the original motion itself or redefines joint parameters.
        correction=(current-reference)@reference.swapaxes(-1,-2)
        gradient=np.zeros_like(correction)
        for order,weights,strength in ((1,edges,self.palm_velocity_weight),
                                       (2,triples,self.palm_acceleration_weight)):
            if not strength:continue
            difference=np.diff(correction,n=order,axis=0)
            energy+=.5*strength*np.sum(weights[...,None,None]*difference*difference)
            change=strength*weights[...,None,None]*difference
            if order==1:
                gradient[:-1]-=change;gradient[1:]+=change
            else:
                gradient[:-2]+=change;gradient[1:-1]-=2*change;gradient[2:]+=change
        matrix_gradient=gradient@reference
        if palm_gradient is None:palm_gradient=matrix_gradient
        else:palm_gradient+=matrix_gradient
        return energy,palm_gradient

    def settings_report(self):
        return {'profile':'source-motion1',
            'source_elbow_acceleration_weight':self.elbow_acceleration_weight,
            'source_wrist_acceleration_weight':self.wrist_acceleration_weight,
            'world_palm_correction_velocity_weight':self.palm_velocity_weight,
            'world_palm_correction_acceleration_weight':self.palm_acceleration_weight,
            'wrist_excess_speed_weight':self.wrist_excess_speed_weight,
            'source_position_scale_limb_fraction':self.source_position_scale,
            'source_orientation_scale_radians_per_frame':self.source_orientation_scale,
            'wrist_source_speed_ratio':self.wrist_speed_ratio,
            'wrist_speed_slack_limb_fraction_per_frame':self.wrist_speed_slack,
            'limits':'soft source-relative costs; no changes to collision or bake acceptance'}

    def deviation_metrics(self, m, orientation, elbows, wrists):
        def stats(value):
            value=np.asarray(value)
            if not value.size:return {'mean':0.,'p95':0.,'max':0.}
            return {'mean':float(value.mean()),'p95':float(np.percentile(value,95)),'max':float(value.max())}
        result={}
        for name,current,reference in (('elbow',elbows,m.elbow_orig),('wrist',wrists,m.wrist_orig)):
            error=current-reference
            result[name+'_source_velocity_error_mm_per_frame']=stats(np.linalg.norm(np.diff(error,axis=0),axis=-1)*1000)
            result[name+'_source_acceleration_error_mm_per_frame2']=stats(np.linalg.norm(np.diff(error,n=2,axis=0),axis=-1)*1000)
        source=m.original_orientation[:,:,m.arm_count]
        correction=(orientation[:,:,m.arm_count]-source)@source.swapaxes(-1,-2)
        result['palm_world_correction_velocity_frobenius']=stats(np.linalg.norm(np.diff(correction,axis=0),axis=(-2,-1)))
        result['palm_world_correction_acceleration_frobenius']=stats(np.linalg.norm(np.diff(correction,n=2,axis=0),axis=(-2,-1)))
        speed=np.linalg.norm(np.diff(wrists,axis=0),axis=-1)
        source_speed=np.linalg.norm(np.diff(m.wrist_orig,axis=0),axis=-1)
        result['wrist_excess_travel_mm']=float(np.maximum(speed-source_speed,0.).sum()*1000)
        result['position_units']='armature coordinates x 1000; apply object uniform scale for world mm'
        return result

    def preconditioner(self, frames, basis, spacing, extra_knots, stationary, spans):
        temporal_key=(frames,spacing,float(self.velocity),float(self.acceleration))
        cuts=() if self.time_partition is None else self.time_partition.cuts
        if cuts:temporal_key+=('source_partition',cuts)
        if extra_knots:temporal_key+=(extra_knots,)
        temporal_cache=getattr(self,'_temporal_eigen_cache',None)
        if temporal_cache is None or temporal_cache[0]!=temporal_key:
            # Project the five nonzero frame diagonals in the original
            # summation order; avoid dense frame-by-frame D.T@D products.
            temporal=temporal_matrix(basis,self.velocity,self.acceleration,cuts)
            eigenvalues,eigenvectors=np.linalg.eigh(temporal)
            self._temporal_eigen_cache=(temporal_key,eigenvalues,eigenvectors)
        else:
            _,eigenvalues,eigenvectors=temporal_cache
        reduction=stationary.shared_reduction() if stationary is not None else None
        if reduction is not None:
            reduced_key=(temporal_key,tuple(spans))
            cached=getattr(self,'_stationary_eigen_cache',None)
            if cached is None or cached[0]!=reduced_key:
                reduced_basis=basis@reduction
                reduced_temporal=temporal_matrix(reduced_basis,self.velocity,self.acceleration,cuts)
                held_values,held_vectors=np.linalg.eigh(reduced_temporal)
                self._stationary_eigen_cache=(reduced_key,held_values,held_vectors)
            else:
                _,held_values,held_vectors=cached
        return eigenvalues, eigenvectors, reduction, (held_values, held_vectors) if reduction is not None else None
