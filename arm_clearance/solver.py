"""Deterministic, whole-clip arm clearance solver. Only NumPy + Blender mathutils."""
import json
import math
import time
import numpy as np
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def rotation_vectors(v):
    """SO(3) exponential map, including a stable zero-angle limit."""
    theta = np.linalg.norm(v, axis=-1)
    a = np.sinc(theta / np.pi)
    b = .5 * np.sinc(theta / (2 * np.pi)) ** 2
    k = np.zeros(v.shape[:-1] + (3, 3), dtype=np.float64)
    k[..., 0, 1] = -v[..., 2]; k[..., 0, 2] = v[..., 1]
    k[..., 1, 0] = v[..., 2]; k[..., 1, 2] = -v[..., 0]
    k[..., 2, 0] = -v[..., 1]; k[..., 2, 1] = v[..., 0]
    return np.eye(3) + a[..., None, None] * k + b[..., None, None] * (k @ k)


def quaternion_multiply(a, b):
    w = a[..., :1]*b[..., :1] - np.sum(a[..., 1:]*b[..., 1:], axis=-1, keepdims=True)
    xyz = a[..., :1]*b[..., 1:] + b[..., :1]*a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate((w, xyz), axis=-1)


def corrected_quaternions(original, delta):
    angle = np.linalg.norm(delta, axis=-1, keepdims=True)
    q = np.concatenate((np.cos(angle/2), .5*np.sinc(angle/(2*np.pi))*delta), axis=-1)
    result = quaternion_multiply(original, q)
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    # A quaternion sign change describes the same pose but breaks interpolation.
    for f in range(1, len(result)):
        result[f] *= np.where(np.sum(result[f]*result[f-1], axis=-1, keepdims=True)<0, -1., 1.)
    return result


def skin_vertices(vertices, joints, weights, rest, pose):
    transform = pose @ np.linalg.inv(rest)
    out = np.zeros((len(vertices), 3), np.float64)
    for k in range(joints.shape[1]):
        m = transform[joints[:, k]]
        out += (np.einsum('nij,nj->ni', m[:, :3, :3], vertices) + m[:, :3, 3]) * weights[:, k, None]
    return out


class ClipModel:
    def __init__(self, matrices, target_ids, wrist_ids, radii, margin=.003):
        self.matrices = np.asarray(matrices, np.float64)
        self.target_ids = target_ids
        self.n = len(matrices)
        self.shoulders = np.take(self.matrices, target_ids[::2], axis=1)[:, :, :3, 3]
        elbows = np.take(self.matrices, target_ids[1::2], axis=1)[:, :, :3, 3]
        wrists = np.take(self.matrices, wrist_ids, axis=1)[:, :, :3, 3]
        self.upper = elbows - self.shoulders
        self.lower = wrists - elbows
        self.rot = np.take(self.matrices, target_ids, axis=1)[:, :, :3, :3].copy()
        # Ignore object/bone scale when turning an axis into an orientation.
        self.rot /= np.maximum(np.linalg.norm(self.rot, axis=-2, keepdims=True), 1e-12)
        self.length = float(np.mean(np.linalg.norm(self.upper, axis=-1)))
        self.tu = np.linspace(.34, 1., 10)
        self.tf = np.linspace(.04, .99, 13)
        self.t = np.concatenate((self.tu, self.tf))
        self.radii = np.asarray(radii)
        self.margin = margin
        self.original_points = self.points(np.zeros((self.n, 4, 3)))

    def points(self, delta):
        local = rotation_vectors(delta)
        world = self.rot @ local @ np.swapaxes(self.rot, -1, -2)
        up = np.einsum('nsij,nsj->nsi', world[:, ::2], self.upper)
        fore = np.einsum('nsij,nsjk,nsk->nsi', world[:, ::2], world[:, 1::2], self.lower)
        elbows = self.shoulders + up
        u = self.shoulders[:, :, None] + self.tu[None, None, :, None] * up[:, :, None]
        f = elbows[:, :, None] + self.tf[None, None, :, None] * fore[:, :, None]
        return np.concatenate((u, f), axis=2)


class ClipSolver:
    def __init__(self, model, collider, matrices, progress=None):
        self.model = model
        self.collider = collider
        self.matrices = matrices
        self.trees = []
        self.progress = progress or (lambda *args: None)
        self.evaluations = 0
        self.penalty = 15000.
        self.velocity = 10.0
        self.acceleration = 3000.
        self.pose_weight = .025
        self.position_weight = .10
        self.history = []
        self.hulls = []
        self.hull_bounds = []

    def prepare(self):
        c = self.collider
        # A union of overlapping convex anatomical slices has a continuous
        # inside/outside field even when the input skin has holes, duplicate
        # shells or intersecting internal surfaces.
        v=c['vertices'];low=float(v[:,2].min());height=float(np.ptp(v[:,2]))
        z=(v[:,2]-low)/height
        regions=[]
        for a,b in ((0.,.32),(.27,.55)):
            for side in (-1,1):regions.append(np.flatnonzero((z>=a)&(z<=b)&(v[:,0]*side>=0)))
        for a,b in ((.48,.65),(.59,.72),(.67,.79),(.75,.86),(.81,1.01)):
            regions.append(np.flatnonzero((z>=a)&(z<=b)))
        self.region_ids=[r for r in regions if len(r)>=8]
        plane_lists=[[] for r in self.region_ids]
        bound_lists=[[] for r in self.region_ids]
        # Amortize BVH construction over the complete optimization.
        for f, pose in enumerate(self.matrices):
            v = skin_vertices(c['vertices'], c['joint'], c['weight'], c['rest'], pose)
            self.trees.append(BVHTree.FromPolygons(v.tolist(), c['triangles'].tolist(), all_triangles=True))
            for j,ids in enumerate(self.region_ids):
                bound_lists[j].append((v[ids].min(0),v[ids].max(0)))
                bm=bmesh.new()
                try:
                    verts=[bm.verts.new(co) for co in v[ids]]
                    hull=bmesh.ops.convex_hull(bm,input=verts,use_existing_faces=False)
                    faces=[x for x in hull['geom'] if isinstance(x,bmesh.types.BMFace)]
                    bmesh.ops.recalc_face_normals(bm,faces=faces)
                    bm.normal_update()
                    center=v[ids].mean(0)
                    planes=[]
                    for face in faces:
                        n=np.array(face.normal);co=np.array(face.verts[0].co)
                        if np.dot(n,co-center)<0:n=-n
                        planes.append(np.r_[n,-np.dot(n,co)])
                    plane_lists[j].append(np.array(planes))
                finally:bm.free()
            if f % 20 == 0:
                self.progress('colliders', f, len(self.matrices))
                yield None
        for lists in plane_lists:
            size=max(len(v) for v in lists)
            planes=np.zeros((len(lists),size,4));planes[:,:,3]=-1e4
            count=np.empty(len(lists))
            for i,v in enumerate(lists):planes[i,:len(v)]=v;count[i]=len(v)
            self.hulls.append((planes,count))
        self.hull_bounds=[np.array(b) for b in bound_lists]

    def body_contact(self,points):
        # Smooth plane maxima and smooth union minima keep the contact normal
        # continuous across facets and overlapping slice boundaries.
        n,s,p,_=points.shape;flat=points.reshape(n,s*p,3)
        distances=[];normals=[];tau=self.model.length*.004
        for (planes,count),bounds in zip(self.hulls,self.hull_bounds):
            padding=float(self.model.radii.max()+self.model.margin+.02)
            active=((flat>=bounds[:,None,0]-padding)&(flat<=bounds[:,None,1]+padding)).all(2).any(1)
            if not active.any():continue
            val=np.einsum('npi,nfi->npf',flat[active],planes[active,:,:3])+planes[active,None,:,3]
            mx=val.max(2);weights=np.exp((val-mx[:,:,None])/tau)
            total=weights.sum(2)
            d=mx+tau*np.log(total/count[active,None])
            normal=np.einsum('npf,nfi->npi',weights,planes[active,:,:3])/total[:,:,None]
            full_d=np.full((n,s*p),1e3);full_n=np.zeros((n,s*p,3))
            full_d[active]=d;full_n[active]=normal
            distances.append(full_d);normals.append(full_n)
        if not distances:
            return np.zeros(points.shape[:-1]),np.zeros_like(points)
        distances=np.stack(distances,axis=2);normals=np.stack(normals,axis=2)
        mn=distances.min(2);union_tau=self.model.length*.004
        w=np.exp((mn[:,:,None]-distances)/union_tau);total=w.sum(2)
        distance=mn-union_tau*np.log(total)
        normal=np.sum(w[:,:,:,None]*normals,axis=2)/total[:,:,None]
        depth=np.maximum(self.model.radii[None]+self.model.margin-distance.reshape(n,s,p),0.)
        return depth,normal.reshape(n,s,p,3)

    def contact(self, points, need_force=True):
        m = self.model
        depth,normals=self.body_contact(points)
        diff = points[:,0,:,None,:] - points[:,1,None,:,:]
        dist = np.linalg.norm(diff,axis=-1)
        threshold = m.radii[0,:,None] + m.radii[1,None,:] + m.margin
        self_depth = np.maximum(threshold[None] - dist, 0.)
        force = -2*depth[...,None]*normals
        pair_force = -2*self_depth[...,None]*diff/np.maximum(dist[...,None],1e-10)
        force[:,0] += pair_force.sum(2)
        force[:,1] -= pair_force.sum(1)
        energy = np.sum(depth**2) + np.sum(self_depth**2)
        return energy, force, depth, self_depth

    def objective(self, flat):
        m = self.model
        x = flat.reshape((m.n, 4, 3))
        points = m.points(x)
        collision, force, body, mutual = self.contact(points)
        factor = self.penalty / (m.length*m.length)
        energy = factor*collision
        point_force = factor*force
        displacement = points-m.original_points
        w = self.position_weight / (m.length*m.length)
        energy += w*np.sum(displacement**2)
        point_force += 2*w*displacement
        # Vectorized FK differences: contacts are differentiated with their
        # geometrical normals rather than twelve extra collision queries.
        grad = np.empty_like(x)
        eps = 1e-5
        for bone in range(4):
            for axis in range(3):
                perturbed = x.copy(); perturbed[:,bone,axis] += eps
                jacobian = (m.points(perturbed)-points)/eps
                grad[:,bone,axis] = np.sum(point_force*jacobian,axis=(1,2,3))
        energy += self.pose_weight*np.sum(x*x)
        grad += 2*self.pose_weight*x
        d1 = np.diff(x,axis=0)
        energy += self.velocity*np.sum(d1*d1)
        grad[:-1] -= 2*self.velocity*d1; grad[1:] += 2*self.velocity*d1
        d2 = np.diff(x,n=2,axis=0)
        energy += self.acceleration*np.sum(d2*d2)
        grad[:-2] += 2*self.acceleration*d2
        grad[1:-1] -= 4*self.acceleration*d2
        grad[2:] += 2*self.acceleration*d2
        # A soft anatomical correction limit, leaving the original bend intact.
        angle = np.linalg.norm(x,axis=-1)
        over = np.maximum(angle-math.radians(70),0)
        energy += 300*np.sum(over**2)
        grad += 600*over[...,None]*x/np.maximum(angle[...,None],1e-10)
        self.evaluations += 1
        self.last_metrics = {'body_max_mm':float(body.max()*1000),
            'mutual_max_mm':float(mutual.max()*1000),'energy':float(energy)}
        return float(energy), grad.ravel()

    def optimize(self, iterations=220, initial=None, knot_spacing=8, free_frames=None):
        """C2 control-space L-BFGS, optionally freezing unrelated controls.

        A control is free when its positive basis support intersects any true
        ``free_frames`` entry.  Its complete bone/axis block is free together;
        the remaining controls retain the initial full-clip spline projection.
        The objective and temporal terms still cover the complete frame range.
        """
        m = self.model
        parameters=getattr(m,'parameter_count',4)
        dimension=parameters*3
        # The unknowns are controls of a C2 cubic spline. Collision correction
        # cannot acquire independent per-frame spikes, even before smoothing.
        spacing=max(1,int(knot_spacing))
        t=np.arange(m.n)/spacing;k=np.floor(t).astype(int);u=t-k
        weights=np.stack(((1-u)**3,3*u**3-6*u*u+4,-3*u**3+3*u*u+3*u+1,u**3),axis=1)/6
        basis=np.zeros((m.n,int(k.max())+4))
        basis[np.arange(m.n)[:,None],k[:,None]+np.arange(4)]=weights
        controls=basis.shape[1]
        x=np.zeros((controls,dimension)).ravel() if initial is None else np.linalg.lstsq(basis,initial.reshape(m.n,dimension),rcond=None)[0].ravel()
        free_variables=None
        free_controls=np.ones(controls,dtype=bool)
        if free_frames is not None:
            free_frames=np.asarray(free_frames,dtype=bool)
            if free_frames.shape!=(m.n,):
                raise ValueError('free_frames must be a boolean mask of shape [frames]')
            free_controls=np.any(basis[free_frames]>0.,axis=0)
            free_variables=np.repeat(free_controls,dimension)
        def frames(var):return (basis@var.reshape(controls,dimension)).reshape(m.n,parameters,3)
        self.editable_frame_mask=np.any(basis[:,free_controls]>0.,axis=1)
        editable_indices=np.flatnonzero(self.editable_frame_mask)
        locked_reference=None if free_variables is None else x[~free_variables].copy()
        self.last_optimization_scope={
            'masked':free_frames is not None,
            'requested_free_frames':m.n if free_frames is None else int(np.count_nonzero(free_frames)),
            'free_controls':int(np.count_nonzero(free_controls)),'total_controls':controls,
            'editable_frame_count':int(len(editable_indices)),
            'editable_frame_index_range':([int(editable_indices[0]),int(editable_indices[-1])] if len(editable_indices) else None),
            'projection_max_abs_error':(0. if initial is None else float(np.max(np.abs(frames(x)-initial)))),
            'locked_control_drift_max_abs':0.}
        def objective(var):
            value,gradient=self.objective(frames(var).ravel())
            projected=(basis.T@gradient.reshape(m.n,dimension)).ravel()
            if free_variables is not None:projected[~free_variables]=0.
            return value,projected
        value, grad = objective(x)
        history = []
        # Use the temporal quadratic as the initial inverse Hessian. This
        # lets a contact move a whole smooth interval instead of a single
        # frame, and avoids extremely slow convergence with stiff smoothing.
        # Staged/warm solves reuse exactly the same original decomposition.
        # Keep only one entry and rebuild whenever any defining input changes.
        temporal_key=(m.n,spacing,float(self.velocity),float(self.acceleration))
        temporal_cache=getattr(self,'_temporal_eigen_cache',None)
        if temporal_cache is None or temporal_cache[0]!=temporal_key:
            d1=np.diff(np.eye(m.n),axis=0)
            d2=np.diff(np.eye(m.n),n=2,axis=0)
            temporal=basis.T@(2*self.velocity*(d1.T@d1)+2*self.acceleration*(d2.T@d2))@basis
            eigenvalues,eigenvectors=np.linalg.eigh(temporal)
            self._temporal_eigen_cache=(temporal_key,eigenvalues,eigenvectors)
        else:
            _,eigenvalues,eigenvectors=temporal_cache
        for iteration in range(iterations):
            q = grad.copy(); alpha = []
            for s,y,rho in reversed(history):
                a = rho*np.dot(s,q);alpha.append(a);q-=a*y
            scale = np.dot(history[-1][0],history[-1][1])/max(np.dot(history[-1][1],history[-1][1]),1e-20) if history else 1e-3
            direction=(eigenvectors@((eigenvectors.T@q.reshape(controls,dimension))/(1./max(scale,1e-12)+eigenvalues[:,None]))).ravel()
            for (s,y,rho),a in zip(history,reversed(alpha)):
                direction+=s*(a-rho*np.dot(y,direction))
            direction=-direction
            if free_variables is not None:direction[~free_variables]=0.
            slope=np.dot(grad,direction)
            if slope>=0:
                direction=-grad*1e-3; slope=np.dot(grad,direction);history=[]
            step=min(1.,.15/max(np.abs(direction).max(),1e-10))
            accepted=False
            for trial in range(16):
                candidate=x+step*direction
                if free_variables is not None:candidate[~free_variables]=x[~free_variables]
                nv,ng=objective(candidate)
                if np.isfinite(nv) and nv<=value+1e-4*step*slope:
                    accepted=True;break
                step*=.5
            if not accepted: break
            s=candidate-x;y=ng-grad;sy=np.dot(s,y)
            if sy>1e-10:
                history.append((s,y,1./sy));history=history[-16:]
            x,value,grad=candidate,nv,ng
            if iteration%5==0:
                self.progress('solve',iteration,iterations,self.last_metrics)
                self.history.append(dict(iteration=iteration,**self.last_metrics))
                yield frames(x).copy()
            if np.max(np.abs(grad))<1e-5: break
        self.result=frames(x)
        if locked_reference is not None and len(locked_reference):
            self.last_optimization_scope['locked_control_drift_max_abs']=float(np.max(np.abs(x[~free_variables]-locked_reference)))
        yield self.result


def estimate_radii(geometry, metadata, samples_upper=10, samples_lower=13):
    v=geometry['vertices']; rest=geometry['rest'];dom=geometry['dominant']
    names=metadata['bones'];parents=metadata['parents']
    radii=[]
    wrists=metadata.get('wrist_ids') or [names.index('手首.L'),names.index('手首.R')]
    for side in range(2):
        ur=metadata['target_ids'][side*2];fr=metadata['target_ids'][side*2+1];wr=wrists[side]
        class_id=np.full(len(names),-1)
        for b in range(len(names)):
            p=b
            while p>=0:
                if p==wr:break
                if p==fr:class_id[b]=1;break
                if p==ur:class_id[b]=0;break
                p=parents[p]
        values=[]
        for segment,(a,b,ts) in enumerate(((ur,fr,np.linspace(.34,1.,samples_upper)),(fr,wr,np.linspace(.04,.99,samples_lower)))):
            h=rest[a,:3,3];u=rest[b,:3,3]-h
            t=(v-h)@u/(u@u);rad=np.linalg.norm(v-h-t[:,None]*u,axis=1)
            valid=(class_id[dom]==segment)
            for x in ts:
                selected=rad[valid&(abs(t-x)<.12)]
                radius=float(np.percentile(selected,95)) if len(selected) else .025
                values.append(min(radius,.055))
        radii.append(values)
    return np.array(radii)


def metrics(model, solver, delta):
    points=model.points(delta);_,_,body,mutual=solver.contact(points,False)
    d1=np.linalg.norm(np.diff(delta,axis=0),axis=-1)*180/np.pi
    d2=np.linalg.norm(np.diff(delta,n=2,axis=0),axis=-1)*180/np.pi
    d3=np.linalg.norm(np.diff(delta,n=3,axis=0),axis=-1)*180/np.pi
    max_depth=np.maximum(body.max(axis=(1,2)),mutual.max(axis=(1,2)))
    return {'frames':model.n,'body_max_mm':float(body.max()*1000),'mutual_max_mm':float(mutual.max()*1000),
        'frames_over_1mm':int(np.sum(max_depth>.001)),
        'frames_over_3mm':int(np.sum(max_depth>.003)),
        'worst_frames_indices':np.argsort(max_depth)[-20:][::-1].tolist(),
        'max_correction_degrees':float(np.linalg.norm(delta,axis=-1).max()*180/np.pi),
        'correction_speed_deg_per_frame':{'max':float(d1.max()),'p95':float(np.percentile(d1,95))},
        'correction_acceleration_deg_per_frame2':{'max':float(d2.max()),'p95':float(np.percentile(d2,95))},
        'correction_jerk_deg_per_frame3':{'max':float(d3.max()) if d3.size else 0.,'p95':float(np.percentile(d3,95)) if d3.size else 0.},
        'curve_model':'C2 cubic B-spline, 8-frame control spacing; optimization couples both arms and the complete frame interval'}
