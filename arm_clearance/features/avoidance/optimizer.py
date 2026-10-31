"""Original control-space L-BFGS loop; injected temporal and hold ports."""
import numpy as np
from ...numerics import project_parameters

class ControlOptimizer:
    def __init__(self, model, collider, matrices, progress=None):
        self.model = model
        self.collider = collider
        self.matrices = matrices
        self.trees = []
        self.progress = progress or (lambda *args: None)
        self.evaluations = 0
        # Objective defaults belong to injected cost components. The control
        # optimizer owns only iteration/history state, not fidelity settings.
        self.history = []
        self.hulls = []
        self.hull_bounds = []

    def optimize(self, iterations=220, initial=None, knot_spacing=8, free_frames=None):
        """C2 control-space L-BFGS, optionally freezing unrelated controls.

        A control is free when its positive basis support intersects any true
        ``free_frames`` entry.  Its complete bone/axis block is free together;
        the remaining controls retain the initial full-clip spline projection.
        The objective and temporal terms still cover the complete frame range.
        """
        self.progress('solve_prepare', 0, iterations)
        m = self.model
        parameters=getattr(m,'parameter_count',4)
        dimension=parameters*3
        # The unknowns are controls of a C2 cubic spline. Collision correction
        # cannot acquire independent per-frame spikes, even before smoothing.
        spacing=max(1,int(knot_spacing))
        basis=self.continuity.basis(m.n,spacing)
        spans=self.stationary.spans
        extra_knots=()
        if spans:
            refined,extra_knots=self.stationary.refine_basis(m.n,spacing)
            if refined is not None:basis=refined
        controls=basis.shape[1]
        fixed=tuple(getattr(m,'fixed_parameter_indices',()))
        fixed_variables=np.tile(np.isin(np.repeat(np.arange(parameters),3),fixed),controls)
        if initial is not None and fixed:initial=project_parameters(m,initial)
        x=np.zeros((controls,dimension)).ravel() if initial is None else np.linalg.lstsq(basis,initial.reshape(m.n,dimension),rcond=None)[0].ravel()
        x[fixed_variables]=0.
        spans=self.stationary.spans
        stationary=self.stationary.controls(basis,parameters)
        if stationary is not None:stationary.project(x)
        x[fixed_variables]=0.
        free_variables=None
        free_controls=np.ones(controls,dtype=bool)
        if free_frames is not None:
            free_frames=np.asarray(free_frames,dtype=bool)
            if free_frames.shape!=(m.n,):
                raise ValueError('free_frames must be a boolean mask of shape [frames]')
            free_controls=np.any(basis[free_frames]>0.,axis=0)
            free_variables=np.repeat(free_controls,dimension)
            if stationary is not None:
                stationary.expand_free(free_variables)
                free_controls=np.any(free_variables.reshape(controls,dimension),axis=1)
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
            if stationary is not None:stationary.project(projected)
            projected[fixed_variables]=0.
            if free_variables is not None:projected[~free_variables]=0.
            return value,projected
        self.progress('solve_initial', 0, iterations)
        value, grad = objective(x)
        history = []
        # Use the temporal quadratic as the initial inverse Hessian. This
        # lets a contact move a whole smooth interval instead of a single
        # frame, and avoids extremely slow convergence with stiff smoothing.
        # Staged/warm solves reuse exactly the same original decomposition.
        # Keep only one entry and rebuild whenever any defining input changes.
        self.progress('solve_temporal', 0, iterations)
        eigenvalues,eigenvectors,reduction,held = self.continuity.preconditioner(
            m.n,basis,spacing,extra_knots,stationary,spans)
        if held is not None:held_values,held_vectors=held
        accepted_steps = 0
        for iteration in range(iterations):
            self.progress('solve_iteration', iteration, iterations)
            q = grad.copy(); alpha = []
            for s,y,rho in reversed(history):
                a = rho*np.dot(s,q);alpha.append(a);q-=a*y
            scale = np.dot(history[-1][0],history[-1][1])/max(np.dot(history[-1][1],history[-1][1]),1e-20) if history else 1e-3
            if reduction is None:
                direction=(eigenvectors@((eigenvectors.T@q.reshape(controls,dimension))/(1./max(scale,1e-12)+eigenvalues[:,None]))).ravel()
            else:
                reduced_gradient=reduction.T@q.reshape(controls,dimension)
                direction=(reduction@(held_vectors@((held_vectors.T@reduced_gradient)/
                    (1./max(scale,1e-12)+held_values[:,None])))).ravel()
            for (s,y,rho),a in zip(history,reversed(alpha)):
                direction+=s*(a-rho*np.dot(y,direction))
            direction=-direction
            if stationary is not None:stationary.project(direction)
            direction[fixed_variables]=0.
            if free_variables is not None:direction[~free_variables]=0.
            slope=np.dot(grad,direction)
            if slope>=0:
                direction=-grad*1e-3; slope=np.dot(grad,direction);history=[]
            step=min(1.,.15/max(np.abs(direction).max(),1e-10))
            accepted=False
            for trial in range(16):
                candidate=x+step*direction
                if stationary is not None:stationary.project(candidate)
                if free_variables is not None:candidate[~free_variables]=x[~free_variables]
                candidate[fixed_variables]=0.
                self.progress('solve_trial', iteration, iterations, trial+1)
                nv,ng=objective(candidate)
                if np.isfinite(nv) and nv<=value+1e-4*step*slope:
                    accepted=True;break
                step*=.5
            if not accepted: break
            s=candidate-x;y=ng-grad;sy=np.dot(s,y)
            if sy>1e-10:
                history.append((s,y,1./sy));history=history[-16:]
            x,value,grad=candidate,nv,ng
            accepted_steps = iteration + 1
            if iteration%5==0:
                self.progress('solve',iteration,iterations,self.last_metrics)
                self.history.append(dict(iteration=iteration,**self.last_metrics))
                yield frames(x).copy()
            if np.max(np.abs(grad))<1e-5: break
        self.progress('solve_done', accepted_steps, iterations)
        self.result=frames(x)
        if stationary is not None:
            self.last_optimization_scope['stationary_control_groups']=len(stationary.groups)
            self.last_optimization_scope['stationary_max_correction_excursion']=max(
                float(np.max(np.abs(self.result[a:b+1,p]-self.result[a,p])))
                for p,a,b in spans)
        if locked_reference is not None and len(locked_reference):
            self.last_optimization_scope['locked_control_drift_max_abs']=float(np.max(np.abs(x[~free_variables]-locked_reference)))
        yield self.result
