"""Unified solver. Concrete feature implementations are bound by composition."""
from dataclasses import replace
import numpy as np
from ...numerics import project_parameters
from ...contracts import CollisionPort, FidelityPort, ContinuityPort, StationaryPort
from .derivatives import analytical_fk_gradient
from .optimizer import ControlOptimizer


class SolveEngine(ControlOptimizer):
    def __init__(self, model, collision: CollisionPort, fidelity: FidelityPort,
                 continuity: ContinuityPort, stationary: StationaryPort, matrices, progress=None):
        self.collision=collision;self.fidelity=fidelity
        self.continuity=continuity;self.stationary=stationary
        super().__init__(model,collision.collider,matrices,progress)
        self.penalty=18000.

    @property
    def body(self):return self.collision.body

    @property
    def stationary_spans(self):return self.stationary.spans

    @stationary_spans.setter
    def stationary_spans(self,value):self.stationary.spans=value

    def prepare(self):
        yield from self.collision.prepare()

    def contacts(self,points,orientation,need_gradient=True):
        return self.collision.evaluate(points,orientation,need_gradient)

    def configure(self, velocity, acceleration, upper_pose_weight, wrist_translation_limit):
        self.velocity=velocity;self.acceleration=acceleration
        self.pose_weights[[0,3]]=upper_pose_weight
        self.wrist_translation_limit=wrist_translation_limit

    def round_fields(self):return self.collision.fields

    def set_round_fields(self, fields):self.collision.set_fields(fields)

    def fidelity_report(self):return self.fidelity.settings_report()

    def motion_report(self):return self.continuity.settings_report()

    def source_deviation_metrics(self, points, orientation, elbows, wrists):
        return self.fidelity.deviation_metrics(self.model,points,orientation,elbows,wrists)

    def source_motion_metrics(self, orientation, elbows, wrists):
        return self.continuity.deviation_metrics(self.model,orientation,elbows,wrists)

    def optimize_staged(self,iterations=600,initial=None,stage_iterations=150,free_frames=None):
        delta=np.zeros((self.model.n,8,3)) if initial is None else project_parameters(self.model,initial)
        stages=range(0,iterations,stage_iterations)
        self.optimization_stage_count=len(stages)
        for stage_index,start in enumerate(stages,1):
            self.optimization_stage_index=stage_index
            self.progress('solve_stage', 0, min(stage_iterations,iterations-start))
            self.collision.begin_stage(delta)
            for value in self.optimize(min(stage_iterations,iterations-start),initial=delta,free_frames=free_frames):
                delta=value
                yield value
            self.collision.end_stage()
        self.result=delta

    def objective(self,flat):
        m=self.model;x=project_parameters(m,np.asarray(flat).reshape(m.n,8,3))
        points,orientation,elbows,wrists=m.state(x)
        collision,force,torque,detail=self.collision.evaluate(points,orientation)
        factor=self.penalty/(m.length*m.length)
        energy=factor*collision;force*=factor;torque*=factor
        energy,force,ef,wf=self.fidelity.add_geometry(m,points,elbows,wrists,energy,force)
        energy,ef,wf=self.continuity.add_geometry(m,elbows,wrists,energy,ef,wf)
        energy,palm_gradient=self.fidelity.add_orientation(m,orientation,energy)
        energy,palm_gradient=self.continuity.add_orientation(m,orientation,energy,palm_gradient)
        grad=analytical_fk_gradient(m,x,points,orientation,elbows,wrists,force,torque,ef,wf,
                                    palm_matrix_gradient=palm_gradient)
        energy,grad=self.fidelity.add_pose(x,energy,grad)
        energy,grad=self.continuity.add_cost(x,energy,grad)
        energy,grad=self.fidelity.add_limits(m,x,energy,grad,self.collision.fields.body_weights)
        grad[:,m.fixed_parameter_indices]=0.
        self.evaluations+=1
        self.last_metrics={'energy':float(energy),'body_max_mm':float(detail['body'].max()*1000),
                           'mutual_max_mm':detail['mutual_max']*1000,'own_max_mm':detail['own_max']*1000}
        return float(energy),grad.ravel()

    @property
    def velocity(self):return self.continuity.velocity
    @velocity.setter
    def velocity(self,value):self.continuity.velocity=value

    @property
    def acceleration(self):return self.continuity.acceleration
    @acceleration.setter
    def acceleration(self,value):self.continuity.acceleration=value

    @property
    def pose_weights(self):return self.fidelity.pose_weights
    @pose_weights.setter
    def pose_weights(self,value):self.fidelity.pose_weights=value

    @property
    def elbow_weight(self):return self.fidelity.elbow_weight
    @elbow_weight.setter
    def elbow_weight(self,value):self.fidelity.elbow_weight=value

    @property
    def wrist_weight(self):return self.fidelity.wrist_weight
    @wrist_weight.setter
    def wrist_weight(self,value):self.fidelity.wrist_weight=value

    @property
    def position_weight(self):return self.fidelity.position_weight
    @position_weight.setter
    def position_weight(self,value):self.fidelity.position_weight=value

    @property
    def wrist_translation_limit(self):return self.fidelity.wrist_translation_limit
    @wrist_translation_limit.setter
    def wrist_translation_limit(self,value):self.fidelity.wrist_translation_limit=value

    @property
    def body_weights(self):return self.collision.fields.body_weights
    @body_weights.setter
    def body_weights(self,value):
        self.collision.set_fields(replace(self.collision.fields,body_weights=value))

    @property
    def mutual_weights(self):return self.collision.fields.mutual_weights
    @mutual_weights.setter
    def mutual_weights(self,value):
        self.collision.set_fields(replace(self.collision.fields,mutual_weights=value))

    @property
    def own_weights(self):return self.collision.fields.own_weights
    @own_weights.setter
    def own_weights(self,value):
        self.collision.set_fields(replace(self.collision.fields,own_weights=value))

    @property
    def body_margin(self):return self.collision.fields.body_margin
    @body_margin.setter
    def body_margin(self,value):
        self.collision.set_fields(replace(self.collision.fields,body_margin=value))

