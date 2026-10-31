"""Compatibility constructors; the application uses composition.make_problem."""
from .composition import solver_parts, partition_from_source
from .features.avoidance.model import LimbModel
from .features.avoidance.solver import SolveEngine
from .features.avoidance.metrics import full_metrics
from .features.collision.service import prepare_limb_geometry


class FullLimbModel(LimbModel):
    def __init__(self, matrices, quaternions, location_bases, geometry, meta, hand=None, margin=.001,time_partition=None):
        partition=time_partition or partition_from_source(matrices,quaternions,meta)
        shapes = prepare_limb_geometry(geometry, meta, matrices, hand,time_partition=partition)
        super().__init__(matrices, quaternions, location_bases, geometry, meta, shapes, margin)
        self.time_partition=partition


class FullLimbSolver(SolveEngine):
    def __init__(self, model, collider, matrices, progress=None):
        super().__init__(model, *solver_parts(model, collider, matrices, progress), matrices, progress)
