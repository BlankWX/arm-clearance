"""Bind concrete feature implementations outside the numerical engine."""
from dataclasses import replace
import numpy as np
from .contracts import SolveComponents
from .source_time import detect_source_partition
from .features.avoidance.model import LimbModel
from .features.avoidance.solver import SolveEngine
from .features.collision.service import CollisionSystem, prepare_limb_geometry
from .features.fidelity import FidelityTerms
from .features.continuity import ContinuityTerms
from .features.stationary import StationaryPolicy


def partition_from_source(matrices, quaternions, meta):
    positions=np.take(np.asarray(matrices),meta['target_ids'],axis=1)[:,:,:3,3]
    return detect_source_partition(quaternions,positions)


def partition_source(clip):
    return clip if clip.time_partition is not None else replace(
        clip,time_partition=partition_from_source(clip.matrices,clip.quaternions,clip.meta))


def make_model(matrices, quaternions, location_bases, geometry, meta, hand=None, margin=.001,time_partition=None):
    partition=time_partition or partition_from_source(matrices,quaternions,meta)
    shapes = prepare_limb_geometry(geometry, meta, matrices, hand,time_partition=partition)
    model=LimbModel(matrices, quaternions, location_bases, geometry, meta, shapes, margin)
    model.time_partition=partition
    return model


def solver_parts(model, collider, matrices, progress=None, surface_checker=None):
    partition=getattr(model,'time_partition',None) or partition_from_source(matrices,model.quaternions,model.meta)
    model.time_partition=partition
    fidelity=FidelityTerms();continuity=ContinuityTerms();stationary=StationaryPolicy()
    for component in (fidelity,continuity,stationary):component.time_partition=partition
    return (CollisionSystem(model, collider, matrices, progress, surface_checker),fidelity,continuity,stationary)


def make_problem(clip, hand, margin, progress, surface_checker):
    clip=partition_source(clip)
    model = make_model(clip.matrices, clip.quaternions, clip.location_bases,
                       clip.geometry, clip.meta, hand=hand, margin=margin / clip.scale,time_partition=clip.time_partition)
    collision, fidelity, continuity, stationary = solver_parts(
        model, clip.collider, clip.matrices, progress, surface_checker)
    solver = SolveEngine(model, collision, fidelity, continuity, stationary, clip.matrices, progress)
    return SolveComponents(model, solver, collision, stationary)
