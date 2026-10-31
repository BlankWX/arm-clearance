"""In-process data contracts; array owners lend read-only references.

No Blender objects, serialization or per-iteration clip copies live here.
Arrays retain the baseline dtype, shape and bone/frame ordering.
"""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from .source_time import SourcePartition


@dataclass(frozen=True)
class SourceClip:
    frames: np.ndarray
    matrices: np.ndarray
    quaternions: np.ndarray
    locations: np.ndarray
    location_bases: np.ndarray
    geometry: dict
    meta: dict
    body: np.ndarray
    collider: dict
    scale: float
    time_partition: SourcePartition = None


@dataclass(frozen=True)
class LimbGeometry:
    arm_geometry: dict
    hand: object


@dataclass(frozen=True)
class CollisionFields:
    body_weights: np.ndarray
    mutual_weights: np.ndarray
    own_weights: np.ndarray
    body_margin: np.ndarray

    @classmethod
    def baseline(cls, frames, samples):
        shape = (frames, 2, samples)
        return cls(np.ones(shape), np.ones(shape), np.ones(shape), np.zeros(shape))

    @classmethod
    def from_mapping(cls, values):
        return cls(**values)

    def as_mapping(self):
        return {name: getattr(self, name) for name in
                ('body_weights', 'mutual_weights', 'own_weights', 'body_margin')}


@dataclass(frozen=True)
class CandidatePose:
    delta: np.ndarray
    quaternions: np.ndarray


@dataclass(frozen=True)
class RepairPlan:
    fields: CollisionFields
    evidence: dict
    failed_hand_categories: tuple
    free_frames: object = None
    scope_evidence: object = None


@dataclass(frozen=True)
class SolveComponents:
    model: object
    solver: object
    collision: object
    stationary: object


class CollisionPort(Protocol):
    """Geometric contribution, with fields fixed for the current solve."""
    collider: dict
    fields: CollisionFields

    def evaluate(self, points, orientation, need_gradient=True): ...
    def prepare(self): ...
    def begin_stage(self, delta): ...
    def end_stage(self): ...
    def set_fields(self, fields: CollisionFields): ...


class FidelityPort(Protocol):
    """Separate accumulation stages preserve the original arithmetic order."""
    pose_weights: np.ndarray
    elbow_weight: float
    wrist_weight: float
    position_weight: float
    wrist_translation_limit: float
    time_partition: SourcePartition

    def add_geometry(self, m, points, elbows, wrists, energy, force): ...
    def add_pose(self, x, energy, grad): ...
    def add_orientation(self, m, orientation, energy): ...
    def add_limits(self, m, x, energy, grad, body_weights): ...
    def settings_report(self): ...
    def deviation_metrics(self, m, points, orientation, elbows, wrists): ...


class ContinuityPort(Protocol):
    velocity: float
    acceleration: float
    time_partition: SourcePartition

    def basis(self, frames, spacing): ...
    def add_cost(self, x, energy, grad): ...
    def add_geometry(self, m, elbows, wrists, energy, elbow_gradient, wrist_gradient): ...
    def add_orientation(self, m, orientation, energy, palm_gradient): ...
    def settings_report(self): ...
    def deviation_metrics(self, m, orientation, elbows, wrists): ...
    def preconditioner(self, frames, basis, spacing, extra_knots, stationary, spans): ...


class StationaryPort(Protocol):
    spans: object
    time_partition: SourcePartition

    def refine_basis(self, frames, spacing): ...
    def controls(self, basis, parameters): ...
