"""Area 6: failure history and declarative retry plans, without solver mutation."""
from dataclasses import replace
import numpy as np
from ...contracts import CollisionFields, RepairPlan
from .fields import build_surface_repair_weights, merge_surface_repair_fields


class RepairPolicy:
    def __init__(self):
        self.history_records = []
        self.time_partition = None

    def record_failure(self, surface):
        self.history_records.extend(surface.get('records', []))
        return len(self.history_records)

    def plan(self, surface, model, geometry, frames, attempt, world_scale, current):
        repair_surface = dict(surface)
        repair_surface['records'] = list(self.history_records)
        fields, evidence = build_surface_repair_weights(
            repair_surface, model, geometry, frames=frames, attempt=attempt, world_scale=world_scale,time_partition=self.time_partition)
        fields = merge_surface_repair_fields(current.as_mapping(), fields, world_scale=world_scale)
        return RepairPlan(CollisionFields.from_mapping(fields), evidence,
                          tuple(failed_hand_categories(surface)))

    def scope(self, plan, surface, frames):
        free, evidence = repair_free_frames(surface, frames, half_window=24.,time_partition=self.time_partition)
        return replace(plan, free_frames=free, scope_evidence=evidence)


def failed_hand_categories(surface):
    """Identify actual hand-related failures without widening arm-only repairs."""
    names = ('hand-body', 'hand-opposite-arm', 'hand-hand', 'hand-own-arm')
    failed = set()
    for name in names:
        summary = surface.get('categories', {}).get(name, {})
        if any(summary.get(key, 0) for key in
               ('intersection_samples', 'intersection_pair_count', 'containment_samples', 'proxy_samples')):
            failed.add(name)
    for record in surface.get('records', []):
        for name in names:
            detail = record.get('categories', {}).get(name, {})
            if detail.get('pair_count') or detail.get('pairs') or detail.get('containments') or detail.get('proxies'):
                failed.add(name)
    return sorted(failed)

def repair_free_frames(surface, frames, half_window=24.,time_partition=None):
    """Request C2-control freedom around current failures, with a safe fallback."""
    frames = np.asarray(frames, dtype=float)
    failures = []
    candidates = list(surface.get('problem_frames') or [])
    candidates.extend(record.get('frame') for record in surface.get('records', []))
    for value in candidates:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            failures.append(value)
    failures = sorted(set(failures))
    fallback = None
    if failures:
        if time_partition is not None and time_partition.cuts:
            free=np.zeros(len(frames),dtype=bool)
            for frame in failures:free|=time_partition.repair_support(frames,frame,half_window)
        else:
            free = np.min(np.abs(frames[:, None] - np.asarray(failures)[None]), axis=1) <= half_window
        if not np.any(free):
            fallback = 'failure_windows_do_not_overlap_the_captured_frame_range'
    else:
        free = np.zeros(len(frames), dtype=bool)
        fallback = 'no_finite_current_failure_frames'
    if fallback is not None:
        if time_partition is None or not time_partition.cuts:free[:] = True
    evidence = {'mode': 'whole_clip_fallback' if fallback else 'current_failure_time_windows',
                'half_window_frames': float(half_window), 'current_problem_frames': failures,
                'free_frame_mask_count': int(np.count_nonzero(free)),
                'locked_frame_mask_count': int(np.count_nonzero(~free)),
                'fallback_reason': fallback,
                'optimization_variables': 'Existing C2 time controls; boolean frame mask forwarded to solver'}
    if time_partition is not None and time_partition.cuts:
        evidence['source_partition']=time_partition.report(frames)
        if fallback:evidence['mode']='unmapped_failure_preserve_partitioned_candidate'
    return free, evidence
