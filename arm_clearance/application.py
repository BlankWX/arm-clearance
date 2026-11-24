"""Ordinary bake orchestration: fixed sequencing, no objective/gradient formulae.

The numerical engine, repair policy, scene transaction, Action writer and real
checker have separate owners. Modal yields retain the v0.2.12 positions.
"""
from itertools import count
import numpy as np
from .contracts import CandidatePose
from .composition import make_problem, partition_source
from .adapters.job_io import BlenderJobIO
from .adapters.deep_repair import recover, prepare_surface
from .adapters.half_repair import repair_halves
from .features.repair.half_search import isolated_half_frames
from .adapters.action_io import bake_action
from .adapters.surface import surface_check
from .features.collision.hand_geometry import build_hand_geometry
from .features.avoidance.metrics import full_metrics
from .features.repair.policy import RepairPolicy, SurfaceQuality
from .features.baking import CandidateBaker
from .features.safety.job import JobGuard
from .features.safety.reporting import JobProgress
from .features.safety.runner import logged_run


def run_job(scene, settings, status, *, console=None):
    return (yield from logged_run(_run_job, scene, settings, status, console=console))


def _run_job(scene, settings, status):
    io = BlenderJobIO(scene, settings)
    guard = JobGuard(io)
    solver = None
    best_action = None
    best_quality = None
    strategy = 0
    policy = RepairPolicy()
    report = guard.report
    try:
        guard.begin(status)
        yield None
        clip = yield from io.capture_source(status, report)
        clip = partition_source(clip)
        policy.time_partition=clip.time_partition
        frames = clip.frames
        report['source_time_partition']=clip.time_partition.report(frames)
        progress = JobProgress(status)
        hand = build_hand_geometry(clip.geometry, clip.meta, clip.matrices, progress=progress,time_partition=clip.time_partition)
        problem = make_problem(clip, hand, float(settings.margin), progress, surface_check)
        model, solver, collision, stationary = (
            problem.model, problem.solver, problem.collision, problem.stationary)
        progress.solver = solver
        holds = stationary.detect(clip.quaternions, clip.locations)
        report['stationary_source_holds'] = [
            {'parameter': p, 'start': int(frames[a]), 'end': int(frames[b])}
            for p, a, b in holds]
        if holds:
            status('静止段保护：%d 个通道共享连续修正姿态；仍检测身体与双手' % len(holds))
        solver.configure(float(settings.velocity_weight), float(settings.acceleration_weight),
                         float(getattr(settings, 'upper_pose_weight', 100.)),
                         float(getattr(settings, 'wrist_translation_limit', .010)) / clip.scale)
        report['parameters'] = {
            'margin_metres': float(settings.margin),
            'wrist_translation_limit_metres': float(getattr(settings, 'wrist_translation_limit', .010)),
            'pose_weights': solver.pose_weights.tolist(), 'velocity_weight': solver.velocity,
            'acceleration_weight': solver.acceleration, 'iterations': int(settings.iterations),
            'surface_step': .5, 'stagnation_strategies': policy.strategies,
            'upper_strong_soft_limit_degrees': '30 by default; up to 35 only in a second confirmed upper-arm/body repair time window',
            'surface_repair_frame_half_window': 24.,
            'refit_warm_iterations': 150, 'deep_body_proxy_reject_world_mm': guard.body_limit_mm}
        report['parameters']['source_fidelity'] = solver.fidelity_report()
        report['parameters']['source_motion'] = solver.motion_report()
        for _ in solver.prepare():
            yield None
        collision.restore_full_planes()
        status('计算原动作整段碰撞指标')
        report['proxy']['baseline'] = full_metrics(model, solver, np.zeros((len(frames), 8, 3)))
        report['proxy']['hand_geometry'] = hand.calibration
        report['proxy']['scope'] = ('身体与椭球/采样代理的残差；不等同于真实表面穿透深度；'
                                    '不因单独超过3毫米拒绝，真实网格复核是提交门禁；'
                                    '另拒绝超过20毫米的身体代理深残差，防止开放表面漏掉完全嵌入')
        report['proxy']['distance_units'] = 'armature coordinate units × 1000; multiply by object uniform scale for world mm'
        guard.write_report()
        for _ in solver.optimize_staged(int(settings.iterations)):
            yield None
        collision.restore_full_planes()
        delta = np.asarray(solver.result)
        if delta.shape != (len(frames), 8, 3) or not np.isfinite(delta).all():
            raise ValueError('优化结果存在无效参数，已保留先前动作')
        baker = CandidateBaker(bake_action, io.discard_action, io.arm, io.source, io.targets)
        for attempt in count():
            collision.restore_full_planes()
            status('计算修正后整段碰撞指标（复核轮次 %d）' % (attempt + 1))
            corrected = full_metrics(model, solver, delta)
            report['proxy']['corrected'] = corrected
            report['proxy']['body_field'] = collision.body_report()
            report['proxy']['history'] = solver.history
            status('写入六根骨骼旋转关键帧（复核轮次 %d）' % (attempt + 1))
            guard.write_report()
            yield None
            guard.candidate = baker.write(clip, CandidatePose(delta, model.quats(delta)))
            report['attempts'].clear()
            report['verification_count'] = attempt + 1
            attempt_report = guard.stage_candidate(corrected, attempt)
            yield None
            for surface in collision.verify(scene, io.arm, io.mesh, clip.geometry, clip.meta,
                                            clip.body, io.start, io.end,
                                            status=status, step=.5, own_hand_arm=True):
                report['surface_verification'] = surface
                attempt_report['surface_verification'] = surface
                yield None
            surface = report['surface_verification']
            safe = guard.check_candidate(surface, corrected, clip.scale, attempt_report)
            quality = SurfaceQuality.measure(surface)
            accepted = safe and (best_quality is None or quality.improves(best_quality))
            if accepted:
                baker.discard_replaced(best_action)
                best_action = guard.candidate
                best_quality = quality
                best_surface = surface
                best_delta = delta.copy()
                best_fields = solver.round_fields()
                best_hand = model.hand
                best_metrics = corrected
                strategy = 0
                if not quality.frames:
                    break
            elif best_action is None:
                first = guard.candidate
                work = first.copy()
                accepted_work = False
                try:
                    io.ad.action = work
                    io.ad.action_slot = io.slot_for(work, io.source_slot.identifier)
                    deep = frames[np.asarray(corrected['body_frame_mm'])*clip.scale > guard.body_limit_mm]
                    topology = prepare_surface(clip.geometry, clip.meta, clip.body)
                    repaired = yield from recover(io, clip, work, delta, surface, deep, topology, holds, status)
                    metrics = full_metrics(model, solver, repaired)
                    status('超限补修结束，重新复核整段动作')
                    for checked in collision.verify(scene, io.arm, io.mesh, clip.geometry, clip.meta,
                                                    clip.body, io.start, io.end, status=status,
                                                    step=.5, own_hand_arm=True):
                        yield None
                    passed_depth = guard.check_candidate(checked, metrics, clip.scale, {})
                    after = SurfaceQuality.measure(checked)
                    report['verification_count'] += 1
                    if ((after.improves(quality) or after == quality) and
                            metrics['body_max_mm'] <= corrected['body_max_mm']+1e-7):
                        report['surface_verification'] = checked
                        report['proxy']['corrected'] = metrics
                        guard.candidate = work
                        accepted_work = True
                        baker.discard_replaced(first)
                        if not passed_depth:
                            report['validation_failure'] = 'deep_body_proxy_residual'
                        elif after.frames:
                            report['validation_failure'] = 'actual_surface_intersection'
                    else:
                        report['validation_failure'] = 'deep_body_proxy_residual'
                finally:
                    if not accepted_work:
                        io.ad.action = first
                        io.ad.action_slot = io.slot_for(first, io.source_slot.identifier)
                        baker.discard_replaced(work)
                break
            else:
                rejected = guard.candidate
                guard.candidate = best_action
                baker.discard_replaced(rejected)
                report['surface_verification'] = best_surface
                report['proxy']['corrected'] = best_metrics
                strategy += 1
            status('复核第 %d 轮：最佳剩余 %d 个采样，连续 %d 次无改善' %
                   (attempt + 1, len(best_quality.frames), strategy))
            if strategy == len(policy.strategies):
                if isolated_half_frames(best_surface):
                    guard.candidate = best_action.copy()
                    changed = yield from repair_halves(io, clip, guard.candidate, best_surface, status)
                    if changed:
                        attempt_report = guard.stage_candidate(best_metrics, attempt+1)
                        report['verification_count'] += 1
                        for surface in collision.verify(scene, io.arm, io.mesh, clip.geometry, clip.meta,
                                                        clip.body, io.start, io.end, status=status,
                                                        step=.5, own_hand_arm=True):
                            yield None
                        if (guard.check_candidate(surface, best_metrics, clip.scale, attempt_report)
                                and SurfaceQuality.measure(surface).improves(best_quality)):
                            baker.discard_replaced(best_action)
                            best_action = guard.candidate
                            best_surface = surface
                            report['surface_verification'] = surface
                            if not surface['collision_samples']:
                                break
                    if guard.candidate != best_action:
                        baker.discard_replaced(guard.candidate)
                        guard.candidate = best_action
                    report['surface_verification'] = best_surface
                report['validation_failure'] = 'actual_surface_intersection'
                break
            io.prepare_retry_source(clip)
            delta = best_delta.copy()
            model.replace_hand_envelopes(best_hand)
            solver.set_round_fields(best_fields)
            solver.history.clear()
            progress.phase = '第 %d 轮补修：%s，剩余 %d 个采样，连续 %d 次无改善' % (
                attempt + 1, policy.strategies[strategy], len(best_quality.frames), strategy)
            status(progress.phase)
            yield None
            for target in policy.targets(best_surface, strategy):
                plan = policy.plan(target, model, clip.geometry, frames, attempt + 1,
                                   clip.scale, solver.round_fields(), strategy)
                solver.set_round_fields(plan.fields)
                if plan.failed_hand_categories:
                    hand, _ = collision.refit(delta, clip)
                    model.replace_hand_envelopes(hand)
                collision.update_repair_band()
                for _ in solver.optimize_staged(150, initial=delta, free_frames=plan.free_frames):
                    yield None
                delta = np.asarray(solver.result)
                if delta.shape != (len(frames), 8, 3) or not np.isfinite(delta).all():
                    raise ValueError('局部连续约束修复后的优化参数无效')
        return guard.finish()
    except GeneratorExit:
        guard.cancel()
        raise
    except BaseException as error:
        guard.error(error)
        raise
    finally:
        if solver is not None:
            solver.collision.restore_full_planes()
        if not guard.complete and best_action is not None:
            if guard.candidate is not None and guard.candidate != best_action:
                baker.discard_replaced(guard.candidate)
            guard.candidate = best_action
            report['surface_verification'] = best_surface
        guard.cleanup()
