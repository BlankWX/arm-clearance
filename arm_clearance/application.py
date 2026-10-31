"""Ordinary bake orchestration: fixed sequencing, no objective/gradient formulae.

The numerical engine, repair policy, scene transaction, Action writer and real
checker have separate owners. Modal yields retain the v0.2.12 positions.
"""
import numpy as np
from .contracts import CandidatePose
from .composition import make_problem, partition_source
from .adapters.job_io import BlenderJobIO
from .adapters.action_io import bake_action
from .adapters.surface import surface_check
from .features.collision.hand_geometry import build_hand_geometry
from .features.avoidance.metrics import full_metrics
from .features.repair.policy import RepairPolicy
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
            'surface_step': .5, 'maximum_surface_repair_retries': 3,
            'upper_strong_soft_limit_degrees': '30 by default; up to 35 only in a second confirmed upper-arm/body repair time window',
            'maximum_hand_refit_retries': 3, 'maximum_surface_verification_attempts': 4,
            'surface_repair_frame_half_window': 24.,
            'refit_warm_iterations': 150, 'deep_body_proxy_reject_world_mm': 20.}
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
        for attempt in range(4):
            collision.restore_full_planes()
            status('计算修正后整段碰撞指标（复核轮次 %d/4）' % (attempt + 1))
            corrected = full_metrics(model, solver, delta)
            report['proxy']['corrected'] = corrected
            report['proxy']['body_field'] = collision.body_report()
            report['proxy']['history'] = solver.history
            status('写入六根骨骼旋转和双腕位置关键帧（复核轮次 %d/4）' % (attempt + 1))
            guard.write_report()
            yield None
            previous_candidate = guard.candidate
            guard.candidate = baker.write(clip, CandidatePose(delta, model.quats(delta)))
            baker.discard_replaced(previous_candidate)
            attempt_report = guard.stage_candidate(corrected, attempt)
            yield None
            for surface in collision.verify(scene, io.arm, io.mesh, clip.geometry, clip.meta,
                                            clip.body, io.start, io.end,
                                            status=status, step=.5, own_hand_arm=True):
                report['surface_verification'] = surface
                attempt_report['surface_verification'] = surface
                yield None
            surface = report['surface_verification']
            if guard.verification_terminal(surface, corrected, clip.scale, attempt_report):
                break
            attempt_report['historical_surface_failure_record_count'] = policy.record_failure(surface)
            if guard.retries_exhausted(attempt):
                break
            io.prepare_retry_source(clip)
            status('将真实失败三角面映射为局部连续碰撞约束（%d/3）' % (attempt + 1))
            yield None
            plan = policy.plan(surface, model, clip.geometry, frames, attempt + 1,
                               clip.scale, solver.round_fields())
            solver.set_round_fields(plan.fields)
            attempt_report['local_surface_repair'] = plan.evidence
            attempt_report['failed_hand_categories'] = list(plan.failed_hand_categories)
            if plan.failed_hand_categories:
                status('仅对实际手部失败平滑重包修正后的手部簇（%d/3）' % (attempt + 1))
                yield None
                hand, refit_report = collision.refit(delta, clip)
                attempt_report['corrected_hand_refit'] = refit_report
                model.replace_hand_envelopes(hand)
            attempt_report['repair_narrow_band_armature_units'] = collision.update_repair_band()
            plan = policy.scope(plan, surface, frames)
            attempt_report['repair_variable_frame_mask'] = plan.scope_evidence
            guard.write_report()
            progress.phase = '追加修复 第%d/3轮' % (attempt + 1)
            for _ in solver.optimize_staged(150, initial=delta, free_frames=plan.free_frames):
                yield None
            if hasattr(solver, 'last_optimization_scope'):
                attempt_report['repair_c2_control_scope'] = dict(solver.last_optimization_scope)
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
        guard.cleanup()
