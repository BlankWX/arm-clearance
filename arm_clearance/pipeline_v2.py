"""Six-bone capture, whole-clip solve, reversible bake and full-skin gate.

No frame/playback handlers are installed. Complete baked data is retained
independently of collision acceptance. Unaccepted or interrupted bakes are
explicitly linked as review actions; cancellation/failure restores the previous
active action and frame. Proxy residuals and actual surface checks are separate.
"""

import json
import time
import numpy as np
import bpy

from .pipeline import validate as legacy_validate, geometry_data, closed_collider
from .baking import bake_action, channels
from .solver_v2 import FullLimbModel, FullLimbSolver, full_metrics
from .hand_geometry import build_hand_geometry, refit_corrected_hand_envelopes
from .full_surface import surface_check
from .surface_repair import build_surface_repair_weights, merge_surface_repair_fields


_LINKS = ('ac_source_action', 'ac_result_action', 'ac_previous_result_action', 'ac_review_action')


def _restore_links(arm, links):
    for key, value in links.items():
        if value is None:
            if key in arm:
                del arm[key]
        else:
            arm[key] = value


def _slot_for(action, identifier):
    found = next((slot for slot in action.slots if slot.identifier == identifier), None)
    if found is not None:
        return found
    objects = [slot for slot in action.slots if slot.target_id_type == 'OBJECT']
    if len(objects) == 1:
        return objects[0]
    if len(action.slots) == 1:
        return action.slots[0]
    raise ValueError('无法唯一确定角色使用的动作槽')


def validate(settings):
    """Reuse existing scene checks and extend editable paths to both wrists."""
    arm, mesh, arm_targets, wrists = legacy_validate(settings)
    targets = [arm_targets[0], arm_targets[1], wrists[0],
               arm_targets[2], arm_targets[3], wrists[1]]
    if len(set(targets)) != 6:
        raise ValueError('六根手臂和手腕修正骨骼不能重复')
    paths = set()
    for name in targets:
        pb = arm.pose.bones[name]
        if pb.rotation_mode != 'QUATERNION':
            raise ValueError(name + ' 需要使用四元数旋转模式')
        if any(not constraint.mute and constraint.influence > 0 for constraint in pb.constraints):
            raise ValueError(name + ' 有活动约束，请先烘焙该骨骼的约束')
        paths.add(pb.path_from_id('rotation_quaternion'))
    for name in wrists:
        paths.add(arm.pose.bones[name].path_from_id('location'))
    for driver in arm.animation_data.drivers:
        if driver.data_path in paths:
            raise ValueError('修正骨骼存在旋转或位置驱动，请先烘焙驱动')
    for curve in channels(arm.animation_data.action, arm.animation_data.action_slot):
        if curve.data_path not in paths:
            continue
        if any(not modifier.mute for modifier in curve.modifiers):
            raise ValueError('修正骨骼的旋转或位置曲线有修改器，请先烘焙曲线修改器')
        if len(curve.sampled_points):
            raise ValueError('修正骨骼的曲线使用采样点，请先转换为关键帧')
    return arm, mesh, targets, wrists, arm_targets


def _write_report(text, report):
    text.clear()
    text.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


def _retain_review_action(arm, source, candidate, report, validation_status):
    """Keep a complete bake saveable without calling it a verified result."""
    source.use_fake_user = True
    candidate.use_fake_user = True
    candidate['arm_clearance_validation_status'] = validation_status
    surface = report.get('surface_verification') or {}
    candidate['arm_clearance_problem_frames'] = list(surface.get('problem_frames', []))
    arm['ac_source_action'] = source.name
    arm['ac_review_action'] = candidate.name
    report.update(result_preserved=True, candidate_discarded=False,
                  review_action=candidate.name, result_action=candidate.name,
                  validation_passed=False)


def _source_for(arm, active):
    source_name = active.get('arm_clearance_source', '')
    if source_name:
        source = bpy.data.actions.get(source_name)
        source = source or bpy.data.actions.get(arm.get('ac_source_action', ''))
        if source is None:
            raise ValueError('修正动作记录的原动作已不存在，请先选择有效的原动作')
        return source
    return active


def _replace_hand_envelopes(model, hand):
    """Freeze refitted shape axes without changing centres or articulation."""
    model.hand = hand
    radii = hand.per_frame_radii
    if radii is None:
        radii = np.broadcast_to(hand.radii, (model.n,) + hand.radii.shape)
    model.hand_radii = radii
    axes = hand.ellipsoid_axes
    if axes is None:
        axes = np.repeat(radii[..., None], 3, axis=-1)
    model.axes = np.concatenate((model.axes[:, :, :model.arm_count], axes), axis=2)
    model.radii = model.axes.max(-1)


def _failed_hand_categories(surface):
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


def _repair_free_frames(surface, frames, half_window=24.):
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
        free = np.min(np.abs(frames[:, None] - np.asarray(failures)[None]), axis=1) <= half_window
        if not np.any(free):
            fallback = 'failure_windows_do_not_overlap_the_captured_frame_range'
    else:
        free = np.zeros(len(frames), dtype=bool)
        fallback = 'no_finite_current_failure_frames'
    if fallback is not None:
        free[:] = True
    evidence = {'mode': 'whole_clip_fallback' if fallback else 'current_failure_time_windows',
                'half_window_frames': float(half_window), 'current_problem_frames': failures,
                'free_frame_mask_count': int(np.count_nonzero(free)),
                'locked_frame_mask_count': int(np.count_nonzero(~free)),
                'fallback_reason': fallback,
                'optimization_variables': 'Existing C2 time controls; boolean frame mask forwarded to solver'}
    return free, evidence


def run_job(scene, settings, status):
    """Generate a complete six-bone result using the scene's current range.

    ``status`` accepts a human-readable progress string; yield points match
    the existing modal operator contract. All wrist locations are absolute
    local values on bake: original_location + model.length * delta[:,6:8].
    ``location_bases`` follows capture_v2.py, including the parent/rest basis
    and excluding the current local wrist quaternion.
    """
    arm, mesh, targets, wrists, arm_targets = validate(settings)
    ad = arm.animation_data
    previous = ad.action
    previous_slot_id = ad.action_slot.identifier
    source = _source_for(arm, previous)
    source_slot = _slot_for(source, previous_slot_id)
    previous_links = {key: arm.get(key) for key in _LINKS}
    fake_users = {action: action.use_fake_user for action in (source, previous)}
    original_rotations = {name: tuple(arm.pose.bones[name].rotation_quaternion) for name in targets}
    original_locations = {name: tuple(arm.pose.bones[name].location) for name in wrists}
    original_frame, original_subframe = scene.frame_current, scene.frame_subframe
    start, end = int(scene.frame_start), int(scene.frame_end)
    if end - start < 2:
        raise ValueError('至少需要三帧才能约束连续性')
    started = time.perf_counter()
    complete = False
    candidate = None
    solver = None
    text = None
    history_records = []
    report = {'version': 2, 'status': 'running', 'range': [start, end],
              'source_action': source.name, 'previous_action': previous.name,
              'targets': targets, 'wrist_location_targets': wrists,
              'proxy': {}, 'surface_verification': None, 'attempts': [],
              'scope': ('整段六骨旋转与双腕位置修正，手指动画保持原样；'
                        '代理残差与真实变形网格验证分别报告；'
                        '烘焙数据独立保留，未通过复核的动作明确标记为待修正；'
                        '离散时刻验证不构成全部连续时刻无穿模的数学证明')}
    try:
        ad.action = source
        ad.action_slot = source_slot
        # Validate the original action as well as the previously active one.
        validate(settings)
        text = bpy.data.texts.new('ArmClearance_V2_Report_' + time.strftime('%Y%m%d_%H%M%S'))
        text.use_fake_user = True
        settings.report_name = text.name
        status('读取原动作，提取身体、双臂与完整手掌手指')
        _write_report(text, report)
        yield None
        geo, meta, body = geometry_data(arm, mesh, arm_targets, settings.body_materials)
        meta['arm_target_ids'] = list(meta['target_ids'])
        meta['targets'] = list(targets)
        meta['target_ids'] = [meta['bones'].index(name) for name in targets]
        meta['wrist_ids'] = [meta['bones'].index(name) for name in wrists]
        meta.update(armature=arm.name, mesh=mesh.name, range=[start, end], source_action=source.name)
        report['original_resolution'] = {'vertices': len(geo['vertices']),
                                         'triangles': len(geo['triangles']),
                                         'body_triangles': len(body)}
        status('封闭身体代理开口并建立连续碰撞缓存')
        yield None
        collider = closed_collider(geo, meta, body, settings.collider_ratio)
        frames = np.arange(start, end + 1, dtype=np.int32)
        matrices = np.empty((len(frames), len(meta['bones']), 4, 4), dtype=np.float32)
        quaternions = np.empty((len(frames), 6, 4), dtype=np.float32)
        locations = np.empty((len(frames), 2, 3), dtype=np.float32)
        location_bases = np.empty((len(frames), 2, 3, 3), dtype=np.float32)
        for index, frame in enumerate(frames):
            scene.frame_set(int(frame))
            matrices[index] = [arm.pose.bones[name].matrix for name in meta['bones']]
            quaternions[index] = [arm.pose.bones[name].rotation_quaternion[:] for name in targets]
            for side, name in enumerate(wrists):
                pb = arm.pose.bones[name]
                locations[index, side] = pb.location[:]
                location_bases[index, side] = (np.asarray(pb.matrix.to_3x3()) @
                    np.asarray(pb.rotation_quaternion.normalized().to_matrix()).T)
            if index % 10 == 0 or index == len(frames) - 1:
                status('采样原动作六骨与双腕位置 %d/%d' % (index + 1, len(frames)))
                yield None
        scale = float(np.mean(arm.matrix_world.to_scale()))
        report['armature_object_uniform_scale'] = scale
        status('拟合完整掌面和所有指节的固定碰撞分区')
        yield None
        optimization_phase = '主优化'
        def progress(stage, current=0, total=0, *info):
            if stage == 'solve':
                status('%s 第%d/%d组 %d/%d步' % (
                    optimization_phase, solver.optimization_stage_index,
                    solver.optimization_stage_count, int(current) + 1, int(total)))
                return
            label = {'hands': '采样完整手掌手指', 'colliders': '建立身体连续碰撞缓存',
                     'refit_hands': '重包修正后混合腕缝'}.get(stage, '处理进度')
            status('%s %d/%d' % (label, int(current), int(total)))
        hand = build_hand_geometry(geo, meta, matrices, progress=progress)
        model = FullLimbModel(matrices, quaternions, location_bases, geo, meta,
                              hand=hand, margin=float(settings.margin) / scale)
        solver = FullLimbSolver(model, collider, matrices, progress)
        solver.velocity = float(settings.velocity_weight)
        solver.acceleration = float(settings.acceleration_weight)
        solver.pose_weights[[0, 3]] = float(getattr(settings, 'upper_pose_weight', 100.))
        solver.wrist_translation_limit = float(getattr(settings, 'wrist_translation_limit', .010)) / scale
        report['parameters'] = {'margin_metres': float(settings.margin),
            'wrist_translation_limit_metres': float(getattr(settings, 'wrist_translation_limit', .010)),
            'pose_weights': solver.pose_weights.tolist(), 'velocity_weight': solver.velocity,
            'acceleration_weight': solver.acceleration, 'iterations': int(settings.iterations),
            'surface_step': .5, 'maximum_surface_repair_retries': 3,
            'upper_strong_soft_limit_degrees': '30 by default; up to 35 only in a second confirmed upper-arm/body repair time window',
            'maximum_hand_refit_retries': 3, 'maximum_surface_verification_attempts': 4,
            'surface_repair_frame_half_window': 24.,
            'refit_warm_iterations': 150, 'deep_body_proxy_reject_world_mm': 20.}
        for _ in solver.prepare():
            yield None
        # Use the complete body planes when reporting either proxy endpoint.
        solver.body.restore_full_planes()
        report['proxy']['baseline'] = full_metrics(model, solver, np.zeros((len(frames), 8, 3)))
        report['proxy']['hand_geometry'] = hand.calibration
        report['proxy']['scope'] = ('身体与椭球/采样代理的残差；不等同于真实表面穿透深度；'
                                    '不因单独超过3毫米拒绝，真实网格复核是提交门禁；'
                                    '另拒绝超过20毫米的身体代理深残差，防止开放表面漏掉完全嵌入')
        report['proxy']['distance_units'] = 'armature coordinate units × 1000; multiply by object uniform scale for world mm'
        _write_report(text, report)
        for _ in solver.optimize_staged(int(settings.iterations)):
            yield None
        solver.body.restore_full_planes()
        delta = np.asarray(solver.result)
        if delta.shape != (len(frames), 8, 3) or not np.isfinite(delta).all():
            raise ValueError('优化结果存在无效参数，已保留先前动作')
        # Actual failures reinforce only their category/side/local samples
        # through fixed C2 temporal fields. A hand envelope is additionally
        # refitted only if a hand category failed. At most three warm passes.
        for attempt in range(4):
            solver.body.restore_full_planes()
            corrected = full_metrics(model, solver, delta)
            report['proxy']['corrected'] = corrected
            report['proxy']['body_field'] = solver.body.report
            report['proxy']['history'] = solver.history
            status('写入六根骨骼旋转和双腕位置关键帧（复核轮次 %d/4）' % (attempt + 1))
            _write_report(text, report)
            yield None
            previous_candidate = candidate
            candidate = bake_action(arm, source, targets, frames, model.quats(delta),
                'AC_v2_' + source.name, locations=locations + model.length*delta[:, 6:], wrist_names=wrists)
            # Replace a checkpoint only after its successor is fully written.
            if previous_candidate is not None:
                bpy.data.actions.remove(previous_candidate)
            # Publish comparison links only after verification has a terminal outcome.
            _restore_links(arm, previous_links)
            candidate['arm_clearance_report'] = text.name
            candidate['arm_clearance_validation_status'] = 'unverified'
            report['baked_candidate_name'] = candidate.name
            report['surface_verification'] = None
            attempt_report = {'attempt': attempt + 1, 'proxy': corrected,
                              'candidate_name': candidate.name, 'surface_verification': None}
            report['attempts'].append(attempt_report)
            yield None
            for surface in surface_check(scene, arm, mesh, geo, meta, body, start, end,
                                          status=status, step=.5, own_hand_arm=True):
                report['surface_verification'] = surface
                attempt_report['surface_verification'] = surface
                yield None
            surface = report['surface_verification']
            if surface is None or not surface['complete']:
                raise RuntimeError('真实网格复核未完成')
            body_depth_world_mm = float(corrected['body_max_mm']) * scale
            attempt_report['body_proxy_max_world_mm'] = body_depth_world_mm
            if body_depth_world_mm > 20.:
                report['validation_failure'] = 'deep_body_proxy_residual'
                break
            if not surface['collision_samples']:
                break
            # Re-apply stronger retry constraints to every region with a real
            # failure in this job. A formerly clear window must not relax
            # again while another window is repaired by the whole-clip solve.
            history_records.extend(surface.get('records', []))
            attempt_report['historical_surface_failure_record_count'] = len(history_records)
            if attempt == 3:
                report['validation_failure'] = 'actual_surface_intersection'
                break
            ad.action = source
            ad.action_slot = source_slot
            # Reestablish original unanimated defaults as well as animated
            # channels before making the next candidate's static guards.
            for j, name in enumerate(targets):
                arm.pose.bones[name].rotation_quaternion = quaternions[-1, j]
            for side, name in enumerate(wrists):
                arm.pose.bones[name].location = locations[-1, side]
            scene.frame_set(end)
            status('将真实失败三角面映射为局部连续碰撞约束（%d/3）' % (attempt + 1))
            yield None
            repair_surface = dict(surface)
            repair_surface['records'] = list(history_records)
            repair, repair_evidence = build_surface_repair_weights(
                repair_surface, model, geo, frames=frames, attempt=attempt + 1, world_scale=scale)
            current = {name: getattr(solver, name) for name in
                       ('body_weights', 'mutual_weights', 'own_weights', 'body_margin')}
            repair = merge_surface_repair_fields(current, repair, world_scale=scale)
            for name, value in repair.items():
                setattr(solver, name, value)
            attempt_report['local_surface_repair'] = repair_evidence
            failed_hand = _failed_hand_categories(surface)
            attempt_report['failed_hand_categories'] = failed_hand
            if failed_hand:
                status('仅对实际手部失败平滑重包修正后的手部簇（%d/3）' % (attempt + 1))
                yield None
                # Start from saved original envelope baselines; do not
                # multiply already expanded shapes on successive attempts.
                hand, refit_report = refit_corrected_hand_envelopes(
                    model.hand, model, delta, geo, meta=meta, matrices=matrices, progress=progress)
                attempt_report['corrected_hand_refit'] = refit_report
                _replace_hand_envelopes(model, hand)
            solver.body.narrow_band = float(np.max(model.radii + model.margin + solver.body_margin) + .006)
            attempt_report['repair_narrow_band_armature_units'] = solver.body.narrow_band
            free_frames, free_evidence = _repair_free_frames(surface, frames, half_window=24.)
            attempt_report['repair_variable_frame_mask'] = free_evidence
            _write_report(text, report)
            optimization_phase = '追加修复 第%d/3轮' % (attempt + 1)
            for _ in solver.optimize_staged(150, initial=delta, free_frames=free_frames):
                yield None
            if hasattr(solver, 'last_optimization_scope'):
                attempt_report['repair_c2_control_scope'] = dict(solver.last_optimization_scope)
            delta = np.asarray(solver.result)
            if delta.shape != (len(frames), 8, 3) or not np.isfinite(delta).all():
                raise ValueError('局部连续约束修复后的优化参数无效')
        source.use_fake_user = True
        previous.use_fake_user = True
        candidate.use_fake_user = True
        # Bind the copied slot explicitly, including multi-slot source actions.
        ad.action = candidate
        ad.action_slot = _slot_for(candidate, source_slot.identifier)
        prior_result = bpy.data.actions.get(previous_links.get('ac_review_action') or '')
        prior_result = prior_result or bpy.data.actions.get(previous_links.get('ac_result_action') or '')
        if prior_result is None and previous != source:
            prior_result = previous
        if prior_result is not None and prior_result != candidate:
            prior_result.use_fake_user = True
            arm['ac_previous_result_action'] = prior_result.name
        arm['ac_source_action'] = source.name
        validation_passed = 'validation_failure' not in report
        report.update(result_action=candidate.name, result_preserved=True,
                      candidate_discarded=False, validation_passed=validation_passed)
        if validation_passed:
            arm['ac_result_action'] = candidate.name
            if 'ac_review_action' in arm:
                del arm['ac_review_action']
            candidate['arm_clearance_validation_status'] = 'passed'
            report['status'] = 'complete'
            settings.status = '完成 %d–%d 帧，复核通过，烘焙动作已保留；保存工程即可保存数据' % (start, end)
        else:
            _retain_review_action(arm, source, candidate, report, 'needs_review')
            report['status'] = 'needs_review'
            reason = ('身体代理仍有深残差' if report['validation_failure'] == 'deep_body_proxy_residual'
                      else '仍有穿模，问题帧见报告')
            settings.status = '已保留 %d–%d 帧烘焙，复核未通过：%s；可保存和切换查看' % (start, end, reason)
        complete = True
        return candidate
    except GeneratorExit:
        report['status'] = 'cancelled'
        settings.status = '已取消，恢复先前动作'
        raise
    except BaseException as error:
        report['status'] = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'error'
        report['error'] = str(error)
        settings.status = '处理失败，已恢复先前动作：' + str(error)
        raise
    finally:
        if solver is not None:
            solver.body.restore_full_planes()
        if not complete:
            ad.action = previous
            ad.action_slot = _slot_for(previous, previous_slot_id)
            _restore_links(arm, previous_links)
            for action, was_fake in fake_users.items():
                action.use_fake_user = was_fake
            for name, value in original_rotations.items():
                arm.pose.bones[name].rotation_quaternion = value
            for name, value in original_locations.items():
                arm.pose.bones[name].location = value
            if candidate is not None:
                _retain_review_action(arm, source, candidate, report, 'unverified')
                settings.status += '；最近完整烘焙已保留，可切换查看'
        scene.frame_set(original_frame, subframe=original_subframe)
        report['seconds'] = time.perf_counter() - started
        if text is not None:
            _write_report(text, report)
