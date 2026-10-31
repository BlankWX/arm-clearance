"""Reversible six-bone baking, with exact source-curve guard preservation.

Only the selected six quaternion paths
are edited in a copy.  The source action and all other channels (including
finger keys, handle types and modifiers) are left untouched.  The original
curves are preserved before ``start - 1`` and after ``end + 1``; those adjacent
one-frame intervals are the explicit transition into/out of the corrected
range.  Different endpoint values cannot also preserve the original function
at every fractional time immediately outside the range while staying
continuous.
"""

import numpy as np

_DEFAULT_END_GUARD = object()


def channels(action, slot):
    """Yield only this action slot's F-curves, without creating channelbags."""
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type == 'KEYFRAME':
                bag = strip.channelbag(slot)
                if bag is not None:
                    yield from bag.fcurves


def _bezier_points(left, right):
    points = np.array([left.co[:], left.handle_right[:],
                       right.handle_left[:], right.co[:]], dtype=np.float64)
    # Blender shortens overly long horizontal Bezier handles before solving
    # x(t).  Split this same effective curve, including hand-edited FREE
    # handles that extend past one another.
    width = float(points[3, 0] - points[0, 0])
    # BKE_fcurve_correct_bezpart limits each handle independently; it does
    # not shorten both in proportion to their combined horizontal length.
    for handle, anchor in ((1, 0), (2, 3)):
        length = abs(float(points[handle, 0] - points[anchor, 0]))
        if length > width and length > 0:
            points[handle] = points[anchor] + (width / length) * (points[handle] - points[anchor])
    return points


def _split_points(points, t):
    s = 1. - t
    a, b, c = s*points[:-1] + t*points[1:]
    d, e = s*np.array([a, b]) + t*np.array([b, c])
    point = s*d + t*e
    return np.array([points[0], a, d, point]), np.array([point, e, c, points[3]])


def _safe_segments(points, depth=0):
    """Subdivide a rare flat-time segment before Blender reclamps its handles.

    De Casteljau can produce an outward subcurve whose control handle extends
    beyond its new endpoint although the full original curve is monotone.
    Splitting at the minimum of dx/dt makes both outward pieces representable
    by Blender's individually limited F-curve handles without changing shape.
    """
    x = points[:, 0]
    epsilon = max(abs(float(x[3] - x[0])) * 1e-10, 1e-12)
    if x[1] <= x[3] + epsilon and x[2] >= x[0] - epsilon:
        return [points]
    if depth > 12:
        raise ValueError('Unable to preserve an extreme source Bezier guard')
    quadratic = -x[0] + 3*x[1] - 3*x[2] + x[3]
    linear = 2*(x[0] - 2*x[1] + x[2])
    t = -linear / (2*quadratic) if abs(quadratic) > 1e-15 else .5
    if not 1e-8 < t < 1 - 1e-8:
        t = .5
    left, right = _split_points(points, float(t))
    return _safe_segments(left, depth + 1) + _safe_segments(right, depth + 1)


def curve_boundary(fc, time, default_value=0.):
    """Describe a guard split without changing the source F-curve.

    Empty curves use the actual original static property supplied by the
    caller; Blender's zero result for an empty curve is not a static default.
    For Bezier segments, De Casteljau subdivision preserves the complete
    original curve on the outward side of the inserted guard.
    """
    time = float(time)
    keys = list(fc.keyframe_points)
    if not keys:
        return {'time': time, 'value': float(default_value), 'interpolation': 'LINEAR',
                'exists': False, 'outside_keys': True, 'outward_slope': 0.}
    matching = next((k for k in keys if abs(float(k.co.x) - time) < 1e-6), None)
    result = {'time': time, 'value': float(fc.evaluate(time)),
              'interpolation': 'LINEAR', 'exists': matching is not None}
    if matching is not None:
        result.update(value=float(matching.co.y), interpolation=matching.interpolation,
                      left_handle=list(matching.handle_left), right_handle=list(matching.handle_right))
        # Existing first/last guard keys may still need their original
        # extrapolation tangent after the inward segment becomes linear.
        if matching == keys[0] or matching == keys[-1]:
            result['outside_keys'] = True
            result['outward_slope'] = _extrapolation_slope(fc, time,
                                                         before=matching == keys[0])
        return result
    left = next((k for k in reversed(keys) if k.co.x < time), None)
    right = next((k for k in keys if k.co.x > time), None)
    if left is None or right is None:
        result['outside_keys'] = True
        result['outward_slope'] = _extrapolation_slope(fc, time, before=left is None)
        return result
    result['interpolation'] = left.interpolation
    if left.interpolation == 'BEZIER':
        points = _bezier_points(left, right)
        low, high = 0., 1.
        for _ in range(60):
            t = (low + high) / 2
            s = 1 - t
            x = s**3*points[0, 0] + 3*s*s*t*points[1, 0] + 3*s*t*t*points[2, 0] + t**3*points[3, 0]
            if x < time:
                low = t
            else:
                high = t
        t = (low + high) / 2
        s = 1 - t
        a, b, c = s*points[:-1] + t*points[1:]
        d, e = s*np.array([a, b]) + t*np.array([b, c])
        value = s*d + t*e
        result.update(value=float(value[1]), left_handle=d.tolist(), right_handle=e.tolist(),
                      previous=float(left.co.x), next=float(right.co.x),
                      previous_right=a.tolist(), next_left=c.tolist(),
                      left_segments=_safe_segments(np.array([points[0], a, d, value])),
                      right_segments=_safe_segments(np.array([value, e, c, points[3]])))
    elif left.interpolation not in ('LINEAR', 'CONSTANT'):
        raise ValueError('Bake guards support Bezier, linear and constant source interpolation')
    return result


def _extrapolation_slope(fc, time, before):
    if fc.extrapolation != 'LINEAR' or len(fc.keyframe_points) < 2:
        return 0.
    # Both evaluations stay on the same original linear extrapolation tail.
    if before:
        return float(fc.evaluate(time) - fc.evaluate(time - 1.))
    return float(fc.evaluate(time + 1.) - fc.evaluate(time))


def _select_source_slot(ad, source):
    current = ad.action_slot
    if current is not None:
        found = next((slot for slot in source.slots if slot.identifier == current.identifier), None)
        if found is not None:
            return found
    object_slots = [slot for slot in source.slots if slot.target_id_type == 'OBJECT']
    if len(object_slots) == 1:
        return object_slots[0]
    if len(source.slots) == 1:
        return source.slots[0]
    raise ValueError('Cannot identify the armature source action slot unambiguously')


def _channelbag(action, slot):
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type == 'KEYFRAME':
                bag = strip.channelbag(slot)
                if bag is not None:
                    return bag
    layer = action.layers[0] if action.layers else action.layers.new('Arm Clearance')
    strip = next((strip for strip in layer.strips if strip.type == 'KEYFRAME'), None)
    if strip is None:
        strip = layer.strips.new(type='KEYFRAME')
    return strip.channelbag(slot, ensure=True)


def _validate_curve(fc):
    if any(not modifier.mute for modifier in fc.modifiers):
        raise ValueError('Bake target curve has active F-curve modifiers: ' + fc.data_path)
    if len(fc.sampled_points):
        raise ValueError('Bake target curve must use keyframes rather than sampled points: ' + fc.data_path)


def _key_snapshot(key):
    return {'time': float(key.co.x), 'co': tuple(key.co),
            'left': tuple(key.handle_left), 'right': tuple(key.handle_right),
            'interpolation': key.interpolation}


def _write_curve(fc, frames, values, default_value, *, end_guard_time=_DEFAULT_END_GUARD):
    _validate_curve(fc)
    start, end = float(np.float32(frames[0])), float(np.float32(frames[-1]))
    # Capture both guards from the untouched curve, even when both fall in
    # one long original Bezier segment. Keep side identity if no tail exists.
    guards = [(0, curve_boundary(fc, start - 1., default_value))]
    if end_guard_time is _DEFAULT_END_GUARD:
        end_guard_time = end + 1.
    if end_guard_time is not None:
        end_guard_time = float(np.float32(end_guard_time))
        if not np.isfinite(end_guard_time) or end_guard_time <= end:
            raise ValueError('Trailing source guard must follow the written bake end')
        guards.append((1, curve_boundary(fc, end_guard_time, default_value)))
    outside = [_key_snapshot(key) for key in fc.keyframe_points
               if not start <= key.co.x <= end]
    for key in list(fc.keyframe_points)[::-1]:
        if start <= key.co.x <= end:
            fc.keyframe_points.remove(key, fast=True)
    for frame, value in zip(frames, values):
        key = fc.keyframe_points.insert(float(frame), float(value), options={'FAST'})
        key.interpolation = 'LINEAR'
    for _, guard in guards:
        if not guard['exists']:
            fc.keyframe_points.insert(guard['time'], guard['value'], options={'FAST'})
    for side, guard in guards:
        segments = guard.get('left_segments' if side == 0 else 'right_segments', ())
        for segment in segments[:-1]:
            point = segment[3]
            key = fc.keyframe_points.insert(float(point[0]), float(point[1]), options={'FAST'})
            key.interpolation = 'BEZIER'
    fc.update()
    by_time = {float(key.co.x): key for key in fc.keyframe_points}
    def at(time):
        # F-curves store float32 times.  Never round arbitrary supplied
        # subframes to a decimal grid and accidentally select another key.
        return by_time[float(np.float32(time))]
    for old in outside:
        key = at(old['time'])
        # Freeze outside target handles to preserve the source curve shape
        # after adding/removing neighboring keys. Other curves are not edited.
        key.handle_left_type = 'FREE'
        key.handle_right_type = 'FREE'
        key.handle_left = old['left']
        key.handle_right = old['right']
        key.interpolation = old['interpolation']
    for side, guard in guards:
        key = at(guard['time'])
        key.handle_left_type = 'FREE'
        key.handle_right_type = 'FREE'
        if side == 0:
            key.interpolation = 'LINEAR'
            if 'left_handle' in guard:
                key.handle_left = guard['left_handle']
            if 'previous_right' in guard:
                previous = at(guard['previous'])
                previous.handle_right_type = 'FREE'
                previous.handle_right = guard['previous_right']
            if guard.get('outside_keys') and fc.extrapolation == 'LINEAR':
                slope = guard.get('outward_slope', 0.)
                # A Bezier first key exposes its independent left tangent
                # to Blender's linear extrapolation. Make its inward segment
                # a straight Bezier, preserving the intended linear blend.
                key.interpolation = 'BEZIER'
                key.handle_left = (guard['time'] - 1., guard['value'] - slope)
                following = next((k for k in fc.keyframe_points if k.co.x > key.co.x), None)
                if following is not None:
                    delta = (np.array(following.co[:]) - np.array(key.co[:])) / 3.
                    key.handle_right = np.array(key.co[:]) + delta
                    following.handle_left_type = 'FREE'
                    following.handle_left = np.array(following.co[:]) - delta
        else:
            key.interpolation = guard['interpolation']
            if 'right_handle' in guard:
                key.handle_right = guard['right_handle']
            if 'next_left' in guard:
                following = at(guard['next'])
                following.handle_left_type = 'FREE'
                following.handle_left = guard['next_left']
            if guard.get('outside_keys') and fc.extrapolation == 'LINEAR':
                slope = guard.get('outward_slope', 0.)
                key.interpolation = 'BEZIER'
                key.handle_right = (guard['time'] + 1., guard['value'] + slope)
        segments = guard.get('left_segments' if side == 0 else 'right_segments', ())
        if len(segments) > 1:
            for segment in segments:
                beginning, ending = at(segment[0, 0]), at(segment[3, 0])
                beginning.handle_right_type = 'FREE'
                ending.handle_left_type = 'FREE'
                beginning.handle_right = segment[1]
                ending.handle_left = segment[2]
                beginning.interpolation = 'BEZIER'
    fc.update()


def bake_action(arm, source, targets6, frames, quaternions, name,
                locations=None, wrist_names=None):
    """Copy and bake six rotations. Legacy wrist-location writes are rejected.

    ``targets6`` ordering matches quaternions ``[N, 6, 4]``.  The default wrist
    order is targets 2 and 5 (upper/fore/wrist per side); callers using another
    ordering must provide ``wrist_names`` explicitly. ``locations[N, 2, 3]``
    are local location values, not offsets.  Missing source location channels
    use the actual pre-bake static pose values for both boundary guards.

    On any exception, including cancellation via ``KeyboardInterrupt``, the
    source action and original source slot are restored, and the partial copy
    is removed. Existing source/result links are changed only upon success.
    """
    import bpy

    if locations is not None:
        raise ValueError('Only six-bone rotations may be baked; wrist locations are forbidden')
    targets = list(targets6)
    frames = np.asarray(frames, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if len(targets) != 6 or len(set(targets)) != 6:
        raise ValueError('Bake requires six distinct arm/wrist target bones')
    if (frames.ndim != 1 or not len(frames) or not np.isfinite(frames).all()
            or np.any(np.diff(frames) <= 0)
            or np.any(np.diff(frames.astype(np.float32)) <= 0)):
        raise ValueError('Bake frames must be finite, distinct, and strictly increasing')
    if quaternions.shape != (len(frames), 6, 4) or not np.isfinite(quaternions).all():
        raise ValueError('Bake quaternions must have finite shape [frames, 6, 4]')
    if np.any(np.linalg.norm(quaternions, axis=-1) < 1e-12):
        raise ValueError('Bake received a zero quaternion')
    wrists = list(wrist_names) if wrist_names is not None else [targets[2], targets[5]]
    if locations is not None:
        locations = np.asarray(locations, dtype=np.float64)
        if locations.shape != (len(frames), 2, 3) or not np.isfinite(locations).all():
            raise ValueError('Bake wrist locations must have finite shape [frames, 2, 3]')
        if len(wrists) != 2 or len(set(wrists)) != 2 or not set(wrists) <= set(targets):
            raise ValueError('Two distinct wrist names must be among the six targets')
    for target in targets:
        if target not in arm.pose.bones:
            raise ValueError('Missing bake target bone: ' + target)
        if arm.pose.bones[target].rotation_mode != 'QUATERNION':
            raise ValueError('Bake target must use quaternion rotation: ' + target)
    ad = arm.animation_data
    if ad is None or source is None:
        raise ValueError('Bake requires armature animation data and a source action')
    source_slot = _select_source_slot(ad, source)
    source_slot_id = source_slot.identifier
    source_end = float(source.curve_frame_range[1])
    written_end = float(np.float32(frames[-1]))
    end_guard_time = min(written_end + 1., source_end) if written_end < source_end else None
    if end_guard_time is not None and end_guard_time - written_end < .01:
        raise ValueError('Trailing source key is too close to bake end to preserve safely')
    source_fake_user = source.use_fake_user
    old_links = {key: arm.get(key) for key in ('ac_source_action', 'ac_result_action')}
    # Static defaults are captured before creating any new curves. In
    # particular, missing location paths must not acquire zero-value guards.
    defaults = {}
    for target in targets:
        pb = arm.pose.bones[target]
        defaults[(target, 'rotation_quaternion')] = tuple(pb.rotation_quaternion)
    if locations is not None:
        for target in wrists:
            defaults[(target, 'location')] = tuple(arm.pose.bones[target].location)
    action = None
    try:
        source.use_fake_user = True
        action = source.copy()
        action.name = name
        copied_slot = next((slot for slot in action.slots if slot.identifier == source_slot_id), None)
        if copied_slot is None:
            raise ValueError('Copied action did not retain the original action slot')
        ad.action = action
        ad.action_slot = copied_slot
        bag = _channelbag(action, copied_slot)
        fcurves = defaultdict_list(channels(action, copied_slot))
        jobs = [(target, 'rotation_quaternion', quaternions[:, j]) for j, target in enumerate(targets)]
        if locations is not None:
            jobs.extend((target, 'location', locations[:, j]) for j, target in enumerate(wrists))
        for target, prop, values in jobs:
            path = arm.pose.bones[target].path_from_id(prop)
            for axis in range(values.shape[1]):
                matches = fcurves.get((path, axis), [])
                if len(matches) > 1:
                    raise ValueError('Bake target appears in multiple layered channelbags: ' + path)
                fc = matches[0] if matches else bag.fcurves.new(path, index=axis)
                _write_curve(fc, frames, values[:, axis], defaults[(target, prop)][axis],
                             end_guard_time=end_guard_time)
        action['arm_clearance_source'] = source.name
        action['arm_clearance_range'] = [float(frames[0]), float(frames[-1])]
        action['arm_clearance_transition_guards'] = ([float(np.float32(frames[0])) - 1.]
            + ([end_guard_time] if end_guard_time is not None else []))
        action['arm_clearance_tail_guard_policy'] = 'within_source_curve_range'
        action.frame_range = (float(frames[0]), float(frames[-1]))
        action.use_frame_range = True
        action['arm_clearance_source_slot'] = source_slot_id
        # The result must survive saving while the user views another action.
        action.use_fake_user = True
        arm['ac_source_action'] = source.name
        arm['ac_result_action'] = action.name
        return action
    except BaseException:
        ad.action = source
        restored_slot = next((slot for slot in source.slots if slot.identifier == source_slot_id), None)
        if restored_slot is not None:
            ad.action_slot = restored_slot
        if action is not None:
            bpy.data.actions.remove(action)
        source.use_fake_user = source_fake_user
        for key, value in old_links.items():
            if value is None:
                if key in arm:
                    del arm[key]
            else:
                arm[key] = value
        raise


def defaultdict_list(curves):
    result = {}
    for curve in curves:
        result.setdefault((curve.data_path, curve.array_index), []).append(curve)
    return result


def trim_terminal_bake_guard(arm, source, result, targets6):
    """Remove only a recorded, redundant terminal guard from an old bake.

    Callers must save a backup first. This migration never solves or rebakes
    animation. All 24 target curves are validated before removing any key;
    original future animation, non-target curves and valid samples are kept.
    """
    targets = list(targets6)
    if (source is None or result is None or source == result
            or result.get('arm_clearance_source') != source.name
            or len(targets) != 6 or len(set(targets)) != 6):
        raise ValueError('Guard migration requires an original source and a six-bone bake')
    valid = np.asarray(result.get('arm_clearance_range', ()), dtype=float)
    if (valid.shape != (2,) or not np.isfinite(valid).all()
            or valid[1] < valid[0] or np.any(valid != np.rint(valid))):
        raise ValueError('Guard migration requires a recorded inclusive integer bake range')
    start, end = map(float, valid.astype(np.float32))
    guard_time = end + 1.
    if float(source.curve_frame_range[1]) > end:
        return {'removed_key_count': 0, 'reason': 'original_animation_follows'}
    guards = list(result.get('arm_clearance_transition_guards', ()))
    if guard_time not in guards:
        return {'removed_key_count': 0, 'reason': 'no_recorded_terminal_guard'}
    slot_id = result.get('arm_clearance_source_slot')
    slot = next((s for s in result.slots if s.identifier == slot_id), None)
    if slot is None or any(n not in arm.pose.bones for n in targets):
        raise ValueError('Guard migration cannot identify the original target slot/bones')
    source_slot = next((s for s in source.slots if s.identifier == slot_id), None)
    if source_slot is None:
        raise ValueError('Guard migration cannot identify the original source slot')
    grouped = defaultdict_list(channels(result, slot))
    source_grouped = defaultdict_list(channels(source, source_slot))
    chosen = []
    expected = np.arange(start, end + 1., dtype=np.float32)
    for target in targets:
        path = arm.pose.bones[target].path_from_id('rotation_quaternion')
        for axis in range(4):
            matches = grouped.get((path, axis), ())
            if len(matches) != 1:
                raise ValueError('Guard migration needs exactly one target component curve')
            original = source_grouped.get((path, axis), ())
            if len(original) != 1:
                raise ValueError('Guard migration needs an original curve to identify the redundant source key')
            fc = matches[0]
            _validate_curve(fc)
            keys = fc.keyframe_points
            times = np.asarray([float(k.co.x) for k in keys if start <= k.co.x <= end])
            if (not np.array_equal(times, expected)
                    or any(k.interpolation != 'LINEAR' for k in keys if start <= k.co.x <= end)
                    or not len(keys) or float(keys[-1].co.x) != guard_time
                    or sum(k.co.x > end for k in keys) != 1
                    or abs(float(keys[-1].co.y) - original[0].evaluate(guard_time)) > 1e-6):
                raise ValueError('Guard migration only accepts complete linear bakes with one trailing guard')
            chosen.append(fc)
    for fc in chosen:
        fc.keyframe_points.remove(fc.keyframe_points[-1], fast=True)
        fc.update()
    result['arm_clearance_transition_guards'] = [g for g in guards if g != guard_time]
    result['arm_clearance_tail_guard_policy'] = 'within_source_curve_range'
    return {'removed_key_count': len(chosen), 'removed_frame': guard_time,
            'preserved_bake_range': [start, end]}
