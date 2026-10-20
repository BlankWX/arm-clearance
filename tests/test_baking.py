"""Independent background Blender regression tests for six-bone baking."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import bpy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/test-results/baking'
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('baking_agent', ROOT / 'arm_clearance/baking.py')
baking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baking)


def key_state(key):
    return (tuple(key.co), tuple(key.handle_left), tuple(key.handle_right),
            key.handle_left_type, key.handle_right_type, key.interpolation,
            key.easing, key.type, key.amplitude, key.back, key.period,
            key.select_control_point, key.select_left_handle, key.select_right_handle)


def curve_state(curve):
    return (curve.data_path, curve.array_index, curve.extrapolation,
            curve.auto_smoothing, curve.color_mode, tuple(curve.color),
            tuple(key_state(key) for key in curve.keyframe_points),
            tuple((m.type, m.mute) for m in curve.modifiers))


def action_state(action):
    return {(slot.identifier, curve.data_path, curve.array_index): curve_state(curve)
            for slot in action.slots for curve in baking.channels(action, slot)}


def add_curve(bag, path, axis, points, interpolation='BEZIER'):
    curve = bag.fcurves.new(path, index=axis)
    for time, value in points:
        key = curve.keyframe_points.insert(time, value, options={'FAST'})
        key.interpolation = interpolation
    curve.update()
    return curve


data = bpy.data.armatures.new('AgentBakeRig')
arm = bpy.data.objects.new('AgentBakeRig', data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
names = ['腕.L', 'ひじ.L', '手首.L', '腕.R', 'ひじ.R', '手首.R']
for i, name in enumerate(names + ['人指１.L']):
    bone = data.edit_bones.new(name)
    bone.head = (i*.1, 0, 0)
    bone.tail = (i*.1, 0, .1)
    if i in (1, 2, 4, 5):
        bone.parent = data.edit_bones[names[i-1]]
    elif i == 6:
        bone.parent = data.edit_bones['手首.L']
bpy.ops.object.mode_set(mode='OBJECT')
for pb in arm.pose.bones:
    pb.rotation_mode = 'QUATERNION'
arm.pose.bones['手首.L'].location = (.02, -.03, .04)
arm.pose.bones['手首.R'].location = (-.05, .06, -.07)
source = bpy.data.actions.new('AgentSource')
decoy = source.slots.new('OBJECT', 'DecoyFirstSlot')
slot = source.slots.new('OBJECT', arm.name)
layer = source.layers.new('Original')
strip = layer.strips.new(type='KEYFRAME')
bag = strip.channelbag(slot, ensure=True)
decoy_bag = strip.channelbag(decoy, ensure=True)
ad = arm.animation_data_create()
ad.action = source
ad.action_slot = slot
path = arm.pose.bones[names[0]].path_from_id('rotation_quaternion')
nonlinear = add_curve(bag, path, 0, [(-10, .8), (0, .9), (20, .6), (40, .7)])
for key in nonlinear.keyframe_points:
    key.handle_left_type = 'FREE'
    key.handle_right_type = 'FREE'
nonlinear.keyframe_points[0].handle_left = (-14, .4)
nonlinear.keyframe_points[0].handle_right = (-5, 2)
nonlinear.keyframe_points[1].handle_left = (-4, -.6)
nonlinear.keyframe_points[1].handle_right = (18, 1.6)
nonlinear.keyframe_points[2].handle_left = (-6, -1.2)
nonlinear.keyframe_points[2].handle_right = (25, 1.3)
nonlinear.keyframe_points[3].handle_left = (34, .3)
nonlinear.keyframe_points[3].handle_right = (45, 1.1)
nonlinear.update()
add_curve(decoy_bag, path, 0, [(0, 99), (50, 111)])
# Existing guard times and AUTO handles on a separate touched path.
existing = add_curve(bag, arm.pose.bones[names[1]].path_from_id('rotation_quaternion'),
                     1, [(-10, -.1), (4, .2), (11, -.2), (40, .1)])
existing.extrapolation = 'LINEAR'
# A one-key source curve whose original extrapolation is nevertheless constant.
single = add_curve(bag, arm.pose.bones[names[4]].path_from_id('rotation_quaternion'), 0, [(6, .95)])
single.extrapolation = 'LINEAR'
# A nearly flat x(t) inside the protected outward segment requires an extra
# De Casteljau support key, or Blender would reclamp the shortened handle.
flat = add_curve(bag, arm.pose.bones[names[2]].path_from_id('rotation_quaternion'), 2, [(0, 0), (7, 1)])
for key in flat.keyframe_points:
    key.handle_left_type = 'FREE'
    key.handle_right_type = 'FREE'
flat.keyframe_points[0].handle_right = (6.5, 3)
flat.keyframe_points[1].handle_left = (.5, -2)
flat.update()
assert len(baking.curve_boundary(flat, 4)['left_segments']) > 1
locpath = arm.pose.bones['手首.R'].path_from_id('location')
location = add_curve(bag, locpath, 0, [(6, .05), (8, .08)])
location.extrapolation = 'LINEAR'
for key in location.keyframe_points:
    key.handle_left_type = 'FREE'
    key.handle_right_type = 'FREE'
location.keyframe_points[0].handle_left = (5, .02)
location.keyframe_points[0].handle_right = (6.8, .1)
location.keyframe_points[1].handle_left = (7.4, .02)
location.keyframe_points[1].handle_right = (9, .12)
location.update()
finger = add_curve(bag, arm.pose.bones['人指１.L'].path_from_id('rotation_quaternion'),
                   2, [(-4, .1), (6, .6), (20, -.2), (50, .1)])
finger.keyframe_points[1].handle_left_type = 'ALIGNED'
finger.keyframe_points[1].handle_right_type = 'ALIGNED'
finger.keyframe_points[1].easing = 'EASE_IN_OUT'
finger.keyframe_points[1].type = 'BREAKDOWN'
finger.modifiers.new(type='NOISE').strength = .1
finger.update()
original_state = action_state(source)
original_slot = slot.identifier
defaults = {name: {'rotation_quaternion': tuple(arm.pose.bones[name].rotation_quaternion),
                   'location': tuple(arm.pose.bones[name].location)} for name in names}
times = np.r_[np.arange(-30, 4.001, .125), np.arange(11, 60.001, .125)]
source_curves = {(fc.data_path, fc.array_index): fc for fc in baking.channels(source, slot)}
source_evals = {key: np.array([fc.evaluate(float(t)) for t in times]) for key, fc in source_curves.items()}
frames = np.arange(5, 11)
quats = np.zeros((len(frames), 6, 4))
quats[:, :, 0] = 1
quats[:, :, 1] = .025
locations = np.array([[defaults['手首.L']['location'], defaults['手首.R']['location']]] * len(frames))
locations[:, 0] += (.01, -.005, .002)
locations[:, 1] += (-.004, .003, -.001)
result = baking.bake_action(arm, source, names, frames, quats, 'AgentResult', locations)
assert ad.action == result
assert ad.action_slot.identifier == original_slot
assert [s.identifier for s in result.slots] == [s.identifier for s in source.slots]
assert action_state(source) == original_state, 'Source keyframes were modified'
result_state = action_state(result)
finger_key = (original_slot, finger.data_path, finger.array_index)
assert result_state[finger_key] == original_state[finger_key], 'Finger handles/types/keys changed'
for key in original_state:
    if key[0] == decoy.identifier:
        assert result_state[key] == original_state[key], 'Wrong action slot was edited'
new_curves = {(fc.data_path, fc.array_index): fc for fc in baking.channels(result, ad.action_slot)}
outside_errors = {}
for key, baseline in source_evals.items():
    updated = np.array([new_curves[key].evaluate(float(t)) for t in times])
    outside_errors[str(key)] = float(np.max(np.abs(updated - baseline)))
max_error = max(outside_errors.values())
if max_error >= 3e-6:
    key = (nonlinear.data_path, nonlinear.array_index)
    print('DEBUG_GUARDS', baking.curve_boundary(nonlinear, 4), baking.curve_boundary(nonlinear, 11))
    print('DEBUG_KEYS_SOURCE', [key_state(k)[:3] for k in nonlinear.keyframe_points])
    print('DEBUG_KEYS_RESULT', [key_state(k)[:3] for k in new_curves[key].keyframe_points])
    print('DEBUG_EVAL', [(t, nonlinear.evaluate(t), new_curves[key].evaluate(t)) for t in (0, 1, 2, 3, 4, 11, 12, 15, 19, 20)])
assert max_error < 3e-6, ('Outside guard curve error', max_error, outside_errors)
static_error = 0.
for name in ('手首.L', '手首.R'):
    path = arm.pose.bones[name].path_from_id('location')
    for axis in range(3):
        key = (path, axis)
        if key in source_curves:
            continue
        fc = new_curves[key]
        expected = defaults[name]['location'][axis]
        static_error = max(static_error, max(abs(fc.evaluate(t) - expected) for t in (-30, 4, 11, 60)))
assert static_error < 1e-7, ('Missing location used zero rather than original static value', static_error)
for i, frame in enumerate(frames):
    for j, name in enumerate(names):
        path = arm.pose.bones[name].path_from_id('rotation_quaternion')
        for axis in range(4):
            assert abs(new_curves[(path, axis)].evaluate(float(frame)) - quats[i, j, axis]) < 1e-7
    for j, name in enumerate(('手首.L', '手首.R')):
        path = arm.pose.bones[name].path_from_id('location')
        for axis in range(3):
            assert abs(new_curves[(path, axis)].evaluate(float(frame)) - locations[i, j, axis]) < 1e-7
# Rotation-only mode must not add wrist location channels.
ad.action = source
ad.action_slot = slot
rotation_only = baking.bake_action(arm, source, names, frames, quats, 'AgentRotationOnly')
rotation_curves = {(fc.data_path, fc.array_index): fc for fc in baking.channels(rotation_only, ad.action_slot)}
assert sum('.location' in key[0] for key in rotation_curves) == sum('.location' in key[0] for key in source_curves)
old_links = {key: arm[key] for key in ('ac_source_action', 'ac_result_action')}
action_count = len(bpy.data.actions)
# Failure occurs after some channels were already baked in the partial copy.
modifier = single.modifiers.new(type='NOISE')
try:
    baking.bake_action(arm, source, names, frames, quats, 'AgentMustBeRemoved', locations)
except ValueError:
    pass
else:
    raise AssertionError('Active target modifier was not rejected')
single.modifiers.remove(modifier)
assert ad.action == source and ad.action_slot.identifier == original_slot
assert len(bpy.data.actions) == action_count and bpy.data.actions.get('AgentMustBeRemoved') is None
assert all(arm[key] == value for key, value in old_links.items())
# A KeyboardInterrupt models cancellation in a synchronous bake.
original_write = baking._write_curve
counter = [0]
def interrupting_write(*args, **kwargs):
    counter[0] += 1
    if counter[0] == 3:
        raise KeyboardInterrupt('Synthetic user cancellation')
    return original_write(*args, **kwargs)
baking._write_curve = interrupting_write
try:
    baking.bake_action(arm, source, names, frames, quats, 'AgentCancelled', locations)
except KeyboardInterrupt:
    pass
else:
    raise AssertionError('Synthetic cancellation was not raised')
finally:
    baking._write_curve = original_write
assert ad.action == source and ad.action_slot.identifier == original_slot
assert len(bpy.data.actions) == action_count and bpy.data.actions.get('AgentCancelled') is None
assert all(arm[key] == value for key, value in old_links.items())
assert action_state(source) == original_state
evidence = {'nonlinear_bezier_guard_outside_max_error': max_error,
            'flat_time_bezier_guard_support_subdivision': True,
            'static_nonzero_missing_location_guard_max_error': static_error,
            'outside_error_by_source_curve': outside_errors,
            'protected_domain': 'frame <= start-1 and frame >= end+1; adjacent one-frame transitions are explicit',
            'finger_keys_handle_types_and_modifiers_unchanged': True,
            'source_and_other_action_slot_unchanged': True,
            'copied_active_slot_identifier': original_slot,
            'rotation_only_adds_no_location_channels': True,
            'exception_restores_source_and_slot_removes_partial_action': True,
            'keyboard_interrupt_restores_source_and_slot_removes_partial_action': True}
(OUT / 'baking.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(evidence, ensure_ascii=False, indent=2))
