"""Validate the delivered v2 ZIP in a fresh Blender 5.2 process.

Example:
  blender -b --factory-startup --python-exit-code 1 --python test_distribution.py
  blender -b --factory-startup --python-exit-code 1 --python test_distribution.py -- --zip PATH --skip-synthetic

Optional globals: AC_V2_ZIP, AC_V2_DISTRIBUTION_REPORT,
AC_V2_RUN_SYNTHETIC_BAKE. No user scene or delivery file is saved/modified.
"""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('腕.L', 'ひじ.L', '手首.L', '腕.R', 'ひじ.R', '手首.R')


def source_version():
    tree = ast.parse((ROOT/'arm_clearance/__init__.py').read_bytes())
    info = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'bl_info' for t in node.targets))
    return '.'.join(map(str, info['version']))


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', dest='zip_path', default=globals().get('AC_V2_ZIP', ROOT/('dist/arm_clearance-v' + source_version() + '.zip')))
    parser.add_argument('--report', default=globals().get('AC_V2_DISTRIBUTION_REPORT', ROOT/'build/test-results/distribution.json'))
    parser.add_argument('--skip-synthetic', action='store_true')
    return parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])


def dependency_audit(path):
    required = {'arm_clearance/__init__.py', 'arm_clearance/pipeline_v2.py',
                'arm_clearance/pipeline.py', 'arm_clearance/solver_v2.py',
                'arm_clearance/solver.py', 'arm_clearance/solver_v2_derivatives.py',
                'arm_clearance/baking.py', 'arm_clearance/stationary.py', 'arm_clearance/progress.py', 'arm_clearance/temporal.py',
                'arm_clearance/hand_geometry.py', 'arm_clearance/arm_geometry.py',
                'arm_clearance/body_field.py', 'arm_clearance/full_surface.py'}
    blender_modules = {'bpy', 'bmesh', 'mathutils', 'numpy', 'gpu', 'gpu_extras', 'blf'}
    external, internal, digests = set(), set(), {}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        assert required <= names, 'Missing packaged modules: '+str(sorted(required-names))
        assert archive.testzip() is None, 'ZIP CRC failure'
        for name in names:
            normalized = name.replace('\\', '/')
            assert not normalized.startswith('/') and '..' not in normalized.split('/'), name
        for name in sorted(n for n in names if n.startswith('arm_clearance/') and n.endswith('.py')):
            source = archive.read(name)
            digests[name] = hashlib.sha256(source).hexdigest()
            tree = ast.parse(source, filename=str(path)+'/'+name)
            compile(tree, str(path)+'/'+name, 'exec')
            module = name[:-3].replace('/', '.')
            package = module[:-9] if module.endswith('.__init__') else module.rpartition('.')[0]
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        if node.module:
                            modules = [importlib.util.resolve_name('.'*node.level+node.module, package)]
                        else:
                            modules = [importlib.util.resolve_name('.'*node.level+alias.name, package) for alias in node.names]
                    elif node.module:
                        modules = [node.module]
                for imported in modules:
                    if imported == 'arm_clearance' or imported.startswith('arm_clearance.'):
                        stem = imported.replace('.', '/')
                        assert stem+'.py' in names or stem+'/__init__.py' in names, (name, imported, 'Not bundled')
                        internal.add(imported)
                    else:
                        root = imported.partition('.')[0]
                        assert root in sys.stdlib_module_names or root in blender_modules, (name, imported, 'Requires an unbundled third-party package')
                        assert importlib.util.find_spec(root) is not None, (name, imported, 'Unavailable in this Blender')
                        external.add(root)
    return {'module_sha256': digests, 'internal_dependencies': sorted(internal),
            'available_builtin_dependencies': sorted(external)}


def curve_data(curve):
    return (curve.data_path, curve.array_index, curve.extrapolation,
            tuple((tuple(k.co), tuple(k.handle_left), tuple(k.handle_right),
                   k.handle_left_type, k.handle_right_type, k.interpolation,
                   k.easing, k.amplitude, k.back, k.period)
                  for k in curve.keyframe_points))


def synthetic_bake(package):
    from arm_clearance.baking import bake_action, channels
    rig_data = bpy.data.armatures.new('_AC_DistributionProbe')
    rig = bpy.data.objects.new('_AC_DistributionProbe', rig_data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for index, name in enumerate(TARGETS+('人指１.L',)):
        bone = rig_data.edit_bones.new(name)
        side = 0 if index < 3 or index == 6 else 1
        segment = index % 3 if index < 6 else 3
        bone.head = ((1 if side == 0 else -1)*(.1+segment*.15), 0., 1.)
        bone.tail = (bone.head.x+(1 if side == 0 else -1)*.12, 0., 1.)
        if index in (1, 2, 4, 5):
            bone.parent = rig_data.edit_bones[TARGETS[index-1]]
        elif index == 6:
            bone.parent = rig_data.edit_bones['手首.L']
    bpy.ops.object.mode_set(mode='OBJECT')
    for pb in rig.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    for side, name in enumerate(('手首.L', '手首.R')):
        rig.pose.bones[name].location = (.012*(1 if side == 0 else -1), .013, -.014)
    source = bpy.data.actions.new('_AC_DistributionSource')
    slot = source.slots.new('OBJECT', rig.name)
    layer = source.layers.new('Probe')
    strip = layer.strips.new(type='KEYFRAME')
    bag = strip.channelbag(slot, ensure=True)
    ad = rig.animation_data_create(); ad.action = source; ad.action_slot = slot
    for name in TARGETS+('人指１.L',):
        path = rig.pose.bones[name].path_from_id('rotation_quaternion')
        for axis in range(4):
            curve = bag.fcurves.new(path, index=axis)
            value = 1. if axis == 0 else (.12 if name == '人指１.L' and axis == 3 else 0.)
            for frame in (-2., 6.):
                key = curve.keyframe_points.insert(frame, value, options={'FAST'})
                key.interpolation = 'LINEAR'
            curve.update()
    original = {(f.data_path, f.array_index): curve_data(f) for f in channels(source, slot)}
    previous = source.copy(); previous.name = '_AC_DistributionPreviousV1'
    previous.use_fake_user = True
    rig['ac_previous_result_action'] = previous.name
    mesh_data = bpy.data.meshes.new('_AC_DistributionProbeMesh')
    mesh_data.from_pydata([(0., 0., 0.), (.1, 0., 0.), (0., .1, 0.)], [], [(0, 1, 2)])
    mesh = bpy.data.objects.new('_AC_DistributionProbeMesh', mesh_data)
    bpy.context.scene.collection.objects.link(mesh)
    mesh.modifiers.new('Armature', 'ARMATURE').object = rig
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 3
    assert bpy.ops.arm_clearance.autofill() == {'FINISHED'}
    assert scene.arm_clearance.armature == rig and scene.arm_clearance.body_mesh == mesh
    frames = np.array([1, 2, 3])
    quats = np.empty((3, 6, 4)); axis = np.array([.3, .4, .5]); axis /= np.linalg.norm(axis)
    for i in range(3):
        for j in range(6):
            angle = .03*(i+1)*(j+1)
            quats[i, j] = np.r_[np.cos(angle/2), axis*np.sin(angle/2)]
    locations = np.array([[[.012, .013, -.014], [-.012, .013, -.014]]]*3)
    locations += np.arange(1, 4)[:, None, None]*np.array([.001, -.002, .001])
    result = bake_action(rig, source, TARGETS, frames, quats,
                         '_AC_DistributionResult', locations=locations)
    result_slot = ad.action_slot
    current = {(f.data_path, f.array_index): f for f in channels(result, result_slot)}
    allowed = {(rig.pose.bones[n].path_from_id('rotation_quaternion'), a) for n in TARGETS for a in range(4)}
    allowed |= {(rig.pose.bones[n].path_from_id('location'), a) for n in ('手首.L', '手首.R') for a in range(3)}
    assert len(allowed) == 30 and allowed <= set(current)
    changed = {key for key, state in original.items() if curve_data(current[key]) != state}
    added = set(current)-set(original)
    assert changed | added <= allowed
    assert {(f.data_path, f.array_index): curve_data(f) for f in channels(source, slot)} == original
    assert all(curve_data(current[key]) == state for key, state in original.items() if key not in allowed)
    for i, frame in enumerate(frames):
        for j, name in enumerate(TARGETS):
            path = rig.pose.bones[name].path_from_id('rotation_quaternion')
            for component in range(4):
                assert abs(current[(path, component)].evaluate(int(frame))-quats[i, j, component]) < 1e-6
        for side, name in enumerate(('手首.L', '手首.R')):
            path = rig.pose.bones[name].path_from_id('location')
            for component in range(3):
                assert abs(current[(path, component)].evaluate(int(frame))-locations[i, side, component]) < 1e-7
    assert bpy.ops.arm_clearance.compare() == {'FINISHED'} and ad.action == source
    assert bpy.ops.arm_clearance.compare() == {'FINISHED'} and ad.action == result
    assert bpy.ops.arm_clearance.previous() == {'FINISHED'} and ad.action == previous
    assert bpy.ops.arm_clearance.compare() == {'FINISHED'} and ad.action == result
    evidence = {'six_bone_rotation_channels': 24, 'wrist_location_channels': 6,
                'changed_or_added_channels': len(changed | added), 'source_and_finger_channels_unchanged': True,
                'autofill_compare_previous_operators': 'passed', 'bake_values': 'passed'}
    scene.arm_clearance.armature = None; scene.arm_clearance.body_mesh = None
    bpy.data.objects.remove(mesh, do_unlink=True); bpy.data.meshes.remove(mesh_data)
    bpy.data.objects.remove(rig, do_unlink=True); bpy.data.armatures.remove(rig_data)
    for action in (source, previous, result):
        bpy.data.actions.remove(action)
    return evidence


def main(options):
    path = Path(options.zip_path).resolve()
    assert bpy.app.version[:2] == (5, 2), ('Use Blender 5.2 for delivery validation', bpy.app.version_string)
    assert path.is_file(), path
    assert not any(name == 'arm_clearance' or name.startswith('arm_clearance.') for name in sys.modules), 'Start a fresh --factory-startup process; an addon module is already loaded'
    report = {'zip': str(path), 'blender': bpy.app.version_string, 'dependencies': dependency_audit(path)}
    sys.path.insert(0, str(path)); importlib.invalidate_caches()
    package = importlib.import_module('arm_clearance')
    normalized = lambda value: str(value).replace('\\', '/').lower()
    assert normalized(path) in normalized(package.__file__), package.__file__
    for imported in report['dependencies']['internal_dependencies']:
        module = importlib.import_module(imported)
        assert normalized(path) in normalized(module.__file__), (imported, module.__file__)
    assert tuple(package.bl_info['version']) >= (0, 2, 0)
    handlers = (tuple(bpy.app.handlers.frame_change_pre), tuple(bpy.app.handlers.frame_change_post))
    package.register()
    try:
        assert hasattr(bpy.types.Scene, 'arm_clearance')
        settings = bpy.context.scene.arm_clearance
        expected = dict(upper_left='腕.L', fore_left='ひじ.L', wrist_left='手首.L',
                        upper_right='腕.R', fore_right='ひじ.R', wrist_right='手首.R')
        assert all(getattr(settings, key) == value for key, value in expected.items())
        for key in ('armature', 'body_mesh', 'margin', 'upper_pose_weight', 'wrist_translation_limit',
                    'iterations', 'velocity_weight', 'acceleration_weight', 'report_name', 'busy', 'console_auto_open'):
            assert key in settings.bl_rna.properties, key
        assert settings.console_auto_open is True
        assert settings.upper_pose_weight > 0 and settings.wrist_translation_limit >= 0
        operators = ('autofill', 'console', 'bake', 'compare', 'previous')
        for name in operators:
            assert getattr(bpy.ops.arm_clearance, name).get_rna_type() is not None, name
        panel = bpy.types.Panel.bl_rna_get_subclass_py('AC_PT_panel')
        assert panel is not None and panel.bl_space_type == 'VIEW_3D' and panel.bl_region_type == 'UI'
        assert panel.bl_category == '手臂避碰'
        if not options.skip_synthetic and globals().get('AC_V2_RUN_SYNTHETIC_BAKE', True):
            report['synthetic_bake'] = synthetic_bake(package)
        else:
            report['synthetic_bake'] = 'skipped by caller'
        assert handlers == (tuple(bpy.app.handlers.frame_change_pre), tuple(bpy.app.handlers.frame_change_post))
        report.update(version=list(package.bl_info['version']), author=package.bl_info['author'], registration='passed',
                      operator_ids=list(operators), six_bone_settings=expected,
                      no_frame_change_handlers_installed=True)
    finally:
        package.unregister()
    assert not hasattr(bpy.types.Scene, 'arm_clearance')
    package.register(); package.unregister()
    assert not hasattr(bpy.types.Scene, 'arm_clearance')
    report['clean_unregister_and_reregister'] = 'passed'
    destination = Path(options.report).resolve(); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    print('DISTRIBUTION V2 PASS '+json.dumps({'zip': str(path), 'report': str(destination)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main(arguments())
