"""Exercise real action baking/commit/rollback and independently reopen files.

Only collision/model computation is replaced with deterministic substitutes.
The rig, source/action slots, bake_action, run_job lifecycle, comparison
operators, .blend serialization, and independent Blender reloads are real.
No MCP connection or existing user project is used.

  blender -b --factory-startup --python-exit-code 1 --python test_persistence.py
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/test-results/persistence'
sys.path.insert(0, str(ROOT))
import arm_clearance as package
from arm_clearance import pipeline_v2 as pipeline
from arm_clearance.baking import channels
from arm_clearance.progress import ConsoleProgress

TARGETS = ('腕.L', 'ひじ.L', '手首.L', '腕.R', 'ひじ.R', '手首.R')
FINGER = '人指１.L'
LINKS = ('ac_source_action', 'ac_result_action', 'ac_previous_result_action', 'ac_review_action')


def digest(action):
    """Hash every slot/channel, not only the active target slot."""
    state = []
    for slot in action.slots:
        curves = []
        for curve in channels(action, slot):
            curves.append((curve.data_path, curve.array_index, curve.extrapolation,
                [(tuple(k.co), tuple(k.handle_left), tuple(k.handle_right),
                  k.handle_left_type, k.handle_right_type, k.interpolation)
                 for k in curve.keyframe_points]))
        state.append((slot.identifier, sorted(curves)))
    return hashlib.sha256(json.dumps(state, ensure_ascii=False, sort_keys=True).encode('utf8')).hexdigest()


def fixture():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if not hasattr(bpy.types.Scene, 'arm_clearance'):
        package.register()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 3
    data = bpy.data.armatures.new('PersistenceRigData')
    arm = bpy.data.objects.new('PersistenceRig', data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for i, name in enumerate(TARGETS + (FINGER,)):
        bone = data.edit_bones.new(name)
        side = 0 if i < 3 or i == 6 else 1
        segment = i % 3 if i < 6 else 3
        sign = 1 if side == 0 else -1
        bone.head = (sign * (.1 + segment * .15), 0., 1.)
        bone.tail = (sign * (.22 + segment * .15), 0., 1.)
        if i in (1, 2, 4, 5):
            bone.parent = data.edit_bones[TARGETS[i - 1]]
        elif i == 6:
            bone.parent = data.edit_bones['手首.L']
    bpy.ops.object.mode_set(mode='OBJECT')
    for bone in arm.pose.bones:
        bone.rotation_mode = 'QUATERNION'
    source = bpy.data.actions.new('PersistenceSource')
    # An unrelated first slot makes implicit-first-slot binding detectable.
    decoy = source.slots.new('OBJECT', 'UnrelatedObject')
    slot = source.slots.new('OBJECT', arm.name)
    strip = source.layers.new('Source').strips.new(type='KEYFRAME')
    decoy_bag = strip.channelbag(decoy, ensure=True)
    decoy_curve = decoy_bag.fcurves.new('location', index=0)
    decoy_curve.keyframe_points.insert(1., 123.)
    bag = strip.channelbag(slot, ensure=True)
    for name in TARGETS + (FINGER,):
        for axis in range(4):
            curve = bag.fcurves.new(arm.pose.bones[name].path_from_id('rotation_quaternion'), index=axis)
            value = 1. if axis == 0 else (.12 if name == FINGER and axis == 3 else 0.)
            for frame in (0., 4.):
                key = curve.keyframe_points.insert(frame, value)
                key.interpolation = 'LINEAR'
            curve.update()
    ad = arm.animation_data_create()
    ad.action, ad.action_slot = source, slot
    prior = source.copy()
    prior.name = 'PersistencePreviousPassed'
    prior.use_fake_user = True
    prior['arm_clearance_source'] = source.name
    prior['arm_clearance_source_slot'] = slot.identifier
    prior['arm_clearance_validation_status'] = 'passed'
    arm['ac_source_action'] = source.name
    arm['ac_result_action'] = prior.name
    mesh_data = bpy.data.meshes.new('PersistenceMeshData')
    mesh_data.from_pydata([(0., 0., 0.), (.1, 0., 0.), (0., .1, 0.)], [], [(0, 1, 2)])
    mesh = bpy.data.objects.new('PersistenceMesh', mesh_data)
    scene.collection.objects.link(mesh)
    mesh.modifiers.new('Armature', 'ARMATURE').object = arm
    settings = scene.arm_clearance
    settings.armature, settings.body_mesh = arm, mesh
    settings.iterations = 50
    scene.frame_set(2, subframe=.25)
    return scene, arm, source, prior, slot.identifier, settings


class DummyBody:
    report = {'synthetic_collision_model': True}
    narrow_band = .1
    def restore_full_planes(self):
        pass


class DummyModel:
    def __init__(self, matrices, quats, bases, geo, meta, hand=None, margin=0.):
        self.n, self.base = len(matrices), np.array(quats, copy=True)
        self.length, self.margin, self.hand = 1., margin, hand
        self.radii = np.full((self.n, 2, 1), .01)
    def quats(self, delta):
        q = self.base.copy()
        for i in range(self.n):
            for j in range(6):
                angle = .05 * (i + 1) * (j + 1)
                q[i, j] = (np.cos(angle / 2), 0., 0., np.sin(angle / 2))
        return q


class DummySolver:
    def __init__(self, model, collider, matrices, progress):
        self.model, self.body, self.progress = model, DummyBody(), progress
        self.pose_weights = np.ones(8)
        self.history = []
        self.optimization_stage_count = self.optimization_stage_index = 1
        self.result = np.zeros((model.n, 8, 3))
        self.result[:, 6:, 0] = np.arange(1, model.n + 1)[:, None] * .001
        for name in ('body_weights', 'mutual_weights', 'own_weights'):
            setattr(self, name, np.ones((model.n, 2, 1)))
        self.body_margin = np.zeros((model.n, 2, 1))
    def prepare(self):
        yield None
    def optimize_staged(self, *args, **kwargs):
        self.progress('solve', 0, 1, {'energy': 123.45, 'body_max_mm': 0.})
        if kwargs.get('free_frames') is not None:
            self.last_optimization_scope = {'masked': True}
        self.history.append({'synthetic': True})
        yield None


def install_doubles(case, arm):
    names = ('geometry_data', 'closed_collider', 'build_hand_geometry', 'FullLimbModel',
             'FullLimbSolver', 'full_metrics', 'surface_check', 'build_surface_repair_weights',
             'merge_surface_repair_fields', 'bake_action')
    original = {name: getattr(pipeline, name) for name in names}
    bone_names = [b.name for b in arm.pose.bones]
    target_ids = [bone_names.index(n) for n in (TARGETS[0], TARGETS[1], TARGETS[3], TARGETS[4])]
    def geometry(*args):
        return ({'vertices': np.zeros((3, 3)), 'triangles': np.array([[0, 1, 2]])},
                {'bones': bone_names, 'target_ids': target_ids}, np.array([[0, 1, 2]]))
    pipeline.geometry_data = geometry
    pipeline.closed_collider = lambda *args: object()
    pipeline.build_hand_geometry = lambda *args, **kwargs: SimpleNamespace(calibration={'synthetic': True})
    pipeline.FullLimbModel, pipeline.FullLimbSolver = DummyModel, DummySolver
    pipeline.full_metrics = lambda *args: {'body_max_mm': 21. if case == 'deep_proxy' else 0.}
    def surface(*args, **kwargs):
        failed = case in ('surface', 'second_bake_error')
        report = {'complete': True, 'samples': 5, 'total_samples': 5,
                  'collision_samples': int(failed), 'problem_frames': [2.] if failed else [],
                  'categories': {'arm-body': {'intersection_samples': int(failed)}}, 'records': []}
        yield report
        if case == 'exception':
            raise RuntimeError('Injected failure after complete bake')
    pipeline.surface_check = surface
    def repair(surface, model, geo, **kwargs):
        fields = {name: np.ones((model.n, 2, 1)) for name in ('body_weights', 'mutual_weights', 'own_weights')}
        fields['body_margin'] = np.zeros((model.n, 2, 1))
        return fields, {'synthetic': True}
    pipeline.build_surface_repair_weights = repair
    pipeline.merge_surface_repair_fields = lambda current, fresh, **kwargs: fresh
    if case == 'second_bake_error':
        calls = 0
        def bake(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('Injected failure before second candidate completes')
            return original['bake_action'](*args, **kwargs)
        pipeline.bake_action = bake
    return original


def save_manifest(case, scene, arm, source, prior, slot_id, settings, report, destination):
    manifest = {'case': case, 'blend': str(destination), 'armature': arm.name,
        'source': source.name, 'prior': prior.name, 'slot_id': slot_id,
        'active': arm.animation_data.action.name,
        'active_slot': arm.animation_data.action_slot.identifier,
        'frame': scene.frame_current, 'subframe': scene.frame_subframe,
        'links': {key: arm.get(key) for key in LINKS},
        'actions': {a.name: {'sha256': digest(a), 'fake_user': a.use_fake_user,
                           'validation_status': a.get('arm_clearance_validation_status')}
                    for a in bpy.data.actions},
        'report_name': settings.report_name,
        'report_fake_user': bpy.data.texts[settings.report_name].use_fake_user,
        'report': report}
    path = destination.with_suffix('.manifest.json')
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    bpy.ops.wm.save_as_mainfile(filepath=str(destination), check_existing=False)
    return path



class RecordingConsoleProgress(ConsoleProgress):
    """Observe emitted terminal output without depending on its display prefix."""
    def __init__(self, case):
        self.capture = io.StringIO()
        self.terminal_emissions = []
        super().__init__(scope='persistence:' + case, stream=self.capture)
    def finish(self, outcome, detail=''):
        before = self.capture.getvalue()
        super().finish(outcome, detail)
        if self.capture.getvalue() != before:
            self.terminal_emissions.append(outcome)


def build_case(case):
    scene, arm, source, prior, slot_id, settings = fixture()
    source_hash, prior_hash = digest(source), digest(prior)
    original = install_doubles(case, arm)
    console = RecordingConsoleProgress(case)
    try:
        job = pipeline.run_job(scene, settings, lambda message: setattr(settings, 'status', message), console=console)
        if case == 'early_cancel':
            next(job)
            job.close()
        elif case == 'cancel':
            while True:
                next(job)
                active = arm.animation_data.action
                if active != source and active.get('arm_clearance_report'):
                    break
            job.close()
        else:
            try:
                for _ in job:
                    pass
            except RuntimeError as error:
                assert case in ('exception', 'second_bake_error') and 'Injected failure' in str(error), str(error)
            else:
                assert case not in ('exception', 'second_bake_error')
    finally:
        for name, value in original.items():
            setattr(pipeline, name, value)
    report_text = bpy.data.texts[settings.report_name]
    assert report_text.use_fake_user, 'Report Text must survive unreferenced saves and appends'
    report = json.loads(report_text.as_string())
    expected_outcome = ('complete' if case == 'success' else
                        'needs_review' if case in ('surface', 'deep_proxy') else
                        'cancelled' if case in ('cancel', 'early_cancel') else 'error')
    assert console.finished and console.outcome == expected_outcome
    assert console.terminal_emissions == [expected_outcome], console.terminal_emissions
    assert console.capture.getvalue().count('[' + expected_outcome + ']') == 1
    assert all('[' + other + ']' not in console.capture.getvalue()
               for other in ('complete', 'needs_review', 'cancelled', 'error') if other != expected_outcome)
    assert report['status'] == console.outcome
    assert console.capture.getvalue() and console.last_message
    assert '读取原动作' in console.capture.getvalue()
    if case != 'early_cancel':
        assert '主优化 第1/1组' in console.capture.getvalue()
    (OUT / (case + '.console.log')).write_text(console.capture.getvalue(), encoding='utf8')
    assert digest(source) == source_hash and digest(prior) == prior_hash
    assert scene.frame_current == 2 and abs(scene.frame_subframe - .25) < 1e-6
    assert arm.animation_data.action_slot.identifier == slot_id
    if case == 'early_cancel':
        assert report['status'] == 'cancelled'
        assert arm.animation_data.action == source
        assert not arm.get('ac_review_action')
        assert {a.name for a in bpy.data.actions} == {source.name, prior.name}
    else:
        assert report['result_preserved'] and not report.get('candidate_discarded', False), report
        candidate = bpy.data.actions[report['result_action'] if case == 'success' else report['review_action']]
        assert candidate.use_fake_user and candidate.get('arm_clearance_report') == settings.report_name
        assert candidate.get('arm_clearance_source') == source.name
        assert list(candidate['arm_clearance_range']) == [1., 3.]
        assert digest(candidate) != source_hash
        candidate_slot = next(s for s in candidate.slots if s.identifier == slot_id)
        source_slot = next(s for s in source.slots if s.identifier == slot_id)
        finger_path = arm.pose.bones[FINGER].path_from_id('rotation_quaternion')
        original_finger = [(c.array_index, [tuple(k.co) for k in c.keyframe_points])
                           for c in channels(source, source_slot) if c.data_path == finger_path]
        candidate_finger = [(c.array_index, [tuple(k.co) for k in c.keyframe_points])
                            for c in channels(candidate, candidate_slot) if c.data_path == finger_path]
        assert original_finger == candidate_finger
        if case == 'success':
            assert report['status'] == 'complete' and report['validation_passed']
            assert arm['ac_result_action'] == candidate.name and not arm.get('ac_review_action')
            assert arm.animation_data.action == candidate
            assert candidate['arm_clearance_validation_status'] == 'passed'
        elif case in ('surface', 'deep_proxy'):
            assert report['status'] == 'needs_review' and not report['validation_passed']
            assert arm['ac_review_action'] == candidate.name and arm['ac_result_action'] == prior.name
            assert arm.animation_data.action == candidate
            assert candidate['arm_clearance_validation_status'] == 'needs_review'
            assert len(report['attempts']) == (4 if case == 'surface' else 1)
        else:
            assert report['status'] == ('cancelled' if case == 'cancel' else 'error')
            assert arm['ac_review_action'] == candidate.name and arm['ac_result_action'] == prior.name
            assert arm.animation_data.action == source
            assert candidate['arm_clearance_validation_status'] == 'unverified'
        # Compare must prioritize retained review data and bind the same
        # named source slot even though an unrelated slot comes first.
        settings.busy = False
        arm.animation_data.action = source
        arm.animation_data.action_slot = source_slot
        assert bpy.ops.arm_clearance.compare() == {'FINISHED'}
        assert arm.animation_data.action == candidate and arm.animation_data.action_slot.identifier == slot_id
        assert bpy.ops.arm_clearance.compare() == {'FINISHED'}
        assert arm.animation_data.action == source and arm.animation_data.action_slot.identifier == slot_id
        # Save once with the source active to prove fake-user/link retention.
        manifests = [save_manifest(case, scene, arm, source, prior, slot_id, settings, report,
                                  OUT / (case + '_source_active.blend'))]
        assert bpy.ops.arm_clearance.compare() == {'FINISHED'}
        manifests.append(save_manifest(case, scene, arm, source, prior, slot_id, settings, report,
                                       OUT / (case + '_candidate_active.blend')))
        return manifests
    return [save_manifest(case, scene, arm, source, prior, slot_id, settings, report,
                          OUT / (case + '.blend'))]


def reopen(manifest_path):
    expected = json.loads(manifest_path.read_text(encoding='utf8'))
    if not hasattr(bpy.types.Scene, 'arm_clearance'):
        package.register()
    bpy.ops.wm.open_mainfile(filepath=expected['blend'])
    arm = bpy.data.objects[expected['armature']]
    scene = bpy.context.scene
    assert arm.animation_data.action.name == expected['active']
    assert arm.animation_data.action_slot.identifier == expected['active_slot']
    assert scene.frame_current == expected['frame'] and abs(scene.frame_subframe - expected['subframe']) < 1e-6
    assert {key: arm.get(key) for key in LINKS} == expected['links']
    assert {a.name for a in bpy.data.actions} == set(expected['actions'])
    for name, state in expected['actions'].items():
        action = bpy.data.actions[name]
        assert digest(action) == state['sha256'], name
        assert action.use_fake_user == state['fake_user'], name
        assert action.get('arm_clearance_validation_status') == state['validation_status'], name
    report_text = bpy.data.texts[expected['report_name']]
    assert expected['report_fake_user'] and report_text.use_fake_user
    assert json.loads(report_text.as_string()) == expected['report']
    assert scene.arm_clearance.armature == arm
    if expected['case'] != 'early_cancel':
        review_name = arm.get('ac_review_action') or arm['ac_result_action']
        source = bpy.data.actions[expected['source']]
        arm.animation_data.action = source
        arm.animation_data.action_slot = next(s for s in source.slots if s.identifier == expected['slot_id'])
        assert bpy.ops.arm_clearance.compare() == {'FINISHED'}
        assert arm.animation_data.action.name == review_name
        assert arm.animation_data.action_slot.identifier == expected['slot_id']
    result = {'passed': True, 'case': expected['case'], 'blend': expected['blend'],
              'actions': len(expected['actions']), 'action_curves_links_slots_fake_users': 'identical after independent reopen',
              'report_text_fake_user': True}
    manifest_path.with_suffix('.reopened.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print('PERSISTENCE REOPEN PASS ' + expected['case'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('build', 'reopen'), default='build')
    parser.add_argument('--manifest', type=Path)
    options = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    OUT.mkdir(parents=True, exist_ok=True)
    if options.phase == 'reopen':
        reopen(options.manifest)
        return
    cases = ('success', 'surface', 'deep_proxy', 'cancel', 'exception', 'second_bake_error', 'early_cancel')
    manifests = []
    for case in cases:
        manifests.extend(build_case(case))
        print('PERSISTENCE BUILD PASS ' + case, flush=True)
    evidence = []
    for manifest in manifests:
        process = subprocess.run([bpy.app.binary_path, '-b', '--factory-startup', '--python-exit-code', '1',
                                  '--python', str(Path(__file__).resolve()), '--', '--phase', 'reopen',
                                  '--manifest', str(manifest)], capture_output=True, text=True,
                                 encoding='utf8', errors='replace', timeout=180)
        manifest.with_suffix('.process.log').write_text(process.stdout + '\n' + process.stderr, encoding='utf8')
        assert process.returncode == 0, (str(manifest), process.returncode, process.stdout[-4000:], process.stderr[-2000:])
        evidence.append(json.loads(manifest.with_suffix('.reopened.json').read_text(encoding='utf8')))
        print('PERSISTENCE INDEPENDENT REOPEN PASS ' + manifest.stem, flush=True)
    report = {'passed': True, 'version': list(package.bl_info['version']), 'cases': list(cases),
              'independent_reopens': len(evidence), 'evidence': evidence,
              'scope': 'Real rig/action baking, lifecycle, slots, comparison and serialization; synthetic collision/model substitutes; no user project touched'}
    (OUT / 'persistence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('PERSISTENCE V022 PASS ' + json.dumps({'cases': len(cases), 'independent_reopens': len(evidence)}), flush=True)


if __name__ == '__main__':
    main()
