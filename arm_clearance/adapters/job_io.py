"""Blender job I/O: capture, opaque Action handles and original-state restoration."""
import bpy
import numpy as np
from ..contracts import SourceClip
from .validation import validate
from .capture import geometry_data, closed_collider

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

def _source_for(arm, active):
    source_name = active.get('arm_clearance_source', '')
    if source_name:
        source = bpy.data.actions.get(source_name)
        source = source or bpy.data.actions.get(arm.get('ac_source_action', ''))
        if source is None:
            raise ValueError('修正动作记录的原动作已不存在，请先选择有效的原动作')
        return source
    return active


def _write_report(text, report):
    """Compatibility port: release builds do not persist report data."""
    pass


class BlenderJobIO:
    def __init__(self, scene, settings):
        self.scene=scene;self.settings=settings
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
        self.arm = arm
        self.mesh = mesh
        self.targets = targets
        self.wrists = wrists
        self.arm_targets = arm_targets
        self.ad = ad
        self.previous = previous
        self.previous_slot_id = previous_slot_id
        self.source = source
        self.source_slot = source_slot
        self.previous_links = previous_links
        self.fake_users = fake_users
        self.original_rotations = original_rotations
        self.original_locations = original_locations
        self.original_frame = original_frame
        self.original_subframe = original_subframe
        self.start = start
        self.end = end

    def activate_source(self):
        self.ad.action=self.source
        self.ad.action_slot=self.source_slot
        validate(self.settings)

    def new_report(self):
        # Reports are internal job state only; do not create saved Text IDs.
        return None

    def publish_report(self, text):
        self.settings.report_name=''

    def write_report(self, text, report):
        # Keep the persistence port inert without touching validation data.
        _write_report(text, report)

    def restore_links(self):
        _restore_links(self.arm,self.previous_links)

    def find_action(self, name):
        return bpy.data.actions.get(name)

    def discard_action(self, action):
        bpy.data.actions.remove(action)

    def prepare_retry_source(self, clip):
        self.ad.action=self.source
        self.ad.action_slot=self.source_slot
        for j,name in enumerate(self.targets):
            self.arm.pose.bones[name].rotation_quaternion=clip.quaternions[-1,j]
        for side,name in enumerate(self.wrists):
            self.arm.pose.bones[name].location=clip.locations[-1,side]
        self.scene.frame_set(self.end)

    def restore_original(self):
        self.ad.action=self.previous
        self.ad.action_slot=_slot_for(self.previous,self.previous_slot_id)
        self.restore_links()
        for action,was_fake in self.fake_users.items():
            action.use_fake_user=was_fake
        for name,value in self.original_rotations.items():
            self.arm.pose.bones[name].rotation_quaternion=value
        for name,value in self.original_locations.items():
            self.arm.pose.bones[name].location=value

    def restore_frame(self):
        self.scene.frame_set(self.original_frame,subframe=self.original_subframe)

    def capture_source(self, status, report):
        scene=self.scene;settings=self.settings
        arm=self.arm;mesh=self.mesh;arm_targets=self.arm_targets;targets=self.targets;wrists=self.wrists
        start=self.start;end=self.end;source=self.source
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
        status('采样原动作六骨与双腕位置 0/%d' % len(frames))
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
        return SourceClip(frames,matrices,quaternions,locations,location_bases,geo,meta,body,collider,scale)

    slot_for = staticmethod(_slot_for)
