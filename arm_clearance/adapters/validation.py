"""Existing scene checks, unchanged in order and rejection policy."""
import numpy as np
from .action_io import channels

def legacy_validate(settings):
    arm, mesh = settings.armature, settings.body_mesh
    if not arm or arm.type != 'ARMATURE': raise ValueError('请选择角色骨架')
    if not mesh or mesh.type != 'MESH': raise ValueError('请选择角色网格')
    scale=np.array(arm.matrix_world.to_scale())
    if scale.min()<=0 or np.ptp(scale)>1e-5:
        raise ValueError('骨架对象需要正的均匀缩放；请先应用非均匀缩放')
    targets = [settings.upper_left, settings.fore_left, settings.upper_right, settings.fore_right]
    wrists = [settings.wrist_left, settings.wrist_right]
    if len(set(targets)) != 4: raise ValueError('四根修正骨骼不能重复')
    for name in targets+wrists:
        if name not in arm.pose.bones: raise ValueError('缺少骨骼：'+name)
    for name in targets:
        pb=arm.pose.bones[name]
        if pb.rotation_mode != 'QUATERNION': raise ValueError(name+' 需要使用四元数旋转模式')
        if any(not c.mute and c.influence>0 for c in pb.constraints):
            raise ValueError(name+' 有活动约束，请先烘焙该骨骼的约束')
    for upper,fore,wrist in ((targets[0],targets[1],wrists[0]),(targets[2],targets[3],wrists[1])):
        if arm.data.bones[upper] not in arm.data.bones[fore].parent_recursive:
            raise ValueError(f'{fore} 必须位于 {upper} 的下级')
        if arm.data.bones[fore] not in arm.data.bones[wrist].parent_recursive:
            raise ValueError(f'{wrist} 必须位于 {fore} 的下级')
    ad=arm.animation_data
    if not ad or not ad.action: raise ValueError('骨架没有活动动作')
    if any(not t.mute and len(t.strips) for t in ad.nla_tracks):
        raise ValueError('当前版本需要单个活动动作；请先将 NLA 合成为一个动作')
    if any(any(f'pose.bones["{n}"]' in d.data_path for n in targets) for d in ad.drivers):
        raise ValueError('修正骨骼存在驱动，请先烘焙驱动')
    paths={arm.pose.bones[n].path_from_id('rotation_quaternion') for n in targets}
    for fc in channels(ad.action,ad.action_slot):
        if fc.data_path in paths and any(not m.mute for m in fc.modifiers):
            raise ValueError('修正骨骼的旋转曲线有修改器，请先烘焙曲线修改器')
        if fc.data_path in paths and len(fc.sampled_points):
            raise ValueError('修正骨骼的曲线使用采样点，请先转换为关键帧')
    modifiers=[m for m in mesh.modifiers if m.type=='ARMATURE' and m.show_viewport]
    if len(modifiers)!=1 or modifiers[0].object!=arm:
        raise ValueError('角色网格需要一个指向所选骨架的活动骨架修改器')
    if modifiers[0].use_deform_preserve_volume:
        raise ValueError('当前碰撞采样使用线性蒙皮，请先关闭骨架修改器的保留体积')
    if any(m.show_viewport and m.type not in ('ARMATURE','WEIGHTED_NORMAL','NORMAL_EDIT') for m in mesh.modifiers):
        raise ValueError('请先应用或禁用骨架以外的几何修改器，保证表面验证的顶点顺序')
    return arm,mesh,targets,wrists

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
