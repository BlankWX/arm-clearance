bl_info = {
    'name':'Arm Clearance · 连续手臂与手部避碰',
    'author':'BlankWX',
    'version':(0,2,3),
    'blender':(4,4,0),
    'location':'3D 视图 > 侧栏 > 手臂避碰',
    'description':'整段动作求解双臂、手掌和手指避碰，优先保留大臂姿态，保留手指动画与原动作',
    'category':'Animation',
}

import bpy
import time
import traceback
from bpy.props import PointerProperty,StringProperty,FloatProperty,IntProperty,BoolProperty
from .pipeline_v2 import run_job, _slot_for


def _show_action(context, arm, action):
    ad = arm.animation_data
    identifier = action.get('arm_clearance_source_slot', '')
    if not identifier and ad.action_slot:
        identifier = ad.action_slot.identifier
    slot = _slot_for(action, identifier)
    ad.action = action
    ad.action_slot = slot
    scene = context.scene
    scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)
    suffix = '（复核未通过）' if action.get('arm_clearance_validation_status') == 'needs_review' else ''
    if action.get('arm_clearance_validation_status') == 'unverified':
        suffix = '（复核未完成）'
    scene.arm_clearance.status = '当前动作：' + action.name + suffix


def arm_poll(self,obj):return obj.type=='ARMATURE'
def mesh_poll(self,obj):return obj.type=='MESH'


class AC_Settings(bpy.types.PropertyGroup):
    armature:PointerProperty(name='角色骨架',type=bpy.types.Object,poll=arm_poll)
    body_mesh:PointerProperty(name='角色网格',type=bpy.types.Object,poll=mesh_poll)
    upper_left:StringProperty(name='左上臂',default='腕.L')
    fore_left:StringProperty(name='左前臂',default='ひじ.L')
    upper_right:StringProperty(name='右上臂',default='腕.R')
    fore_right:StringProperty(name='右前臂',default='ひじ.R')
    wrist_left:StringProperty(name='左手腕',default='手首.L',description='允许旋转和少量平移；其下手掌与手指参与碰撞，保留手指原动画')
    wrist_right:StringProperty(name='右手腕',default='手首.R')
    body_materials:StringProperty(name='检测材质',default='Body,絲襪,高跟鞋,颜',description='按材质筛选身体碰撞面的候选范围，身体与双臂按骨骼权重区分；英文逗号分隔；留空使用除双臂外的全部材质，请勿包含头发和饰物')
    margin:FloatProperty(name='安全间距',default=.001,min=0.,max=.03,subtype='DISTANCE',unit='LENGTH')
    upper_pose_weight:FloatProperty(name='大臂姿态保留',default=100.,min=1.,max=5000.,description='越高越倾向由前臂和手腕承担修正；仍以实际表面检测为准')
    wrist_translation_limit:FloatProperty(name='手腕平移软限制',default=.010,min=0.,max=.05,subtype='DISTANCE',unit='LENGTH',description='超出该距离增加惩罚；不是严格上限')
    collider_ratio:FloatProperty(name='身体网格保留率',default=.10,min=.03,max=.5)
    velocity_weight:FloatProperty(name='修正速度约束',default=10.,min=.01,max=1000.)
    acceleration_weight:FloatProperty(name='修正加速度约束',default=3000.,min=.01,max=100000.)
    iterations:IntProperty(name='优化迭代上限',default=600,min=50,max=3000)
    status:StringProperty(name='状态',default='选择骨架和网格，使用场景开始/结束帧')
    report_name:StringProperty()
    busy:BoolProperty(default=False,options={'SKIP_SAVE'})
    advanced:BoolProperty(name='骨骼与高级设置',default=False)


class AC_OT_autofill(bpy.types.Operator):
    bl_idname='arm_clearance.autofill';bl_label='识别当前角色'
    def execute(self,context):
        s=context.scene.arm_clearance
        arms=[o for o in context.scene.objects if o.type=='ARMATURE']
        active=context.active_object
        arm=active if active and active.type=='ARMATURE' else next((o for o in arms if s.upper_left in o.pose.bones),None)
        if not arm:self.report({'ERROR'},'未找到含指定手臂骨骼的骨架');return {'CANCELLED'}
        meshes=[o for o in context.scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' and m.object==arm for m in o.modifiers)]
        s.armature=arm
        s.body_mesh=max(meshes,key=lambda o:len(o.data.vertices)) if meshes else None
        s.status=f'场景范围：{context.scene.frame_start}–{context.scene.frame_end} 帧'
        return {'FINISHED'}


class AC_OT_bake(bpy.types.Operator):
    bl_idname='arm_clearance.bake';bl_label='分析并烘焙场景范围';bl_options={'REGISTER'}
    _timer=None;_job=None
    @classmethod
    def poll(cls,context):return not context.scene.arm_clearance.busy
    def cleanup(self,context):
        if self._job is not None:self._job.close();self._job=None
        if self._timer is not None:context.window_manager.event_timer_remove(self._timer);self._timer=None
        context.scene.arm_clearance.busy=False
        for area in context.screen.areas:area.tag_redraw()
    def execute(self,context):
        if context.screen.is_animation_playing:bpy.ops.screen.animation_cancel(restore_frame=False)
        s=context.scene.arm_clearance;s.busy=True
        def status(message):s.status=message
        self._job=run_job(context.scene,s,status)
        self._timer=context.window_manager.event_timer_add(.1,window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    def modal(self,context,event):
        if event.type=='ESC':
            self.cleanup(context)
            return {'CANCELLED'}
        if event.type!='TIMER':return {'PASS_THROUGH'}
        deadline=time.perf_counter()+.075
        try:
            while time.perf_counter()<deadline:next(self._job)
        except StopIteration as finished:
            result = finished.value
            self.cleanup(context)
            if result is None:
                self.report({'ERROR'}, context.scene.arm_clearance.status)
                return {'CANCELLED'}
            level = 'INFO' if result.get('arm_clearance_validation_status') == 'passed' else 'WARNING'
            self.report({level}, context.scene.arm_clearance.status)
            return {'FINISHED'}
        except Exception as exc:
            traceback.print_exc();self.cleanup(context)
            self.report({'ERROR'},context.scene.arm_clearance.status)
            return {'CANCELLED'}
        for area in context.screen.areas:area.tag_redraw()
        return {'RUNNING_MODAL'}
    def cancel(self,context):self.cleanup(context)


class AC_OT_compare(bpy.types.Operator):
    bl_idname='arm_clearance.compare';bl_label='切换原动作 / 烘焙动作'
    @classmethod
    def poll(cls,context):return not context.scene.arm_clearance.busy
    def execute(self,context):
        arm=context.scene.arm_clearance.armature
        if not arm or not arm.animation_data:return {'CANCELLED'}
        original=bpy.data.actions.get(arm.get('ac_source_action',''))
        result=bpy.data.actions.get(arm.get('ac_review_action',''))
        result=result or bpy.data.actions.get(arm.get('ac_result_action',''))
        if not original or not result:
            self.report({'WARNING'},'尚无可对比的烘焙动作');return {'CANCELLED'}
        _show_action(context, arm, original if arm.animation_data.action==result else result)
        return {'FINISHED'}


class AC_OT_previous(bpy.types.Operator):
    bl_idname='arm_clearance.previous';bl_label='查看上一版结果'
    @classmethod
    def poll(cls,context):return not context.scene.arm_clearance.busy
    def execute(self,context):
        arm=context.scene.arm_clearance.armature
        action=bpy.data.actions.get(arm.get('ac_previous_result_action','')) if arm else None
        if not action:
            self.report({'WARNING'},'尚无上一版结果');return {'CANCELLED'}
        _show_action(context, arm, action)
        return {'FINISHED'}


class AC_PT_panel(bpy.types.Panel):
    bl_label='连续手臂避碰';bl_idname='AC_PT_panel';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='手臂避碰'
    def draw(self,context):
        s=context.scene.arm_clearance;l=self.layout
        col=l.column();col.enabled=not s.busy
        col.operator('arm_clearance.autofill',icon='EYEDROPPER')
        col.prop(s,'armature');col.prop(s,'body_mesh');col.prop(s,'body_materials')
        col.label(text=f'工作范围：{context.scene.frame_start}–{context.scene.frame_end} 帧',icon='TIME')
        col.prop(s,'margin');col.prop(s,'upper_pose_weight');col.prop(s,'advanced',icon='PREFERENCES')
        if s.advanced:
            box=col.box()
            for prop in ('upper_left','fore_left','upper_right','fore_right','wrist_left','wrist_right','wrist_translation_limit','velocity_weight','acceleration_weight','iterations','collider_ratio'):box.prop(s,prop)
        col.operator('arm_clearance.bake',icon='ACTION')
        col.operator('arm_clearance.compare',icon='ARROW_LEFTRIGHT')
        arm=s.armature
        if arm and arm.get('ac_previous_result_action'):col.operator('arm_clearance.previous',icon='PREV_KEYFRAME')
        review=bpy.data.actions.get(arm.get('ac_review_action','')) if arm else None
        if review:
            box=l.box()
            box.label(text='烘焙已保留，复核未通过' if review.get('arm_clearance_validation_status') == 'needs_review'
                           else '烘焙已保留，复核未完成',icon='ERROR')
            box.label(text='可切换查看；保存工程即可保留数据')
        l.separator();l.label(text=s.status[:60],icon='INFO')
        if s.busy:l.label(text='Esc 可取消；优化期间请勿编辑骨架')
        l.label(text='手掌、手指参与检测；保留手指原动画')
        l.label(text='仅修改双臂与手腕；烘焙后正常空格播放')
        if s.report_name:l.label(text='详细指标见文本编辑器中的报告')


classes=(AC_Settings,AC_OT_autofill,AC_OT_bake,AC_OT_compare,AC_OT_previous,AC_PT_panel)
def register():
    for cls in classes:bpy.utils.register_class(cls)
    bpy.types.Scene.arm_clearance=PointerProperty(type=AC_Settings)
def unregister():
    if hasattr(bpy.types.Scene,'arm_clearance'):del bpy.types.Scene.arm_clearance
    for cls in reversed(classes):bpy.utils.unregister_class(cls)

if __name__=='__main__':register()
