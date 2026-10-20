"""Scene capture and reversible baking. No frame-change/playback handlers."""
import json
import time
import bpy
import bmesh
import numpy as np
from .solver import ClipModel, ClipSolver, estimate_radii, corrected_quaternions, metrics
from mathutils.bvhtree import BVHTree
from mathutils import Vector


def channels(action, slot):
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type == 'KEYFRAME':
                bag = strip.channelbag(slot)
                if bag is not None:
                    yield from bag.fcurves


def curve_boundary(fc,time):
    keys=list(fc.keyframe_points)
    if any(abs(k.co.x-time)<1e-5 for k in keys):return None
    left=next((k for k in reversed(keys) if k.co.x<time),None)
    right=next((k for k in keys if k.co.x>time),None)
    result={'time':float(time),'value':fc.evaluate(time),'interpolation':'LINEAR'}
    if left is None or right is None:return result
    result['interpolation']=left.interpolation
    if left.interpolation=='BEZIER':
        points=np.array([left.co[:],left.handle_right[:],right.handle_left[:],right.co[:]])
        # De Casteljau subdivision preserves the source curve on the side
        # outside the bake interval, including its original timing handles.
        lo,hi=0.,1.
        for _ in range(55):
            t=(lo+hi)/2;s=1-t
            x=s**3*points[0,0]+3*s*s*t*points[1,0]+3*s*t*t*points[2,0]+t**3*points[3,0]
            if x<time:lo=t
            else:hi=t
        t=(lo+hi)/2;s=1-t
        a,b,c=s*points[:-1]+t*points[1:]
        d,e=s*np.array([a,b])+t*np.array([b,c])
        value=s*d+t*e
        result.update(value=float(value[1]),left_handle=d.tolist(),right_handle=e.tolist(),
                      previous=float(left.co.x),next=float(right.co.x),previous_right=a.tolist(),next_left=c.tolist())
    elif left.interpolation not in ('LINEAR','CONSTANT'):
        raise ValueError('范围边界的旋转曲线需为贝塞尔、线性或常量插值')
    return result


def validate(settings):
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


def geometry_data(arm,mesh,targets,materials):
    names=[b.name for b in arm.data.bones];index={n:i for i,n in enumerate(names)}
    parents=[index[b.parent.name] if b.parent else -1 for b in arm.data.bones]
    coords=np.empty(len(mesh.data.vertices)*3,np.float32);mesh.data.vertices.foreach_get('co',coords)
    coords=coords.reshape(-1,3)
    # All distances are measured in armature coordinates, including objects
    # with different origins/transforms.
    relative=arm.matrix_world.inverted()@mesh.matrix_world
    coords=coords@np.asarray(relative.to_3x3()).T+np.asarray(relative.translation)
    joint=np.zeros((len(coords),4),np.int32);weight=np.zeros((len(coords),4),np.float32)
    dominant=np.zeros(len(coords),np.int32);groups=[g.name for g in mesh.vertex_groups]
    for v in mesh.data.vertices:
        ws=sorted([(g.weight,index[groups[g.group]]) for g in v.groups if groups[g.group] in index],reverse=True)
        if len(ws)>4 and sum(w for w,j in ws[4:])>.001:
            raise ValueError('网格有超过四个显著骨骼权重；当前版本不支持截断这些权重')
        if ws:dominant[v.index]=ws[0][1]
        for k,(w,j) in enumerate(ws[:4]):weight[v.index,k]=w;joint[v.index,k]=j
    weight/=np.maximum(weight.sum(1,keepdims=True),1e-10)
    mesh.data.calc_loop_triangles()
    triangles=np.array([t.vertices[:] for t in mesh.data.loop_triangles],np.int32)
    mat=np.array([t.material_index for t in mesh.data.loop_triangles])
    allowed={s.strip() for s in materials.split(',') if s.strip()}
    if allowed:
        mat_ids=[i for i,m in enumerate(mesh.data.materials) if m and m.name in allowed]
        if not mat_ids:raise ValueError('检测材质名称未匹配网格，请填写材质名称，使用英文逗号分隔')
        mask=np.isin(mat,mat_ids)
    else:mask=np.ones(len(triangles),bool)
    excluded=set()
    roots={targets[0],targets[2]}|{n for n in ('肩P.L','肩P.R') if n in index}
    for b in arm.data.bones:
        if b.name in roots or any(p.name in roots for p in b.parent_recursive):excluded.add(index[b.name])
    mask&=~np.isin(dominant[triangles],list(excluded)).any(1)
    body=triangles[mask]
    if len(body)<50:raise ValueError('提取的身体碰撞面过少，请检查网格权重和材质选择')
    rest=np.array([b.matrix_local for b in arm.data.bones],np.float64)
    geo={'vertices':coords,'joint':joint,'weight':weight,'dominant':dominant,'rest':rest,'triangles':triangles}
    meta={'bones':names,'parents':parents,'targets':targets,'target_ids':[index[n] for n in targets],
          'wrist_ids':[index['手首.L'],index['手首.R']] if '手首.L' in index and '手首.R' in index else []}
    return geo,meta,body


def closed_collider(geo,meta,body,ratio):
    ids=np.unique(body);mapping=np.full(len(geo['vertices']),-1);mapping[ids]=np.arange(len(ids))
    me=bpy.data.meshes.new('_ArmClearance_Collider')
    me.from_pydata(geo['vertices'][ids].tolist(),[],mapping[body].tolist());me.update()
    ob=bpy.data.objects.new('_ArmClearance_Collider',me);bpy.context.scene.collection.objects.link(ob)
    try:
        ob.hide_render=True
        for name in meta['bones']:ob.vertex_groups.new(name=name)
        for i,old in enumerate(ids):
            for j,w in zip(geo['joint'][old],geo['weight'][old]):
                if w>0:ob.vertex_groups[int(j)].add([i],float(w),'REPLACE')
        bm=bmesh.new()
        try:
            bm.from_mesh(me)
            height=float(np.ptp(geo['vertices'][:,2]))
            bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=height*1.8e-5)
            bmesh.ops.holes_fill(bm,edges=[e for e in bm.edges if e.is_boundary],sides=0)
            bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
            bm.to_mesh(me)
        finally:bm.free()
        me.update()
        mod=ob.modifiers.new('Collision mesh','DECIMATE');mod.ratio=ratio;mod.use_collapse_triangulate=True
        ev=ob.evaluated_get(bpy.context.evaluated_depsgraph_get());em=ev.to_mesh()
        try:
            vertices=np.array([v.co[:] for v in em.vertices],np.float32)
            joints=np.zeros((len(vertices),4),np.int32);weights=np.zeros((len(vertices),4),np.float32)
            for v in em.vertices:
                for k,w in enumerate(sorted(v.groups,key=lambda w:w.weight,reverse=True)[:4]):joints[v.index,k]=w.group;weights[v.index,k]=w.weight
            weights/=np.maximum(weights.sum(1,keepdims=True),1e-10)
            em.calc_loop_triangles();tris=np.array([t.vertices[:] for t in em.loop_triangles],np.int32)
            return {'vertices':vertices,'joint':joints,'weight':weights,'triangles':tris,'rest':geo['rest']}
        finally:ev.to_mesh_clear()
    finally:
        bpy.data.objects.remove(ob,do_unlink=True);bpy.data.meshes.remove(me)


def bake_action(arm,source,targets,frames,quaternions,name):
    source.use_fake_user=True
    action=source.copy();action.name=name
    ad=arm.animation_data;ad.action=action
    slot=ad.action_slot
    try:
        fcurves=list(channels(action,slot))
        bag=action.layers[0].strips[0].channelbag(slot,ensure=True)
        for j,bone in enumerate(targets):
            path=arm.pose.bones[bone].path_from_id('rotation_quaternion')
            for axis in range(4):
                fc=next((f for f in fcurves if f.data_path==path and f.array_index==axis),None)
                if fc is None:fc=bag.fcurves.new(path,index=axis)
                outside=[{'time':float(k.co.x),'co':k.co[:],'left':k.handle_left[:],'right':k.handle_right[:],
                          'interpolation':k.interpolation} for k in fc.keyframe_points if not frames[0]<=k.co.x<=frames[-1]]
                guards=[curve_boundary(fc,float(frames[0]-1)),curve_boundary(fc,float(frames[-1]+1))]
                for k in list(fc.keyframe_points)[::-1]:
                    if frames[0]<=k.co.x<=frames[-1]:fc.keyframe_points.remove(k,fast=True)
                for f,q in zip(frames,quaternions[:,j,axis]):
                    k=fc.keyframe_points.insert(float(f),float(q),options={'FAST'});k.interpolation='LINEAR'
                for guard in guards:
                    if guard is not None:fc.keyframe_points.insert(guard['time'],guard['value'],options={'FAST'})
                fc.update()
                by_time={round(float(k.co.x),5):k for k in fc.keyframe_points}
                for old in outside:
                    k=by_time[round(old['time'],5)]
                    k.handle_left_type='FREE';k.handle_right_type='FREE'
                    k.handle_left=old['left'];k.handle_right=old['right'];k.interpolation=old['interpolation']
                for side,guard in enumerate(guards):
                    if guard is None:continue
                    k=by_time[round(guard['time'],5)]
                    k.handle_left_type='FREE';k.handle_right_type='FREE'
                    if side==0:
                        k.interpolation='LINEAR'
                        if 'left_handle' in guard:
                            k.handle_left=guard['left_handle']
                            by_time[round(guard['previous'],5)].handle_right=guard['previous_right']
                    else:
                        k.interpolation=guard['interpolation']
                        if 'right_handle' in guard:
                            k.handle_right=guard['right_handle']
                            by_time[round(guard['next'],5)].handle_left=guard['next_left']
        action['arm_clearance_source']=source.name
        action['arm_clearance_range']=[int(frames[0]),int(frames[-1])]
        arm['ac_source_action']=source.name;arm['ac_result_action']=action.name
        return action
    except Exception:
        ad.action=source;bpy.data.actions.remove(action);raise


def surface_check(scene,arm,mesh,geo,meta,body,start,end,status):
    """Verify the baked, evaluated skin at integer and half frames."""
    names=meta['bones'];parents=meta['parents'];rest=geo['rest'];v=geo['vertices']
    classes=np.full(len(names),-1)
    for b in range(len(names)):
        for side in range(2):
            upper,fore=meta['target_ids'][side*2:side*2+2];wrist=meta['wrist_ids'][side]
            p=b
            while p>=0:
                if p==wrist:break
                if p==fore:classes[b]=side*2+1;break
                if p==upper:classes[b]=side*2;break
                p=parents[p]
    arm_indices=[];arm_triangles=[]
    for side in range(2):
        upper,fore=meta['target_ids'][side*2:side*2+2];wrist=meta['wrist_ids'][side]
        mask=np.zeros(len(v),bool)
        for segment,(a,b) in enumerate(((upper,fore),(fore,wrist))):
            h=rest[a,:3,3];u=rest[b,:3,3]-h;t=(v-h)@u/(u@u)
            mask|=(classes[geo['dominant']]==side*2+segment)&(t>(.34 if segment==0 else 0.))&(t<(.995 if segment==1 else 1.1))
        arm_indices.append(np.flatnonzero(mask))
        arm_triangles.append(geo['triangles'][mask[geo['triangles']].all(1)].tolist())
    body_ids=np.unique(body);mapping=np.full(len(v),-1);mapping[body_ids]=np.arange(len(body_ids))
    body_triangles=mapping[body].tolist()
    result={'samples':0,'body_intersection_samples':0,'mutual_intersection_samples':0,
            'near_surface_depth_max_mm':0.,'problem_frames':[],
            'scope':'原始分辨率蒙皮；整数和半帧；排除手掌、手指、肩部连接区；不构成所有连续时刻的数学证明'}
    for i,f in enumerate(np.arange(start,end+.1,.5)):
        scene.frame_set(int(f),subframe=float(f%1))
        ev=mesh.evaluated_get(bpy.context.evaluated_depsgraph_get());em=ev.to_mesh()
        try:
            if len(em.vertices)!=len(v):raise ValueError('表面验证需要保持顶点顺序的网格修改器')
            coords=np.empty(len(v)*3,np.float32);em.vertices.foreach_get('co',coords);coords=coords.reshape(-1,3)
        finally:ev.to_mesh_clear()
        tree=BVHTree.FromPolygons(coords[body_ids].tolist(),body_triangles,all_triangles=True)
        full=coords.tolist()
        arms=[BVHTree.FromPolygons(full,tri,all_triangles=True) for tri in arm_triangles]
        body_hit=any(tree.overlap(a) for a in arms);mutual_hit=bool(arms[0].overlap(arms[1]))
        result['samples']+=1;result['body_intersection_samples']+=int(body_hit);result['mutual_intersection_samples']+=int(mutual_hit)
        if body_hit or mutual_hit:result['problem_frames'].append(float(f))
        for ids in arm_indices:
            for index in ids[::3]:
                point=Vector(coords[index]);co,n,face,distance=tree.find_nearest(point)
                if distance is not None and distance<.03 and (point-co).dot(n)<-1e-7:
                    result['near_surface_depth_max_mm']=max(result['near_surface_depth_max_mm'],distance*1000)
        if i%10==0:
            status(f'高精度表面复核 {f:g}/{end}');yield result
    yield result


def run_job(scene,settings,status):
    arm,mesh,targets,wrists=validate(settings)
    ad=arm.animation_data
    source=bpy.data.actions.get(arm.get('ac_source_action','')) if ad.action.get('arm_clearance_source') else ad.action
    source=source or ad.action
    previous=ad.action;original_frame=scene.frame_current
    original_subframe=scene.frame_subframe
    previous_links={k:arm.get(k) for k in ('ac_source_action','ac_result_action')}
    start,end=scene.frame_start,scene.frame_end
    if end-start<2:raise ValueError('至少需要三帧才能约束连续性')
    complete=False
    try:
        ad.action=source
        status('提取身体和双臂权重');yield None
        geo,meta,body=geometry_data(arm,mesh,targets,settings.body_materials)
        meta['wrist_ids']=[meta['bones'].index(n) for n in wrists]
        status('封闭肩部开口并简化碰撞面');yield None
        collider=closed_collider(geo,meta,body,settings.collider_ratio)
        frames=np.arange(start,end+1)
        matrices=np.empty((len(frames),len(meta['bones']),4,4),np.float32)
        quats=np.empty((len(frames),4,4),np.float32)
        for i,f in enumerate(frames):
            scene.frame_set(int(f));matrices[i]=[p.matrix for p in arm.pose.bones]
            quats[i]=[arm.pose.bones[n].rotation_quaternion[:] for n in targets]
            if i%10==0:status(f'采样原动作 {i+1}/{len(frames)}');yield None
        radii=estimate_radii(geo,meta)
        # Margin UI is in metres regardless of the model's overall scale.
        scale=float(np.mean(arm.matrix_world.to_scale()))
        model=ClipModel(matrices,meta['target_ids'],meta['wrist_ids'],radii,settings.margin/max(scale,1e-8))
        def progress(stage,i,n,*args):status(f'{"构建碰撞缓存" if stage=="colliders" else "全段连续优化"} {i}/{n}')
        solver=ClipSolver(model,collider,matrices,progress)
        solver.velocity=settings.velocity_weight;solver.acceleration=settings.acceleration_weight
        for _ in solver.prepare():yield None
        baseline=metrics(model,solver,np.zeros((len(frames),4,3)))
        for _ in solver.optimize(settings.iterations):yield None
        report={'range':[int(start),int(end)],'baseline':baseline,'corrected':metrics(model,solver,solver.result),
                'scope':'封闭简化身体网格与变半径手臂采样；不包含手掌、手指；不等同于完整表面认证'}
        name='ArmClearance_Report_'+time.strftime('%Y%m%d_%H%M%S')
        text=bpy.data.texts.new(name);text.write(json.dumps(report,ensure_ascii=False,indent=2))
        settings.report_name=text.name
        if report['corrected']['frames_over_3mm']:
            settings.status='仍有明显碰撞，已保留原动作。请查看报告并调整参数。'
            return
        status('写入四根骨骼关键帧');yield None
        result=bake_action(arm,source,targets,frames,corrected_quaternions(quats,solver.result),'AC_'+source.name)
        result['arm_clearance_report']=text.name
        for surface in surface_check(scene,arm,mesh,geo,meta,body,start,end,status):yield None
        report['surface_verification']=surface
        text.clear();text.write(json.dumps(report,ensure_ascii=False,indent=2))
        if surface['body_intersection_samples'] or surface['mutual_intersection_samples']:
            settings.status='表面复核仍有交叉，已恢复先前动作；详细帧号见报告。'
            result.name+='_未通过复核'
            return
        settings.status=f'完成 {start}–{end} 帧，可切换原动作对比；报告：{text.name}'
        complete=True
    finally:
        if not complete:
            ad.action=previous
            for k,v in previous_links.items():
                if v is None:
                    if k in arm:del arm[k]
                else:arm[k]=v
        scene.frame_set(original_frame,subframe=original_subframe)
