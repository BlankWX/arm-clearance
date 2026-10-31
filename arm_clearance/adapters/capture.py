"""Original Blender geometry/collider and integer-pose capture."""
import bpy
import bmesh
import numpy as np
from ..contracts import SourceClip

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
