"""Local, temporally smooth constraint fields from real-surface failures.

This module is NumPy only. It neither edits a Blender scene nor runs a solve.
The four returned arrays all have shape ``[model.n, 2, model.p]``. Weights
multiply their corresponding *squared collision energies*, hence the same
weights must multiply both positional force and orientation torque. The
``body_margin`` array is an ADDITION to the model's existing body clearance;
it must not be applied to opposite-limb or same-side hand/arm contacts.

In a pair contact use the maximum of its two endpoint weights, or the smooth
bounded union helper below. These fields are fixed throughout a solve. They
do not depend on the current corrected pose and introduce no missing spatial
derivatives. Final actual-surface validation is still required.
"""

import math
import numpy as np
from ...anatomy import classify_bones


def _temporal_bump(frames, center, radius, plateau,time_partition=None):
    """C2 bump, identically zero outside radius and one on the plateau."""
    u=np.clip((np.abs(frames-center)-plateau)/(radius-plateau),0.,1.)
    result=1.-u**3*(10.+u*(-15.+6.*u))
    if time_partition is not None and time_partition.cuts:
        result*=time_partition.repair_support(frames,center,radius)
    return result


def bounded_union(a,b,cap=8.,baseline=1.):
    """Smooth monotone union: >= both inputs and <= the given fixed cap.

    This avoids a timewise max of two smooth fields, whose crossing may have
    a derivative kink. ``cap`` is fixed, not estimated per frame. At baseline
    this is the identity, and an input equal to cap saturates the result.
    """
    if cap<=baseline:raise ValueError('cap must exceed baseline')
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    if np.any(a<baseline-1e-10) or np.any(b<baseline-1e-10) or np.any(a>cap+1e-10) or np.any(b>cap+1e-10):
        raise ValueError('Union input lies outside its baseline/cap')
    return cap-(cap-a)*(cap-b)/(cap-baseline)


def merge_surface_repair_fields(existing,added,world_scale=1.,weight_cap=8.):
    """Merge fixed fields without lowering prior constraints or C2 max kinks.

    Use the same ``world_scale`` as for building both fields. For an unmodified
    base solver pass its weights=1 and margin=0, or simply assign added fields.
    Repeated merging can strengthen overlapping windows up to the fixed cap;
    rebuilding from the accumulated reports avoids duplicate reinforcement.
    """
    if not math.isfinite(world_scale) or world_scale<=0:raise ValueError('world_scale must be positive')
    result={}
    for name in ('body_weights','mutual_weights','own_weights'):
        if np.shape(existing[name])!=np.shape(added[name]):raise ValueError('Repair field shapes differ')
        result[name]=bounded_union(existing[name],added[name],cap=weight_cap)
    if np.shape(existing['body_margin'])!=np.shape(added['body_margin']):raise ValueError('Repair margin shapes differ')
    result['body_margin']=bounded_union(existing['body_margin'],added['body_margin'],cap=.003/world_scale,baseline=0.)
    return result


def build_surface_repair_weights(surface_report,model,geometry,frames=None,attempt=1,
                                 world_scale=1.,half_window=24.,plateau=4.,diagnostics=None,time_partition=None):
    """Return ``(fields, evidence)`` from exact original-triangle face IDs.

    ``frames`` contains the actual scene frames aligned to model rows; supply
    it for a range starting elsewhere than one. If omitted, ``model.frames``
    is used when present, otherwise metadata frame_start (default 1) + arange.

    Attempt 1 uses 4x collision energy and 2 mm extra body clearance; attempt
    2 and later use 8x and 3 mm. Only reported categories, affected side and
    mapped local samples receive those changes. Millimetres are physical world
    distances and are divided by the armature's positive uniform world_scale.
    The center +/-4 frame plateau fades to zero at +/-24 using a quintic C2
    curve. Each unique (category-family,frame,side,sample) contributes once;
    repeated triangle pairs therefore do not accidentally amplify a window.

    Optional enriched ``diagnostics`` may contain ``frames`` records, each
    with a ``frame`` and ``pairs`` including face_a/b, part_a/b, and
    proxy_distance_at_actual_intersection_mm. A diagnosed positive field gap
    plus 0.5 mm replaces the default extra margin, capped at 3 mm. Unknown or
    negative gap values retain the default. No frame number is special-cased.
    """
    if not math.isfinite(world_scale) or world_scale<=0:raise ValueError('world_scale must be positive')
    if attempt<1:raise ValueError('attempt must be positive')
    if not 0<=plateau<half_window:raise ValueError('Require 0 <= plateau < half_window')
    n,p=int(model.n),int(model.p)
    if frames is None:
        frames=getattr(model,'frames',None)
        if frames is None:frames=np.arange(n,dtype=float)+float(model.meta.get('frame_start',1))
    frames=np.asarray(frames,np.float64)
    if frames.shape!=(n,) or not np.isfinite(frames).all() or np.any(np.diff(frames)<=0):
        raise ValueError('frames must be finite, increasing, and aligned with model.n')
    factor=4. if attempt==1 else 8.;default_margin_mm=2. if attempt==1 else 3.
    fields={name:np.ones((n,2,p),np.float64) for name in ('body_weights','mutual_weights','own_weights')}
    fields['body_margin']=np.zeros((n,2,p),np.float64)
    evidence={'attempt':int(attempt),'factor':factor,'default_extra_margin_world_mm':default_margin_mm,
              'world_scale':float(world_scale),'half_window_frames':float(half_window),
              'plateau_half_width_frames':float(plateau),'frame_range':[float(frames[0]),float(frames[-1])],
              'events':[],'unmapped':[],
              'scope':'Only actual failed category/side/sample; fixed C2 temporal fields; no hand/finger bone edits.'}
    if time_partition is not None and time_partition.cuts:
        evidence['source_partition']=time_partition.report(frames)
    triangles=np.asarray(geometry['triangles'],np.int32);vertices=np.asarray(geometry['vertices'],np.float64)
    dominant=np.asarray(geometry['dominant'],np.int32);rest=np.asarray(geometry['rest'],np.float64)
    classes=classify_bones(model.meta)
    arm_count=int(model.arm_count)
    arm_ids=classes['arm_ids'];wrist_ids=classes['wrist_ids']
    tu=np.asarray(model.tu,np.float64);tf=np.asarray(model.tf,np.float64)
    lookup=[]
    for side in range(2):
        lookup.append({tuple(sorted(map(int,t))):int(j) for t,j in
                       zip(model.hand.triangles[side],model.hand.triangle_spheres[side])})
    inverse_rest=None

    def arm_samples(face,side):
        ids=triangles[face];regions=classes['region'][dominant[ids]]
        sides=classes['side'][dominant[ids]];indices=[];detail=[]
        for segment,(a,b,ts,offset) in enumerate(((arm_ids[side*2],arm_ids[side*2+1],tu,0),
                                                (arm_ids[side*2+1],wrist_ids[side],tf,len(tu)))):
            if not np.any((regions==segment+1)&(sides==side)):continue
            axis=rest[b,:3,3]-rest[a,:3,3];denominator=float(axis@axis)
            if denominator<=1e-15:raise ValueError('Zero-length arm segment')
            t=(vertices[ids]-rest[a,:3,3])@axis/denominator
            edges=np.r_[-np.inf,(ts[:-1]+ts[1:])*.5,np.inf]
            chosen=np.flatnonzero((edges[:-1]<=t.max())&(edges[1:]>=t.min()))
            if not len(chosen):chosen=np.array([np.argmin(abs(ts-t.mean()))])
            indices.extend((chosen+offset).tolist())
            detail.append({'segment':'upper' if segment==0 else 'fore','rest_t_min':float(t.min()),
                           'rest_t_max':float(t.max()),'samples':(chosen+offset).tolist()})
        return sorted(set(indices)),{'method':'rest-segment Voronoi interval of complete triangle','segments':detail}

    def hand_samples(face,side,frame):
        nonlocal inverse_rest
        ids=triangles[face];key=tuple(sorted(map(int,ids)))
        exact=lookup[side].get(key)
        if exact is not None:return [arm_count+exact],{'method':'fixed full-triangle hand assignment'}
        # The real-surface checker keeps any-hand-vertex seam triangles, even
        # when calibration's >=2 hand-vertex selection has no exact triangle.
        # Map only its hand corner(s) to an anatomically compatible sample.
        hand_vertex=(classes['region'][dominant[ids]]==3)&(classes['side'][dominant[ids]]==side)
        ids=ids[hand_vertex]
        if not len(ids):return [],{'method':'no hand corner in reported hand face'}
        valid=np.asarray(model.hand.valid[side],bool)
        owners=np.asarray(model.hand.bone_ids[side])
        votes=(owners[:,None]==dominant[ids][None]).sum(1)
        allowed=valid&(votes==votes[valid].max())
        if inverse_rest is None:inverse_rest=np.linalg.inv(rest)
        fi=int(np.argmin(abs(frames-frame)))
        transform=np.asarray(model.matrices[fi])@inverse_rest
        point=np.zeros(3)
        joint=np.asarray(geometry['joint']);weight=np.asarray(geometry['weight'])
        for j in range(joint.shape[1]):
            m=transform[joint[ids,j]]
            skinned=np.einsum('vij,vj->vi',m[:,:3,:3],vertices[ids])+m[:,:3,3]
            point+=(skinned*weight[ids,j,None]).mean(0)
        distance=np.linalg.norm(model.hand.original_points[fi,side]-point,axis=-1)
        chosen=int(np.argmin(np.where(allowed,distance,np.inf)))
        return [arm_count+chosen],{'method':'boundary hand corner; source-pose closest matching anatomical owner',
                                   'hand_vertices':ids.tolist(),'owner_votes':int(votes[chosen])}

    def map_part(part,face,frame):
        if part=='body':return None
        if not isinstance(part,str) or not part.endswith(('.L','.R')):return None
        side=0 if part.endswith('.L') else 1
        face=int(face)
        if not 0<=face<len(triangles):raise ValueError('Surface report face index outside captured triangles')
        if part.startswith(('arm.','arm_body.','own_arm.')):samples,detail=arm_samples(face,side)
        elif part.startswith('hand.'):samples,detail=hand_samples(face,side,frame)
        else:return None
        return {'side':side,'samples':samples,'part':part,'face':face,'mapping':detail}

    diag={}
    for record in (diagnostics or {}).get('frames',[]):
        for pair in record.get('pairs',[]):
            key=(float(record['frame']),pair.get('part_a'),pair.get('part_b'),pair.get('face_a'),pair.get('face_b'))
            gap=pair.get('proxy_distance_at_actual_intersection_mm')
            if gap is not None and math.isfinite(float(gap)):diag[key]=float(gap)
    events={};margins={}
    families={'arm-body':'body','hand-body':'body','arm-arm':'mutual',
              'hand-opposite-arm':'mutual','hand-hand':'mutual','hand-own-arm':'own'}
    for record in surface_report.get('records',[]):
        frame=float(record['frame'])
        if not math.isfinite(frame):raise ValueError('Nonfinite failure frame')
        for category,data in record.get('categories',{}).items():
            family=families.get(category)
            if family is None:continue
            for pair in data.get('pairs',[]):
                mapped=[]
                for suffix in ('a','b'):
                    mapping=map_part(pair.get('part_'+suffix),pair.get('face_'+suffix,-1),frame)
                    if mapping is not None and mapping['samples']:mapped.append(mapping)
                if not mapped:
                    evidence['unmapped'].append({'frame':frame,'category':category,'pair':dict(pair)})
                    continue
                key=(frame,pair.get('part_a'),pair.get('part_b'),pair.get('face_a'),pair.get('face_b'))
                gap=diag.get(key,pair.get('proxy_distance_at_actual_intersection_mm'))
                margin_mm=default_margin_mm;margin_source='default actual-failure local margin'
                if gap is not None and math.isfinite(float(gap)) and float(gap)>0:
                    margin_mm=min(3.,float(gap)+.5);margin_source='positive actual-intersection field gap +0.5 mm, capped3'
                for mapping in mapped:
                    for sample in mapping['samples']:
                        event_key=(family,frame,mapping['side'],sample)
                        events[event_key]=1.
                        if family=='body':margins[event_key]=max(margins.get(event_key,0.),margin_mm/world_scale/1000.)
                evidence['events'].append({'frame':frame,'category':category,'family':family,
                    'face_a':pair.get('face_a'),'face_b':pair.get('face_b'),'mapped':mapped,
                    'extra_margin_world_mm':margin_mm if family=='body' else 0.,
                    'margin_source':margin_source if family=='body' else None})
            if data.get('containments') or data.get('proxies'):
                # No broad all-hand/all-arm fallback: missing anatomy must be
                # visible to the caller, which must retain its validation gate.
                evidence['unmapped'].append({'frame':frame,'category':category,
                    'reason':'Containment/proxy finding needs explicit original-face mapping',
                    'containments':data.get('containments',[]),'proxies':data.get('proxies',[])})
    for family in ('body','mutual','own'):
        residual=np.ones((n,2,p))
        last_frame=None;last_bump=None
        for f,frame,side,sample in events:
            if f==family:
                if frame!=last_frame:
                    last_frame=frame;last_bump=_temporal_bump(frames,frame,half_window,plateau,time_partition)
                residual[:,side,sample]*=1.-last_bump
        fields[family+'_weights']=1.+(factor-1.)*(1.-residual)
    if margins:
        cap=max(margins.values());residual=np.ones((n,2,p))
        last_frame=None;last_bump=None
        for (_,frame,side,sample),amplitude in margins.items():
            if frame!=last_frame:
                last_frame=frame;last_bump=_temporal_bump(frames,frame,half_window,plateau,time_partition)
            residual[:,side,sample]*=1.-last_bump*amplitude/cap
        fields['body_margin']=cap*(1.-residual)
    evidence['unique_sample_events']=len(events)
    evidence['field_summary']={name:{'min':float(v.min()),'max':float(v.max()),
        'modified_entries':int(np.count_nonzero(v>(0. if name=='body_margin' else 1.)+1e-12))}
        for name,v in fields.items()}
    evidence['complete_mapping']=not evidence['unmapped']
    return fields,evidence
