"""Repair a completed, deep first bake in bounded source-relative windows."""
from dataclasses import replace
import math

import numpy as np
from mathutils import Quaternion

from ..composition import make_problem, partition_source
from ..features.collision.hand_geometry import build_hand_geometry
from ..features.repair.half_search import contact_sides
from ..features.repair.policy import RepairPolicy, SurfaceQuality
from .action_io import channels
from .surface import prepare_surface, surface_check


def windows(frames, failures, partition):
    """Bound the solve span and keep local changes inside their source segment."""
    for lo, hi in partition.blocks:
        start, end = int(frames[lo]), int(frames[hi-1])
        times = sorted(t for t in failures if start <= t <= end)
        groups = []
        for t in times:
            if groups and t-groups[-1][1] <= 9 and t-groups[-1][0] < 96:
                groups[-1][1] = t
            else:
                groups.append([t,t])
        for left,right in groups:
            yield max(start,math.floor(left)-9), min(end,math.ceil(right)+9)


class WindowKeys:
    def __init__(self, io, action, frames):
        slot = io.slot_for(action,io.source_slot.identifier)
        by_path = {(c.data_path,c.array_index):c for c in channels(action,slot)}
        self.action = action
        self.components = []
        for name in io.targets:
            path = io.arm.pose.bones[name].path_from_id('rotation_quaternion')
            group = []
            for axis in range(4):
                curve = by_path[path,axis]
                points = curve.keyframe_points
                left,right = 0,len(points)
                while left < right:
                    mid = (left+right)//2
                    if points[mid].co.x < frames[0]:
                        left = mid+1
                    else:
                        right = mid
                indices = list(range(left,left+len(frames)))
                if any(points[k].co.x != f for k,f in zip(indices,frames)):
                    raise ValueError('局部补修缺少首轮烘焙的连续整数键')
                group.append((curve,indices))
            self.components.append(group)
        self.reference = np.array([[[c.keyframe_points[k].co.y for k in indices]
                                    for c,indices in group] for group in self.components]).transpose(2,0,1)

    def write(self, quats):
        quats = np.where((quats*self.reference).sum(-1,keepdims=True)<0,-quats,quats)
        for bone,group in enumerate(self.components):
            for axis,(curve,indices) in enumerate(group):
                for index,value in zip(indices,quats[:,bone,axis]):
                    curve.keyframe_points[index].co.y = value
                curve.update()
        self.action.update_tag()


def verify(io, clip, topology, start, end):
    for surface in surface_check(io.scene,io.arm,io.mesh,clip.geometry,clip.meta,clip.body,
                                 start,end,topology=topology):
        yield None
    if not surface['complete']:
        raise RuntimeError('局部真实网格复核未完成')
    return surface


def swivel(io, clip, model, delta, side, angles):
    result = delta.copy()
    names = [io.targets[3*side+j] for j in (0,2)]
    upper,wrist = [io.arm.pose.bones[name] for name in names]
    quats = model.quats(delta)
    for i,(frame,angle) in enumerate(zip(clip.frames,angles)):
        if not angle:
            continue
        io.scene.frame_set(int(frame))
        axis = (wrist.head-upper.head).normalized()
        for j,bone,sign in ((side*3,upper,1),(side*3+2,wrist,-1)):
            local = bone.matrix.to_quaternion().inverted() @ axis
            rotation = Quaternion(quats[i,j]) @ Quaternion(local,sign*angle)
            q = Quaternion(clip.quaternions[i,j]).normalized().rotation_difference(rotation)
            if q.w < 0:
                q.negate()
            result[i,j] = q.to_exponential_map()
        yield None
    return result


def recover(io, clip, action, delta, surface, deep_frames, topology, stationary, status):
    result = delta.copy()
    failures = set(surface['problem_frames']) | set(deep_frames)
    jobs = list(windows(clip.frames,failures,clip.time_partition))
    for number,(start,end) in enumerate(jobs,1):
        status('首轮超限自动补修 %d/%d：%d–%d 帧' % (number,len(jobs),start,end))
        yield None
        if end-start < 2:
            continue
        sl = slice(start-int(clip.frames[0]),end-int(clip.frames[0])+1)
        local = partition_source(replace(clip,frames=clip.frames[sl],matrices=clip.matrices[sl],
                                         quaternions=clip.quaternions[sl],locations=clip.locations[sl],
                                         location_bases=clip.location_bases[sl],time_partition=None))
        hand = build_hand_geometry(local.geometry,local.meta,local.matrices,time_partition=local.time_partition)
        problem = make_problem(local,hand,io.settings.margin,lambda *args:None,surface_check)
        solver,model = problem.solver,problem.model
        solver.configure(io.settings.velocity_weight,io.settings.acceleration_weight,
                         io.settings.upper_pose_weight,0.)
        solver.stationary.detect(local.quaternions,local.locations)
        yield from solver.prepare()
        fields = solver.round_fields()
        policy = RepairPolicy()
        policy.time_partition = local.time_partition
        keys = WindowKeys(io,action,local.frames)
        best = result[sl].copy()
        initial = best.copy()
        first_quats = keys.reference.copy()
        best_surface = yield from verify(io,local,topology,max(io.start,start-1),min(io.end,end+1))
        def write(pose):
            quats = model.quats(pose)
            unchanged = np.all(pose[:,:6]==initial[:,:6],axis=-1)
            quats[unchanged] = first_quats[unchanged]
            keys.write(quats)
        def measure(pose):
            solver.set_round_fields(fields)
            solver.collision.restore_full_planes()
            points,orientation,_,wrists = model.state(pose)
            *_,contact = solver.contacts(points,orientation,False)
            depth = contact['body']*local.scale*1000
            return depth,wrists-model.wrist_orig
        best_depth,wrist_error = measure(best)
        step_limit = np.linalg.norm(np.diff(wrist_error,axis=0),axis=-1).max(axis=0)
        step_limit += np.percentile(np.linalg.norm(np.diff(model.wrist_orig,axis=0),axis=-1),95,axis=0)
        # Blend into unchanged neighbouring keys. Preserve truly stationary channels.
        weight = np.clip(np.minimum(local.frames-start+1,end-local.frames+1)/4.,0.,1.)
        weight = weight*weight*(3-2*weight)
        locked = np.zeros((len(weight),6),bool)
        for param,left,right in stationary:
            if param<6:
                locked[:,param] |= (local.frames>=clip.frames[left]) & (local.frames<=clip.frames[right])
        def consider(trial):
            nonlocal best,best_surface,best_depth
            trial[:,:6][locked] = initial[:,:6][locked]
            depth,err = measure(trial)
            if (depth.max()>best_depth.max()+1e-5 or
                    np.any(np.linalg.norm(np.diff(err,axis=0),axis=-1).max(axis=0)>step_limit+1e-5)):
                return
            write(trial)
            found = yield from verify(io,local,topology,max(io.start,start-1),min(io.end,end+1))
            q,prev = SurfaceQuality.measure(found),SurfaceQuality.measure(best_surface)
            if q.improves(prev) or (q==prev and depth.max()<best_depth.max()-.001):
                best,best_surface,best_depth = trial,found,depth
            else:
                write(best)
        for side in contact_sides(best_surface):
            for degrees in ((14,-14,24,-24) if side==0 else (-14,14,-24,24)):
                if not best_surface['problem_frames']:
                    break
                trial = yield from swivel(io,local,model,best,side,weight*math.radians(degrees))
                yield from consider(trial)
        for strategy in range(len(policy.strategies)):
            if not best_surface['problem_frames']:
                break
            for target in policy.targets(best_surface,strategy):
                plan = policy.plan(target,model,local.geometry,local.frames,strategy+2,
                                   local.scale,fields,strategy)
                solver.set_round_fields(plan.fields)
                solver.collision.update_repair_band()
                yield from solver.optimize_staged(150,initial=best,free_frames=plan.free_frames)
                change = (solver.result-best)*weight[:,None,None]
                involved = contact_sides(target)
                for side in (0,1):
                    if side not in involved:
                        change[:,3*side:3*side+3] = 0.
                origin = best
                for factor in (.65,.3):
                    yield from consider(origin+factor*change)
                    if best is not origin:
                        break
        if any(start<=f<=end for f in deep_frames):
            for strength in (6.,16.,32.):
                peak = float(best_depth.max())
                active = best_depth.max(axis=(0,2))>=.8*peak
                weights = fields.body_weights.copy()
                weights += strength*(best_depth/max(1.,peak))**2*active[None,:,None]
                solver.set_round_fields(replace(fields,body_weights=weights))
                solver.collision.update_repair_band()
                yield from solver.optimize_staged(150,initial=best,free_frames=np.ones(len(weight),bool))
                change = (solver.result-best)*weight[:,None,None]
                for side in (0,1):
                    if not active[side]:
                        change[:,3*side:3*side+3] = 0.
                origin = best
                for factor in (1.,.55,.25):
                    yield from consider(origin+factor*change)
                    if best is not origin:
                        break
        write(best)
        result[sl] = best
    return result
