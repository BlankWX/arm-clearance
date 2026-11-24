"""Connect candidate curve edits to Blender's evaluated full-surface verifier."""
import math
import numpy as np
from .half_keys import HalfKeys
from .surface import prepare_surface, surface_check
from ..features.repair.half_search import isolated_half_frames, search_half_pose


def repair_halves(io, clip, candidate, previous_surface, status):
    isolated = isolated_half_frames(previous_surface)
    if not isolated:
        return False
    io.ad.action = candidate
    slot = io.slot_for(candidate, io.source_slot.identifier)
    io.ad.action_slot = slot
    topology = prepare_surface(clip.geometry, clip.meta, clip.body)
    changed = False
    for frame in isolated:
        points = np.arange(-4,5)/8+frame
        keys = HalfKeys(io.scene,io.arm,candidate,slot,io.targets,frame)
        speed_limit = None
        bones = [io.arm.pose.bones[name] for name in io.targets]
        def evaluate():
            wrists, angles = [], []
            for surface in surface_check(io.scene,io.arm,io.mesh,clip.geometry,clip.meta,
                                         clip.body,io.start,io.end,frames=points,
                                         topology=topology,max_pairs=None):
                wrists.append(keys.wrists())
                angles.append([tuple(b.rotation_quaternion.normalized()) for b in bones])
            wrists = np.array(wrists)
            angles = np.array(angles)
            dot = np.abs(np.sum(angles[1:]*angles[:-1],axis=2))
            angular_step = np.max(2*np.arccos(np.clip(dot,0.,1.)),axis=0)
            wrist_step = np.max(np.linalg.norm(np.diff(wrists,axis=0),axis=2),axis=0)
            speed = np.r_[angular_step,wrist_step]
            nonlocal speed_limit
            if speed_limit is None:
                lengths = [sum(bones[j].length for j in (i,i+1))*clip.scale for i in (0,3)]
                speed_limit = 1.75*speed + np.r_[np.full(6,math.radians(2)),.015*np.array(lengths)] * (8/(len(points)-1))
            io.scene.frame_set(math.floor(frame),subframe=.5)
            wrist_error = float(np.linalg.norm(keys.wrists()-keys.original_wrists,axis=1).max())
            return surface,wrist_error,bool(np.all(speed <= speed_limit))
        baseline,_,_ = evaluate()
        status('局部半帧补修 %g：检查相邻整数帧与八分之一帧' % frame)
        vector = yield from search_half_pose(keys,baseline,evaluate)
        if vector.any():
            # Certify the two subintervals around each search sample. Continue
            # from the accepted direction if a narrower contact still exists.
            points = np.arange(-8,9)/16+frame
            keys.write(np.zeros_like(vector))
            speed_limit = None
            original_dense,_,_ = evaluate()
            keys.write(vector)
            dense,_,smooth = evaluate()
            if (dense['collision_samples'] and smooth and
                    set(dense['problem_frames']) <= set(original_dense['problem_frames'])):
                vector = yield from search_half_pose(keys,dense,evaluate,initial=vector)
            final,_,smooth = evaluate()
            # Commit the extra keys only when the full local transition clears.
            if smooth and not final['collision_samples']:
                changed = True
            else:
                keys.write(np.zeros_like(vector))
    return changed
