"""Local quaternion editing in a disposable, explicitly bound Action copy."""
import math
import numpy as np
from mathutils import Quaternion, Vector
from .action_io import channels


class HalfKeys:
    def __init__(self, scene, arm, action, slot, targets, frame):
        self.scene = scene
        self.arm = arm
        self.frame = frame
        self.targets = tuple(targets)
        self.curves = []
        self.base = []
        self.rotation_limits = []
        self.has_key = []
        self.scene.frame_set(math.floor(frame), subframe=.5)
        paths = {(c.data_path, c.array_index): c for c in channels(action, slot)}
        for name in self.targets:
            bone = arm.pose.bones[name]
            path = bone.path_from_id('rotation_quaternion')
            curves = [paths[path, axis] for axis in range(4)]
            self.curves.append(curves)
            def quat(t):
                return Quaternion([curve.evaluate(t) for curve in curves]).normalized()
            self.base.append(quat(frame))
            left, right = quat(frame-.5), quat(frame+.5)
            sweep = 2*math.acos(min(1., abs(left.dot(right))))
            # Keep the detour proportional to the source transition, including
            # a small allowance for bones with little endpoint rotation.
            self.rotation_limits.append(min(math.radians(55), .45*sweep+math.radians(8)))
            self.has_key.append(False)
        self.original_wrists = self.wrists()
        points = [arm.matrix_world @ arm.pose.bones[n].head for n in self.targets]
        outward = (points[0]-points[3]).normalized()
        self.swing_axes = []
        for side, shoulder in enumerate((0, 3)):
            direction = points[shoulder+2]-points[shoulder]
            axis = direction.cross(outward if side == 0 else -outward)
            bone = arm.pose.bones[self.targets[shoulder]]
            basis = (arm.matrix_world @ bone.matrix).to_quaternion()
            self.swing_axes.append(np.array(basis.inverted() @ axis.normalized()))

    def wrists(self):
        return np.array([self.arm.matrix_world @ self.arm.pose.bones[self.targets[i]].head
                         for i in (2, 5)])

    def key(self, curve):
        points = curve.keyframe_points
        low, high = 0, len(points)
        while low < high:
            mid = (low+high)//2
            if points[mid].co.x < self.frame:
                low = mid+1
            else:
                high = mid
        if low >= len(points) or points[low].co.x != self.frame:
            raise ValueError('Missing trial quaternion key')
        return points[low]

    def write(self, rotations):
        for index, vector in enumerate(rotations):
            angle = float(np.linalg.norm(vector))
            curves = self.curves[index]
            if angle < 1e-9:
                if self.has_key[index]:
                    for curve in curves:
                        curve.keyframe_points.remove(self.key(curve))
                        curve.update()
                    self.has_key[index] = False
                continue
            quat = self.base[index] @ Quaternion(Vector(vector / angle), angle)
            quat.make_compatible(self.base[index])
            for c, value in zip(curves, quat):
                if not self.has_key[index]:
                    key = c.keyframe_points.insert(self.frame, value, options={'FAST'})
                else:
                    key = self.key(c)
                key.co[1] = value
                key.interpolation = 'LINEAR'
                c.update()
            self.has_key[index] = True
        self.arm.animation_data.action.update_tag()
