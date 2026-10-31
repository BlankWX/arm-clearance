"""Constant source poses share C2 correction controls during a held pose.

Hold detection uses the entire excursion from its first sample, not just
neighboring differences, so slow intentional animation is not classified as
static. Quaternion antipodes describe the same pose. Collision objectives are
still evaluated at every captured frame, including moving body/finger geometry.
"""
import numpy as np
from ..source_time import cubic_basis, join_bases

MIN_HOLD_FRAMES = 16
ROTATION_CHORD_TOLERANCE = 1e-7
LOCATION_TOLERANCE = 1e-8


def _runs(values, tolerance, quaternion=False, minimum=MIN_HOLD_FRAMES):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Stationary source values must be finite vectors')
    if quaternion:
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError('Stationary source contains a zero quaternion')
        values = values / norms
    start = 0
    for end in range(1, len(values) + 1):
        same = False
        if end < len(values):
            distance = np.linalg.norm(values[end] - values[start])
            if quaternion:
                distance = min(distance, np.linalg.norm(values[end] + values[start]))
            same = distance <= tolerance
        if not same:
            if end-start >= minimum:
                yield start, end-1
            start = end


def stationary_spans(quaternions, locations=None):
    """Protect complete bilateral held poses, not isolated joints during motion.

    An elbow or wrist can be locally stationary while its upper arm moves.
    Such a joint still needs time-varying collision correction. Require all six
    source rotations (and both wrist positions when supplied) to share a hold.
    """
    q = np.asarray(quaternions)
    if q.ndim != 3 or q.shape[1:] != (6, 4):
        raise ValueError('Expected source quaternions [frames, 6, 4]')
    common = [(0, len(q)-1)]
    def intersect(runs):
        return [(max(a, c), min(b, d)) for a, b in common for c, d in runs
                if min(b, d)-max(a, c)+1 >= MIN_HOLD_FRAMES]
    for parameter in range(6):
        common = intersect(list(_runs(q[:, parameter], ROTATION_CHORD_TOLERANCE, True)))
    parameters = 6
    if locations is not None:
        loc = np.asarray(locations)
        if loc.shape != (len(q), 2, 3):
            raise ValueError('Expected source wrist locations [frames, 2, 3]')
        for side in range(2):
            common = intersect(list(_runs(loc[:, side], LOCATION_TOLERANCE)))
        parameters = 8
    return [(parameter, a, b) for parameter in range(parameters) for a, b in common]


class StationaryControls:
    """Orthogonal projection onto equal-control groups for held source poses.

    Equal cubic-spline controls give a constant correction inside the hold and
    keep the existing C2 transition outside it. Overlapping groups are merged.
    A masked warm pass releases the complete tied group when any member is
    needed; it cannot repair one frame by twisting a different part of the hold.
    """
    def __init__(self, basis, parameters, spans):
        self.controls = basis.shape[1]
        self.dimension = parameters * 3
        groups = []
        for parameter in range(parameters):
            sets = []
            for p, start, end in spans:
                if p != parameter:
                    continue
                active = set(np.flatnonzero(np.any(basis[start:end+1] > 0., axis=0)).tolist())
                merged = []
                for other in sets:
                    if active & other:
                        active |= other
                    else:
                        merged.append(other)
                sets = merged + [active]
            groups.extend((np.array(sorted(indices)), slice(parameter*3, parameter*3+3))
                          for indices in sets if len(indices) > 1)
        self.groups = groups

    def project(self, values):
        view = values.reshape(self.controls, self.dimension)
        for indices, columns in self.groups:
            view[indices, columns] = np.mean(view[indices, columns], axis=0)
        return values

    def expand_free(self, mask):
        view = mask.reshape(self.controls, self.dimension)
        for indices, columns in self.groups:
            if np.any(view[indices, columns]):
                view[indices, columns] = True
        return mask


    def shared_reduction(self):
        """Orthonormal basis when every parameter has the same tied groups."""
        signatures=[]
        for offset in range(0,self.dimension,3):
            signatures.append(tuple(tuple(indices) for indices,columns in self.groups
                                    if columns.start == offset))
        if not signatures or any(value != signatures[0] for value in signatures[1:]):
            return None
        used=set();columns=[]
        for group in signatures[0]:
            column=np.zeros(self.controls)
            column[list(group)]=1./np.sqrt(len(group))
            columns.append(column);used.update(group)
        for index in range(self.controls):
            if index not in used:
                column=np.zeros(self.controls);column[index]=1.;columns.append(column)
        return np.stack(columns,axis=1)


def transition_basis(frames, spacing, spans):
    """Refine only held-pose boundaries, preserving the cubic C2 spline space.

    The old eight-frame lattice can only release a tied hold gradually over
    several supports. Additional knots allow a moving pose to clear contact
    promptly after a hold, without anticipatory motion inside the hold.
    """
    count=int((frames-1)//spacing)+4
    base=np.arange(-3,count+1,dtype=float)*spacing
    added=set()
    for _,start,end in spans:
        for boundary,sign in ((start,-1),(end,1)):
            if boundary in (0,frames-1):continue
            for offset in range(0,3*spacing+1,max(1,spacing//2)):
                value=float(boundary+sign*offset)
                if 0<value<frames-1 and value not in base:added.add(value)
    if not added:return None,()
    knots=np.sort(np.concatenate((base,np.array(sorted(added)))))
    times=np.arange(frames,dtype=float)[:,None]
    basis=((times>=knots[:-1])&(times<knots[1:])).astype(float)
    for degree in range(1,4):
        size=len(knots)-degree-1
        left=knots[degree:degree+size]-knots[:size]
        right=knots[degree+1:degree+1+size]-knots[1:1+size]
        basis=((times-knots[:size])/left*basis[:,:size]
               +(knots[degree+1:degree+1+size]-times)/right*basis[:,1:size+1])
    return basis,tuple(sorted(added))


class StationaryPolicy:
    """An explicit control-constraint port, not an extra soft objective."""
    def __init__(self, spans=()):
        self.spans = spans
        self.time_partition = None

    def detect(self, quaternions, locations):
        self.spans = stationary_spans(quaternions, locations)
        if self.time_partition is not None and self.time_partition.cuts:
            self.spans=[(p,max(a,c),min(b,d-1)) for p,a,b in self.spans
                        for c,d in self.time_partition.blocks if min(b,d-1)-max(a,c)+1>=MIN_HOLD_FRAMES]
        return self.spans

    def refine_basis(self, frames, spacing):
        if self.time_partition is not None and self.time_partition.cuts:
            blocks=[];knots=[]
            for a,b in self.time_partition.blocks:
                spans=[(p,max(x,a)-a,min(y,b-1)-a) for p,x,y in self.spans if max(x,a)<=min(y,b-1)]
                block,extra=transition_basis(b-a,spacing,spans)
                blocks.append(cubic_basis(b-a,spacing) if block is None else block)
                if extra:knots.append((a,extra))
            return join_bases(blocks),tuple(knots)
        return transition_basis(frames, spacing, self.spans)

    def controls(self, basis, parameters):
        return StationaryControls(basis, parameters, self.spans) if self.spans else None
