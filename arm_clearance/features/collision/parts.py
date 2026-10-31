"""Full skin verification for the six-bone arm/wrist correction.

The classification helpers use NumPy only and can run outside Blender.  The
generator evaluates the *actual mesh* and yields after every sample, so closing
it cancels promptly and restores the original frame/subframe.  It never writes
an action, a bone, or a mesh.

Each original triangle belongs to exactly one body/arm/hand part.  A triangle
touching a hand vertex belongs to the hand, including palm/finger/wrist seams;
there is deliberately no ``all(vertices_in_hand)`` boundary erosion.  Shoulder
attachment faces are excluded only from the arm/body test. Same-side hand/arm
checks retain the whole hand and exclude the distal forearm attachment band.

BVH overlap detects surface intersections, not all enclosed configurations.
Optional containment tests use ray parity on genuinely closed, manifold
components only, never the sign of an open mesh's nearest normal.  An external
``containment_probe`` may supplement these with the solver's closed proxies;
such results are explicitly reported as proxy evidence.
"""

from collections import defaultdict
import math
import numpy as np
from ...anatomy import _bone_ids, classify_bones


CATEGORIES = ('arm-body', 'hand-body', 'arm-arm',
              'hand-opposite-arm', 'hand-hand')
OWN_CATEGORY = 'hand-own-arm'
PART_NAMES = ('body', 'arm.L', 'hand.L', 'arm.R', 'hand.R')






def _triangle_lookup_mask(triangles, selected):
    selected = np.asarray(selected, dtype=np.int64).reshape(-1, 3)
    wanted = {tuple(sorted(map(int, t))) for t in selected}
    return np.fromiter((tuple(sorted(map(int, t))) in wanted for t in triangles),
                       dtype=bool, count=len(triangles))


def build_surface_parts(geo, meta, body=None, shoulder_fraction=.34,
                        own_forearm_cutoff=.78):
    """Partition full-resolution original faces, preserving boundary faces.

    ``body`` uses original mesh vertex indices, as returned by
    ``pipeline.geometry_data``.  If omitted, exported ``body_ids`` and
    ``body_triangles`` are honored; otherwise every non-limb face is body.
    Returns NumPy arrays for reuse by offline scripts and the Blender checker.
    ``shoulder_fraction`` is a documented proximal upper-arm attachment band,
    expressed as shoulder-to-elbow rest-space length; it does not affect
    hand/body or opposite-limb tests.
    """
    if not 0 <= shoulder_fraction < 1:
        raise ValueError('shoulder_fraction must be in [0, 1)')
    if not 0 < own_forearm_cutoff < 1:
        raise ValueError('own_forearm_cutoff must be in (0, 1)')
    classification = classify_bones(meta)
    vertices = np.asarray(geo['vertices'])
    triangles = np.asarray(geo['triangles'], dtype=np.int32).reshape(-1, 3)
    dominant = np.asarray(geo['dominant'], dtype=np.int32)
    if len(dominant) != len(vertices) or (len(triangles) and
            (triangles.min() < 0 or triangles.max() >= len(vertices))):
        raise ValueError('Invalid original mesh topology or dominant weights')
    if dominant.size and (dominant.min() < 0 or dominant.max() >= len(meta['bones'])):
        raise ValueError('Dominant bone index outside metadata')
    vertex_side = classification['side'][dominant]
    vertex_region = classification['region'][dominant]
    vertex_finger = classification['finger'][dominant]
    face_sides = vertex_side[triangles]
    face_regions = vertex_region[triangles]
    left_count = (face_sides == 0).sum(axis=1)
    right_count = (face_sides == 1).sum(axis=1)
    any_limb = (left_count + right_count) > 0
    face_side = np.where(any_limb, (right_count > left_count).astype(np.int8), -1)
    # Any hand vertex retains the complete original face in hand tests.
    face_hand = ((face_regions == 3) & (face_sides == face_side[:, None])).any(axis=1)
    part_id = np.zeros(len(triangles), dtype=np.int8)
    part_id[any_limb] = 1 + face_side[any_limb] * 2 + face_hand[any_limb]
    if body is None and 'body_ids' in geo and 'body_triangles' in geo:
        body = np.asarray(geo['body_ids'])[np.asarray(geo['body_triangles'])]
    allowed_body = (np.ones(len(triangles), dtype=bool) if body is None
                    else _triangle_lookup_mask(triangles, body))
    part_id[(part_id == 0) & ~allowed_body] = -1
    shoulder = np.zeros(len(triangles), dtype=bool)
    rest = np.asarray(geo['rest'])
    for side in range(2):
        upper, fore = classification['arm_ids'][2 * side:2 * side + 2]
        origin = rest[upper, :3, 3]
        axis = rest[fore, :3, 3] - origin
        length_squared = float(axis @ axis)
        if length_squared <= 1e-14:
            raise ValueError('Upper arm has zero rest length')
        along = ((vertices - origin) @ axis) / length_squared
        # Only upper-arm-root faces may be attachment faces.  Retain a whole
        # face once any of its vertices extends beyond the attachment band.
        shoulder |= ((part_id == 1 + side * 2) &
                     ~np.isin(face_regions, (2, 3)).any(axis=1) &
                     (along[triangles].max(axis=1) <= shoulder_fraction))
    parts = {name: np.flatnonzero(part_id == i) for i, name in enumerate(PART_NAMES)}
    parts['arm_body.L'] = parts['arm.L'][~shoulder[parts['arm.L']]]
    parts['arm_body.R'] = parts['arm.R'][~shoulder[parts['arm.R']]]
    own_excluded = []
    for side, suffix in enumerate(('L', 'R')):
        fore = classification['arm_ids'][2*side+1]
        wrist = classification['wrist_ids'][side]
        origin = rest[fore, :3, 3]
        axis = rest[wrist, :3, 3] - origin
        length_squared = float(axis @ axis)
        if length_squared <= 1e-14:
            raise ValueError('Forearm has zero rest length')
        t = (vertices-origin) @ axis / length_squared
        ids = parts['arm.' + suffix]
        # Exclude only faces wholly in the last 22% forearm band. A complete
        # original triangle crossing the .78 cut is retained. No hand or
        # finger faces are removed from the same-side collision surface.
        has_forearm = ((face_regions[ids] == 2) & (face_sides[ids] == side)).any(axis=1)
        distal = has_forearm & (t[triangles[ids]].min(axis=1) >= own_forearm_cutoff)
        parts['own_arm.' + suffix] = ids[~distal]
        own_excluded.append(int(distal.sum()))
    mixed_hand = face_hand & (face_regions != 3).any(axis=1)
    return {'triangles': triangles, 'parts': parts, 'part_id': part_id,
            'shoulder_faces': np.flatnonzero(shoulder),
            'vertex_side': vertex_side, 'vertex_region': vertex_region,
            'vertex_finger': vertex_finger, 'classification': classification,
            'diagnostics': {
                'original_faces': len(triangles),
                'part_faces': {name: int(len(parts[name])) for name in PART_NAMES},
                'hand_boundary_faces_preserved': int(mixed_hand.sum()),
                'cross_side_source_faces': np.flatnonzero((left_count > 0) & (right_count > 0)).tolist(),
                'shoulder_attachment_faces_excluded_from_body_test': int(shoulder.sum()),
                'shoulder_fraction': float(shoulder_fraction),
                'own_forearm_cutoff': float(own_forearm_cutoff),
                'own_distal_forearm_connection_faces_excluded': own_excluded,
                'ignored_nonbody_material_faces': int((part_id < 0).sum()),
            }}


def sample_frames(start, end, step=.5):
    """Inclusive sample times without accumulating floating-point drift."""
    start, end, step = float(start), float(end), float(step)
    if not all(math.isfinite(v) for v in (start, end, step)) or step <= 0 or end < start:
        raise ValueError('Invalid surface verification frame range/step')
    n = int(math.floor((end - start) / step + 1e-8))
    frames = start + np.arange(n + 1, dtype=np.float64) * step
    if frames[-1] < end - 1e-7:
        frames = np.append(frames, end)
    else:
        frames[-1] = end
    return frames


def _components(triangles, face_ids):
    """Edge-connected face components and an orientation-free closed flag."""
    face_ids = np.asarray(face_ids, dtype=np.int32)
    parent = np.arange(len(face_ids), dtype=np.int32)
    edges = defaultdict(list)
    for local, face in enumerate(triangles[face_ids]):
        a, b, c = map(int, face)
        for x, y in ((a, b), (b, c), (c, a)):
            edges[(min(x, y), max(x, y))].append(local)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i
    for linked in edges.values():
        root = find(linked[0])
        for i in linked[1:]:
            other = find(i)
            if root != other:
                parent[other] = root
    groups = defaultdict(list)
    for local, face_id in enumerate(face_ids):
        groups[find(local)].append(int(face_id))
    closed = {root: True for root in groups}
    for linked in edges.values():
        if len(linked) != 2:
            closed[find(linked[0])] = False
    return [{'faces': np.asarray(faces, dtype=np.int32),
             'vertices': np.unique(triangles[faces]), 'closed': closed[root]}
            for root, faces in groups.items()]


def _inside_closed(tree, point, epsilon, diagonal):
    """Conservative three-ray parity; ambiguous parity is not a penetration."""
    from mathutils import Vector
    point = Vector(point)
    near = tree.find_nearest(point)
    if near[3] is None or near[3] <= epsilon * 4:
        return False
    parities = []
    for direction in ((1., .371, .529), (.277, 1., .613), (.419, .233, 1.)):
        direction = Vector(direction).normalized()
        origin = point.copy()
        count = 0
        traveled = 0.
        # A finite limit also protects against degenerate duplicate faces.
        for _ in range(4096):
            location, normal, face, distance = tree.ray_cast(origin, direction, diagonal * 2)
            if location is None:
                break
            count += 1
            advance = float(distance) + epsilon
            traveled += advance
            origin = location + direction * epsilon
            if traveled > diagonal * 2:
                break
        else:
            return False
        parities.append(count % 2)
    return all(p == 1 for p in parities)


