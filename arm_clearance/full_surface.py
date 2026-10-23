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


CATEGORIES = ('arm-body', 'hand-body', 'arm-arm',
              'hand-opposite-arm', 'hand-hand')
OWN_CATEGORY = 'hand-own-arm'
PART_NAMES = ('body', 'arm.L', 'hand.L', 'arm.R', 'hand.R')


def _bone_ids(meta):
    names = list(meta['bones'])
    index = {name: i for i, name in enumerate(names)}
    target = list(meta.get('arm_target_ids', ()))
    all_targets = list(meta.get('target_ids', ()))
    if len(target) != 4:
        if len(all_targets) == 4:
            target = all_targets
        elif len(all_targets) == 6:
            target = [all_targets[i] for i in (0, 1, 3, 4)]
        else:
            target = [index[name] for name in ('腕.L', 'ひじ.L', '腕.R', 'ひじ.R')]
    wrists = list(meta.get('wrist_ids', ()))
    if len(wrists) != 2:
        wrists = ([all_targets[2], all_targets[5]] if len(all_targets) == 6
                  else [index[name] for name in ('手首.L', '手首.R')])
    ids = target + wrists
    if len(set(ids)) != 6 or any(i < 0 or i >= len(names) for i in ids):
        raise ValueError('Full surface verification requires six distinct arm/wrist bones')
    return target, wrists


def classify_bones(meta):
    """Return tree-derived ``side`` (-1/body, 0/L, 1/R) and ``region`` arrays.

    Regions are 0/body, 1/upper arm, 2/forearm, 3/hand and every wrist
    descendant.  Twist/helper descendants are included by ancestry.  ``finger``
    identifies a named finger branch only for optional same-side diagnostics;
    it does not control inclusion in the hand collision surface.
    """
    names = list(meta['bones'])
    parents = np.asarray(meta['parents'], dtype=np.int32)
    if len(parents) != len(names):
        raise ValueError('Bone parent count does not match bone count')
    arm_ids, wrist_ids = _bone_ids(meta)
    roots = {}
    for side in range(2):
        upper, fore = arm_ids[2 * side:2 * side + 2]
        wrist = wrist_ids[side]
        roots.update({upper: (side, 1), fore: (side, 2), wrist: (side, 3)})
    side = np.full(len(names), -1, dtype=np.int8)
    region = np.zeros(len(names), dtype=np.int8)
    finger = np.full(len(names), -1, dtype=np.int32)
    for bone in range(len(names)):
        chain = []
        current = bone
        seen = set()
        while current >= 0:
            if current >= len(names) or current in seen:
                raise ValueError('Invalid or cyclic bone hierarchy')
            seen.add(current)
            chain.append(current)
            if current in roots:
                side[bone], region[bone] = roots[current]
                if region[bone] == 3:
                    # The highest named finger ancestor below the wrist is
                    # stable across all phalanges and driven helper bones.
                    candidates = [i for i in chain[:-1] if '指' in names[i]]
                    if candidates:
                        finger[bone] = candidates[-1]
                break
            current = int(parents[current])
    return {'side': side, 'region': region, 'finger': finger,
            'arm_ids': arm_ids, 'wrist_ids': wrist_ids}


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


def surface_check(scene, arm, mesh, geo, meta, body, start, end, status=None,
                  step=.5, shoulder_fraction=.34, max_pairs=64,
                  check_containment=True, containment_probe=None,
                  include_self_hand=False, frames=None,
                  own_hand_arm=True, own_forearm_cutoff=.78):
    """Yield cumulative JSON-compatible full-skin diagnostics each sample.

    ``step`` supports integer (1), half (.5), or quarter (.25) frames.
    ``frames`` may override the inclusive range with explicit sample times.
    Stored face IDs are original ``geo['triangles']`` indices, not compact BVH
    indices or mesh polygon indices.  Counts are exact; stored face pairs are
    capped by ``max_pairs`` per category per sample (``None`` means uncapped).
    ``own_hand_arm`` defaults to True, adding ``hand-own-arm`` without changing
    the original five category names. It tests all hand faces against the
    whole upper arm and forearm proximal to the .78 rest-length cutoff.

    ``containment_probe(frame, coords_world, prepared_parts)`` may return
    dictionaries with ``category`` and optional diagnostic fields for closed
    solver proxies.  Such findings fail the check, but remain marked as proxy
    evidence instead of being called actual surface intersections.
    """
    import bpy
    from mathutils.bvhtree import BVHTree

    if status:
        status('真实网格复核：按骨骼权重拆分身体、双臂与手部')
    prepared = build_surface_parts(geo, meta, body, shoulder_fraction, own_forearm_cutoff)
    triangles, parts = prepared['triangles'], prepared['parts']
    n_vertices = len(geo['vertices'])
    samples = sample_frames(start, end, step) if frames is None else np.asarray(frames, dtype=float)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Surface verification needs finite sample times')
    if max_pairs is not None and max_pairs < 0:
        raise ValueError('max_pairs must be nonnegative or None')
    pairs = [('arm-body', 'arm_body.L', 'body'), ('arm-body', 'arm_body.R', 'body'),
             ('hand-body', 'hand.L', 'body'), ('hand-body', 'hand.R', 'body'),
             ('arm-arm', 'arm.L', 'arm.R'),
             ('hand-opposite-arm', 'hand.L', 'arm.R'),
             ('hand-opposite-arm', 'hand.R', 'arm.L'), ('hand-hand', 'hand.L', 'hand.R')]
    if own_hand_arm:
        pairs.extend([(OWN_CATEGORY, 'hand.L', 'own_arm.L'),
                      (OWN_CATEGORY, 'hand.R', 'own_arm.R')])
    if status:
        status('真实网格复核：建立连通分区')
    components = {name: _components(triangles, ids) for name, ids in parts.items()}
    if include_self_hand:
        # These are diagnostics only: rotating a wrist cannot remove relative
        # finger/finger intersections already in the source animation.
        vf = prepared['vertex_finger']
        for side, suffix in enumerate(('L', 'R')):
            branch_parts = []
            ids = parts['hand.' + suffix]
            for finger in np.unique(vf[triangles[ids]]):
                if finger < 0:
                    continue
                branch = 'finger.%s.%d' % (suffix, finger)
                # Give a mixed web face to one branch.  Unlike cross-body
                # validation, this optional diagnostic must not compare a
                # triangle with itself through two neighboring finger sets.
                face_fingers = vf[triangles[ids]]
                valid = np.where(face_fingers >= 0, face_fingers, np.iinfo(np.int32).max)
                owner = valid.min(axis=1)
                parts[branch] = ids[owner == finger]
                branch_parts.append(branch)
            for i, a in enumerate(branch_parts):
                for b in branch_parts[i + 1:]:
                    pairs.append(('same-side-finger-diagnostic', a, b))
    # Preserve original polygon order and winding while removing vertices
    # unreferenced by each part. Collision records and seam checks still use
    # the original global face/vertex IDs below.
    part_vertices = {}
    part_polygons = {}
    for name, ids in parts.items():
        vertices, inverse = np.unique(triangles[ids], return_inverse=True)
        part_vertices[name] = vertices
        part_polygons[name] = inverse.reshape(-1, 3).tolist()
    closed_components = {name: [component for component in group if component['closed']]
                         for name, group in components.items()}
    component_polygons = {(name, int(component['faces'][0])):
                          triangles[component['faces']].tolist()
                          for name, group in closed_components.items() for component in group}
    main_categories = CATEGORIES + ((OWN_CATEGORY,) if own_hand_arm else ())
    category_names = main_categories + (('same-side-finger-diagnostic',) if include_self_hand else ())
    result = {
        'samples': 0, 'total_samples': int(len(samples)), 'step': float(step),
        'body_intersection_samples': 0, 'mutual_intersection_samples': 0,
        'own_hand_arm_intersection_samples': 0,
        'collision_samples': 0, 'problem_frames': [], 'records': [],
        'categories': {c: {'intersection_samples': 0, 'intersection_pair_count': 0,
                           'containment_samples': 0, 'proxy_samples': 0,
                           'problem_frames': []} for c in category_names},
        'classification': prepared['diagnostics'],
        'closed_components': {name: sum(c['closed'] for c in group)
                              for name, group in components.items()},
        'scope': ('原始全分辨率变形蒙皮；手掌和全部手指及其边界三角面；'
                  '双侧上肢/手与身体和对侧互检；仅臂身检测排除肩根连接带；'
                  + ('同侧臂手互检，排除末%g%%前臂连接及实际共享点三角对；' %
                     ((1-own_forearm_cutoff)*100) if own_hand_arm else '未启用同侧臂手互检；') +
                  '射线奇偶仅用于真实封闭分量；'
                  '不以开放网格法线符号判断穿模；离散采样不构成连续时间数学证明'),
        'near_surface_depth_max_mm': None,
        'complete': False,
    }
    original_frame, original_subframe = scene.frame_current, scene.frame_subframe
    try:
        for sample_index, frame in enumerate(samples):
            integer = math.floor(float(frame))
            scene.frame_set(integer, subframe=float(frame) - integer)
            evaluated = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
            evaluated_mesh = evaluated.to_mesh()
            try:
                if len(evaluated_mesh.vertices) != n_vertices:
                    raise ValueError('Full surface verification requires unchanged vertex topology')
                if sample_index == 0:
                    evaluated_mesh.calc_loop_triangles()
                    actual = np.empty(len(evaluated_mesh.loop_triangles) * 3, dtype=np.int32)
                    evaluated_mesh.loop_triangles.foreach_get('vertices', actual)
                    if actual.size != triangles.size or not np.array_equal(actual.reshape(-1, 3), triangles):
                        raise ValueError('Evaluated triangle order differs from captured geometry')
                coords = np.empty(n_vertices * 3, dtype=np.float32)
                evaluated_mesh.vertices.foreach_get('co', coords)
                coords = coords.reshape(-1, 3).astype(np.float64)
                matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
                coords = coords @ matrix[:3, :3].T + matrix[:3, 3]
            finally:
                evaluated.to_mesh_clear()
            diagonal = max(float(np.linalg.norm(np.ptp(coords, axis=0))), 1e-6)
            epsilon = max(diagonal * 1e-7, 1e-9)
            full = coords.tolist()
            trees = {name: (BVHTree.FromPolygons(coords[part_vertices[name]].tolist(), polygons,
                                                all_triangles=True)
                            if polygons else None) for name, polygons in part_polygons.items()}
            closed_trees = {}
            record = {'frame': float(frame), 'categories': {c: {'pair_count': 0, 'pairs': [],
                      'containments': [], 'proxies': []} for c in category_names}}
            for category, name_a, name_b in pairs:
                tree_a, tree_b = trees[name_a], trees[name_b]
                if tree_a is None or tree_b is None:
                    continue
                out = record['categories'][category]
                hits = tree_a.overlap(tree_b)
                hit_count = 0
                for local_a, local_b in hits:
                    face_a, face_b = int(parts[name_a][local_a]), int(parts[name_b][local_b])
                    # A shared mesh edge/vertex is a connected skin seam,
                    # not penetration.  No distance-based broad deletion.
                    if np.intersect1d(triangles[face_a], triangles[face_b]).size:
                        continue
                    hit_count += 1
                    if max_pairs is None or len(out['pairs']) < max_pairs:
                        out['pairs'].append({'part_a': name_a, 'part_b': name_b,
                                             'face_a': face_a, 'face_b': face_b})
                out['pair_count'] += hit_count
                if not check_containment or hit_count or category == 'same-side-finger-diagnostic':
                    continue
                # With no surface crossing, one representative per connected
                # source component is sufficient for a fully embedded case.
                for source, target in ((name_a, name_b), (name_b, name_a)):
                    for component in closed_components[target]:
                        key = (target, int(component['faces'][0]))
                        target_coords = coords[component['vertices']]
                        low, high = target_coords.min(axis=0), target_coords.max(axis=0)
                        for candidate in components[source]:
                            vertex = int(candidate['vertices'][0])
                            point = coords[vertex]
                            if np.any(point <= low + epsilon) or np.any(point >= high - epsilon):
                                continue
                            if key not in closed_trees:
                                closed_trees[key] = BVHTree.FromPolygons(
                                    full, component_polygons[key], all_triangles=True)
                            if _inside_closed(closed_trees[key], point, epsilon, diagonal):
                                out['containments'].append({'source_part': source, 'target_part': target,
                                    'source_vertex': vertex, 'source_face': int(candidate['faces'][0]),
                                    'target_component_face': int(component['faces'][0]),
                                    'method': 'closed-manifold-three-ray-parity'})
            if containment_probe is not None:
                for finding in containment_probe(float(frame), coords, prepared) or ():
                    finding = dict(finding)
                    category = finding.pop('category')
                    if category not in main_categories:
                        raise ValueError('Unknown proxy collision category: ' + category)
                    finding['method'] = 'external-closed-proxy'
                    record['categories'][category]['proxies'].append(finding)
            failed = set()
            for category, data in record['categories'].items():
                summary = result['categories'][category]
                summary['intersection_samples'] += int(data['pair_count'] > 0)
                summary['intersection_pair_count'] += data['pair_count']
                summary['containment_samples'] += int(bool(data['containments']))
                summary['proxy_samples'] += int(bool(data['proxies']))
                if data['pair_count'] or data['containments'] or data['proxies']:
                    summary['problem_frames'].append(float(frame))
                    if category in main_categories:
                        failed.add(category)
            result['samples'] += 1
            result['body_intersection_samples'] += int(bool(failed & {'arm-body', 'hand-body'}))
            result['mutual_intersection_samples'] += int(bool(failed - {'arm-body', 'hand-body'}))
            result['own_hand_arm_intersection_samples'] += int(OWN_CATEGORY in failed)
            if failed:
                result['collision_samples'] += 1
                result['problem_frames'].append(float(frame))
            if failed or any(r['pair_count'] for r in record['categories'].values()):
                result['records'].append(record)
            result['current_frame'] = float(frame)
            result['complete'] = sample_index == len(samples) - 1
            if status is not None:
                status('全分辨率臂手复核 %g/%g（%d/%d）' %
                       (frame, samples[-1], sample_index + 1, len(samples)))
            yield result
    finally:
        scene.frame_set(original_frame, subframe=original_subframe)
