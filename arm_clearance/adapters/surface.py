"""Blender mesh evaluation and exact original surface-check implementation."""
from collections import defaultdict
import math
import numpy as np
from ..features.collision.parts import (CATEGORIES, OWN_CATEGORY, build_surface_parts,
    sample_frames, _components, _inside_closed)

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
