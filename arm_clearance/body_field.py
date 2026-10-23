"""Tight, conservative body distance proxies using posed convex body patches.

No scene changes.  Blender bmesh is used only to construct cached convex hulls.
Body triangles are covered by overlapping
thin anatomical slabs.  Triangles are clipped against the slabs in rest space;
the resulting vertices retain exact barycentric coordinates on their original
triangles.  Each posed collider triangle is covered by those interpolated
patches and hence by their regional convex proxies.  This covers the supplied
collider; it does not assert that a decimated collider encloses the original
full-resolution skin, which must still be checked separately.
"""

import numpy as np


def _directions(count=64):
    """Antipodal unit directions including the six Cartesian directions."""
    # Antipodal samples avoid a one-sided sampling bias and make the bound
    # deterministic.  All offsets are recomputed from actual posed vertices.
    pairs = (count-6)//2
    k = np.arange(pairs, dtype=np.float64)
    z = (k+.5)/pairs
    phi = k*np.pi*(3.-np.sqrt(5.))
    ring = np.sqrt(np.maximum(1.-z*z, 0.))
    hemisphere = np.column_stack((ring*np.cos(phi), ring*np.sin(phi), z))
    return np.concatenate((np.eye(3), -np.eye(3), hemisphere, -hemisphere))


def _skin(vertices, joint, weight, rest_inverse, pose):
    transform = pose @ rest_inverse
    result = np.zeros_like(vertices, dtype=np.float64)
    for k in range(joint.shape[1]):
        matrix = transform[joint[:, k]]
        result += (np.einsum('nij,nj->ni', matrix[:, :3, :3], vertices)
                   + matrix[:, :3, 3])*weight[:, k, None]
    return result


def _orthogonal_rotations(matrices):
    left, _, right = np.linalg.svd(matrices)
    rotation = left @ right
    bad = np.linalg.det(rotation) < 0.
    if np.any(bad):
        left[bad, :, -1] *= -1.
        rotation = left @ right
    return rotation


def _clip_barycentric(polygon, signed_at_corners):
    """Clip a barycentric polygon to a linear rest-space halfspace."""
    if not len(polygon):
        return []
    output = []
    previous = polygon[-1]
    previous_distance = float(previous @ signed_at_corners)
    previous_inside = previous_distance >= -1e-12
    for current in polygon:
        distance = float(current @ signed_at_corners)
        inside = distance >= -1e-12
        if inside != previous_inside:
            fraction = previous_distance/(previous_distance-distance)
            output.append(previous+fraction*(current-previous))
        if inside:
            output.append(current)
        previous, previous_distance, previous_inside = current, distance, inside
    return output


def _convex_directions(cloud, fallback):
    """All real convex-hull face normals, including thin-slab cap faces."""
    try:
        import bmesh
    except ImportError:
        return fallback.copy(), 'uniform-fallback'
    mesh = bmesh.new()
    try:
        verts = [mesh.verts.new(co) for co in cloud]
        hull = bmesh.ops.convex_hull(mesh, input=verts, use_existing_faces=False)
        faces = [item for item in hull['geom'] if isinstance(item, bmesh.types.BMFace)]
        bmesh.ops.recalc_face_normals(mesh, faces=faces); mesh.normal_update()
        centre = cloud.mean(axis=0)
        normals = []; seen = set()
        for face in faces:
            normal = np.array(face.normal, dtype=np.float64)
            length = np.linalg.norm(normal)
            if length < .9:
                continue
            if np.dot(normal, np.array(face.verts[0].co)-centre) < 0.:
                normal = -normal
            # Negating a normal preserves its norm exactly; reuse that result.
            normal /= length
            # Only merge numerically identical planes; important anatomical
            # normals must not be discarded merely to meet a plane budget.
            signature = tuple(round(float(component)) for component in normal*1e7)
            if signature not in seen:
                seen.add(signature); normals.append(normal)
        if len(normals) < 4:
            return fallback.copy(), 'uniform-fallback'
        return np.array(normals), 'all-posed-hull-faces'
    finally:
        mesh.free()


class TightBodyField:
    """True regional hull field with a documented small smooth bias.

    ``distances`` returns a conservative lower estimate of signed distance to
    the union of regional convex proxies.  Its normals are the analytical
    gradient of that field, not unit vectors near blended facets.  Callers
    needing support radii should normalize the normals themselves.

    Smooth plane max uses the per-pose constant ``tau*log(face_count)``
    correction required to remain conservative.  With ~0.1 mm tau its
    numerical expansion, including the smooth AABB intersection and union,
    is reported explicitly.  No radius or user clearance is included.

    Outside ``narrow_band`` from a region's AABB, its conservative AABB distance
    replaces the plane query.  Those positive distances are larger than the
    largest intended limb support.  The contact band is continuous and fully
    evaluated; do not use this field as an accurate far-away distance oracle.
    """

    def __init__(self, collider, matrices, metadata, length, progress=None):
        self.collider = collider
        self.matrices = np.asarray(matrices, dtype=np.float64)
        self.metadata = metadata
        self.length = float(length)
        self.progress = progress or (lambda *args: None)
        if self.length <= 0. or self.matrices.ndim != 4:
            raise ValueError('Positive arm length and [frames,bones,4,4] matrices required')
        self.n = len(self.matrices)
        self.plane_directions = _directions(64)
        self.plane_tau = self.length*.0005
        self.union_tau = self.length*.00025
        self.narrow_band = self.length*.55
        self.regions = []
        self.prepared = False

    def _region_seeds(self):
        vertices = np.asarray(self.collider['vertices'], dtype=np.float64)
        triangles = np.asarray(self.collider['triangles'], dtype=np.int64)
        height = float(np.ptp(vertices[:, 2]))
        if height <= 1e-10:
            raise ValueError('Body collider must have non-zero height')
        bottom = float(vertices[:, 2].min())
        corners = vertices[triangles]
        minimum = corners.min(axis=1); maximum = corners.max(axis=1)
        # 8.5% height bands overlap by 3%.  Splitting both legs and torso
        # preserves inter-leg, waist and breast concavities that broad hulls fill.
        centres = np.arange(0., 1.00001, .055)
        centres = np.r_[centres, 1.] if centres[-1] < 1. else centres
        regions = []
        covered = np.zeros(len(triangles), dtype=bool)
        for centre in centres:
            low = bottom+(centre-.0425)*height
            high = bottom+(centre+.0425)*height
            sides = (-1, 1) if centre < .91 else (0,)
            for side in sides:
                candidate = (maximum[:, 2] >= low)&(minimum[:, 2] <= high)
                if side:
                    candidate &= (maximum[:, 0] if side > 0 else -minimum[:, 0]) >= -height*.003
                triangle_ids = np.flatnonzero(candidate)
                if not len(triangle_ids):
                    continue
                source_ids = []; barycentrics = []; seen = set()
                for triangle_id in triangle_ids:
                    tri = triangles[triangle_id]; co = corners[triangle_id]
                    polygon = list(np.eye(3))
                    polygon = _clip_barycentric(polygon, co[:, 2]-low)
                    polygon = _clip_barycentric(polygon, high-co[:, 2])
                    if side:
                        polygon = _clip_barycentric(polygon, co[:, 0]*side+height*.003)
                    if len(polygon) < 3:
                        continue
                    covered[triangle_id] = True
                    for bary in polygon:
                        # Adjacent triangles share many original vertices and
                        # edge cuts.  Merge identical source/barycentric pairs,
                        # preserving separately weighted duplicate shells.
                        nonzero = np.flatnonzero(np.abs(bary)>1e-10)
                        signature = tuple(sorted((int(tri[j]), round(float(bary[j]), 10)) for j in nonzero))
                        if signature in seen:
                            continue
                        seen.add(signature); source_ids.append(tri); barycentrics.append(bary)
                if len(source_ids) < 4:
                    continue
                source_ids = np.array(source_ids, dtype=np.int32)
                barycentrics = np.array(barycentrics, dtype=np.float64)
                mass = np.zeros(self.matrices.shape[1], dtype=np.float64)
                joints = np.asarray(self.collider['joint'])[source_ids]
                weights = np.asarray(self.collider['weight'])[source_ids]*barycentrics[:, :, None]
                np.add.at(mass, joints.ravel(), weights.ravel())
                anchor = int(np.argmax(mass))
                regions.append({'sources':source_ids, 'barycentric':barycentrics,
                                'anchor':anchor, 'centre':float(centre),
                                'side':int(side)})
        if not regions:
            raise ValueError('No body regions could be formed')
        # The overlapping z intervals cover the complete height, and the left /
        # right halfspaces cover x.  Clipping every crossing triangle by those
        # same linear planes therefore covers its full barycentric domain.
        if not np.all(covered):
            raise ValueError('Body region seed layout left collider triangles uncovered')
        self.triangle_coverage = int(covered.sum())
        return regions

    def prepare(self):
        """Cache all posed support offsets; yield periodically for UI updates."""
        self.prepared = False
        self.progress('collider_seed', 0, 1)
        self.regions = self._region_seeds()
        self.progress('collider_seed', 1, 1)
        vertices = np.asarray(self.collider['vertices'], dtype=np.float64)
        joint = np.asarray(self.collider['joint'], dtype=np.int64)
        weight = np.asarray(self.collider['weight'], dtype=np.float64)
        rest_inverse = np.linalg.inv(np.asarray(self.collider['rest'], dtype=np.float64))
        anchors = sorted({region['anchor'] for region in self.regions})
        rotations = {bone:_orthogonal_rotations(self.matrices[:, bone, :3, :3]) for bone in anchors}
        origins = {bone:self.matrices[:, bone, :3, 3] for bone in anchors}
        rest_rotations = _orthogonal_rotations(np.asarray(self.collider['rest'], dtype=np.float64)[:, :3, :3])
        for region in self.regions:
            region['rotations'] = rotations[region['anchor']]
            region['origins'] = origins[region['anchor']]
            region['cap_directions'] = np.array([[0.,0.,1.],[0.,0.,-1.],[1.,0.,0.],[-1.,0.,0.]]) @ rest_rotations[region['anchor']]
            region['plane_frames'] = []
            region['offset_frames'] = []
            region['direction_sources'] = set()
            region['bounds'] = np.empty((self.n, 2, 3), dtype=np.float64)
        maximum_plane_error = 0.
        for frame, pose in enumerate(self.matrices):
            posed = _skin(vertices, joint, weight, rest_inverse, pose)
            for region in self.regions:
                cloud = np.sum(posed[region['sources']]*region['barycentric'][:, :, None], axis=1)
                region['bounds'][frame, 0] = cloud.min(axis=0)
                region['bounds'][frame, 1] = cloud.max(axis=0)
                local = (cloud-region['origins'][frame]) @ region['rotations'][frame]
                directions, source = _convex_directions(local, self.plane_directions)
                region['plane_frames'].append(directions)
                region['direction_sources'].add(source)
                support = local @ directions.T
                offsets = support.max(axis=0)
                region['offset_frames'].append(offsets)
                maximum_plane_error = max(maximum_plane_error, float((support-offsets).max()))
            if frame % 20 == 0:
                self.progress('colliders', frame, self.n)
                yield None
        self.progress('collider_pack', 0, len(self.regions))
        maximum_planes = 0
        for region in self.regions:
            counts = np.array([len(v) for v in region['plane_frames']], dtype=np.int32)
            maximum = int(counts.max()); maximum_planes = max(maximum_planes, maximum)
            region['counts'] = counts
            region['directions'] = np.zeros((self.n, maximum, 3), dtype=np.float64)
            region['offsets'] = np.full((self.n, maximum), 1e4, dtype=np.float64)
            for frame, (directions, offsets) in enumerate(zip(region['plane_frames'], region['offset_frames'])):
                region['directions'][frame, :len(directions)] = directions
                region['offsets'][frame, :len(offsets)] = offsets
            del region['plane_frames']; del region['offset_frames']
        self.report = {
            'regions':len(self.regions), 'maximum_planes_per_region':maximum_planes,
            'mean_planes_per_region':float(np.mean([r['counts'].mean() for r in self.regions])),
            'direction_sources':sorted({source for r in self.regions for source in r['direction_sources']}),
            'collider_triangles_covered':self.triangle_coverage,
            'all_region_vertices_inside_support_planes':bool(maximum_plane_error <= 1e-12),
            'plane_tau':self.plane_tau, 'union_tau':self.union_tau,
            'plane_smoothing_expansion_bound':self.plane_tau*np.log(maximum_planes),
            'union_smoothing_expansion_bound':self.union_tau*np.log(len(self.regions)),
            'box_intersection_smoothing_expansion_bound':self.plane_tau*np.log(2.),
            'total_smoothing_expansion_bound':self.plane_tau*np.log(maximum_planes)+self.plane_tau*np.log(2.)+self.union_tau*np.log(len(self.regions)),
            'narrow_band':self.narrow_band,
            'anchors':[{'bone_id':r['anchor'], 'bone':self.metadata['bones'][r['anchor']],
                        'z_centre':r['centre'], 'side':r['side'], 'vertices':len(r['sources'])}
                       for r in self.regions],
            'scope':'Conservative cover of the supplied posed collider triangles. Regional convexity and finite directions add geometric clearance beyond the explicit smoothing bound. Full skin verification remains required.'}
        self.prepared = True
        self.progress('colliders', self.n, self.n)
        yield None

    def update_active_planes(self, points, threshold_tau=15.):
        """Freeze query-relevant true hull planes for the next solve stage.

        This is an explicit stage-boundary operation.  Never call it during an
        Armijo line search: a stage must optimize one unchanged objective.
        Every actual maximum plane and every plane within threshold_tau*tau of
        it is retained at all supplied contact-band points.  Six local axial
        support planes are additionally retained for each frame.  No maximum
        plane budget silently discards a relevant face.

        Omitted planes keep their original normalization count.  Consequently
        the reduced field is conservative everywhere, including after points
        move.  At the supplied points its distance loss is at most
        tau*log(1 + full_face_count*exp(-threshold_tau)).  A new stage should
        refresh this cache, and final verification should restore full planes.
        """
        if not self.prepared:
            raise RuntimeError('Consume prepare() before selecting active planes')
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 4 or points.shape[0] != self.n or points.shape[1] != 2 or points.shape[-1] != 3:
            raise ValueError('points must have shape [frames,2,samples,3]')
        if threshold_tau <= 0.:
            raise ValueError('threshold_tau must be positive')
        flat = points.reshape(self.n, -1, 3)
        all_counts = []; maximum_loss = 0.
        for region in self.regions:
            if 'full_directions' not in region:
                region['full_directions'] = region['directions']
                region['full_offsets'] = region['offsets']
            directions = region['full_directions']; offsets = region['full_offsets']
            selected = np.zeros(directions.shape[:2], dtype=bool)
            # Preserve each available positive/negative local-axis support.
            for axis in range(3):
                value = directions[..., axis].copy()
                valid = np.arange(value.shape[1])[None] < region['counts'][:, None]
                selected[np.arange(self.n), np.argmax(np.where(valid, value, -1e4), axis=1)] = True
                selected[np.arange(self.n), np.argmin(np.where(valid, value, 1e4), axis=1)] = True
            for cap in region['cap_directions']:
                value = np.einsum('nfi,i->nf', directions, cap)
                selected[np.arange(self.n), np.argmax(np.where(valid, value, -1e4), axis=1)] = True
            lo = region['bounds'][:, None, 0]; hi = region['bounds'][:, None, 1]
            box_vector = np.maximum(flat-hi, 0.)-np.maximum(lo-flat, 0.)
            active = np.linalg.norm(box_vector, axis=-1) <= self.narrow_band
            if np.any(active):
                frame_ids, point_ids = np.nonzero(active)
                relative = flat[frame_ids, point_ids]-region['origins'][frame_ids]
                local = np.einsum('ni,nij->nj', relative, region['rotations'][frame_ids])
                values = np.einsum('ai,afi->af', local, directions[frame_ids])-offsets[frame_ids]
                close = values >= values.max(axis=1, keepdims=True)-threshold_tau*self.plane_tau
                np.logical_or.at(selected, frame_ids, close)
            counts = selected.sum(axis=1)
            maximum = int(counts.max())
            reduced_directions = np.zeros((self.n, maximum, 3), dtype=np.float64)
            reduced_offsets = np.full((self.n, maximum), 1e4, dtype=np.float64)
            for frame in range(self.n):
                ids = np.flatnonzero(selected[frame])
                reduced_directions[frame, :len(ids)] = directions[frame, ids]
                reduced_offsets[frame, :len(ids)] = offsets[frame, ids]
            region['directions'] = reduced_directions; region['offsets'] = reduced_offsets
            region['active_plane_counts'] = counts
            all_counts.extend(counts.tolist())
            maximum_loss = max(maximum_loss, self.plane_tau*np.log1p(float(region['counts'].max())*np.exp(-threshold_tau)))
        result = {'mean_active_planes':float(np.mean(all_counts)),
                  'maximum_active_planes':int(np.max(all_counts)),
                  'p95_active_planes':float(np.percentile(all_counts,95)),
                  'threshold_tau':float(threshold_tau),
                  'distance_loss_bound_at_update_points':float(maximum_loss),
                  'scope':'Frozen conservative subset; refresh only between solve stages; restore full planes for final verification.'}
        self.report['active_plane_cache'] = result
        return result

    def restore_full_planes(self):
        """Restore exact cached hull planes before final collision checks."""
        for region in self.regions:
            if 'full_directions' in region:
                region['directions'] = region.pop('full_directions')
                region['offsets'] = region.pop('full_offsets')
                region.pop('active_plane_counts', None)
        self.report.pop('active_plane_cache', None)

    def distances(self, points, hessian=False, hessian_mask=None,
                  hessian_distance_limit=None, contact_limit=None):
        """Return the regional field and its analytic gradient/Hessian.

        Optional Hessian mask/limit arrays broadcast to [frames,2,samples].
        A Hessian limit omits negligible second moments beyond limit+24*tau.
        With contact_limit, points proven farther than their support/clearance
        bound use an AABB lower bound with zero derivatives. This preserves the
        exact contact energy, but that option is not a far-field normal query.
        Omitting contact_limit evaluates the full original field at every point.
        """
        if not self.prepared:
            raise RuntimeError('Consume prepare() before querying the body field')
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 4 or points.shape[0] != self.n or points.shape[1] != 2 or points.shape[-1] != 3:
            raise ValueError('points must have shape [frames,2,samples,3] matching the cached clip')
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        count = int(np.prod(shape[1:]))
        frame_ids = np.repeat(np.arange(self.n), count)
        wanted = (np.ones(len(flat), dtype=bool) if hessian_mask is None else
                  np.broadcast_to(np.asarray(hessian_mask, dtype=bool), shape).reshape(-1))
        limit = (None if hessian_distance_limit is None else
                 np.broadcast_to(np.asarray(hessian_distance_limit, dtype=np.float64), shape).reshape(-1))
        if contact_limit is None:
            values = self._distances_indexed(flat, frame_ids, hessian, wanted, limit)
        else:
            contact_limit = np.broadcast_to(np.asarray(contact_limit, dtype=np.float64), shape).reshape(-1)
            if not np.isfinite(contact_limit).all() or np.any(contact_limit < 0.):
                raise ValueError('contact_limit must be finite and nonnegative')
            posed = flat.reshape(self.n, count, 3)
            nearest_squared = np.full((self.n, count), np.inf)
            for region in self.regions:
                lo = region['bounds'][:, None, 0]; hi = region['bounds'][:, None, 1]
                outside = np.maximum(posed-hi, 0.)-np.maximum(lo-posed, 0.)
                np.minimum(nearest_squared, np.sum(outside*outside, axis=-1), out=nearest_squared)
            nearest_squared = nearest_squared.reshape(-1)
            # Every regional normalized smooth intersection >= box distance
            # minus plane_tau*log(2); the smooth union loses at most log(R).
            bias = self.plane_tau*np.log(2.)+self.union_tau*np.log(len(self.regions))
            keep = ((nearest_squared <= (contact_limit+bias+1e-12)**2)
                    | (nearest_squared == 0.))
            counts = keep.reshape(self.n, count).sum(axis=1)
            self.last_contact_query = {
                'points': int(len(flat)), 'possible_points': int(keep.sum()),
                'original_points_per_frame': count,
                'maximum_possible_points_per_frame': int(counts.max()),
                'indexed_points': int(keep.sum()), 'padding_points': 0}
            if keep.all():
                values = self._distances_indexed(flat, frame_ids, hessian, wanted, limit)
            else:
                distance = np.sqrt(nearest_squared)-bias
                gradient = np.zeros_like(flat)
                curvature = np.zeros((len(flat), 3, 3)) if hessian else None
                if keep.any():
                    selected = self._distances_indexed(flat[keep], frame_ids[keep], hessian,
                        wanted[keep], None if limit is None else limit[keep])
                    distance[keep] = selected[0]
                    gradient[keep] = selected[1]
                    if hessian:
                        curvature[keep] = selected[2]
                values = (distance, gradient, curvature) if hessian else (distance, gradient)
        result = (values[0].reshape(shape), values[1].reshape(points.shape))
        return result+(values[2].reshape(shape+(3, 3)),) if hessian else result

    def _distances_indexed(self, flat, frame_ids, hessian, hessian_wanted,
                           hessian_limit):
        """Evaluate the unchanged regional field for independent indexed points.

        Each point carries its actual animation-frame index. Flattening removes
        padding and lets contact queries allocate union/Hessian arrays only for
        points whose body contact has not already been ruled out.
        """
        union_distance = None; union_normal = None; union_hessian = None
        for region in self.regions:
            lo = region['bounds'][frame_ids, 0]; hi = region['bounds'][frame_ids, 1]
            box_vector = np.maximum(flat-hi, 0.)-np.maximum(lo-flat, 0.)
            box_distance = np.linalg.norm(box_vector, axis=-1)
            active = box_distance <= self.narrow_band
            distance = box_distance.copy()
            normal = box_vector/np.maximum(box_distance[..., None], 1e-15)
            curvature = np.zeros((len(flat), 3, 3), dtype=np.float64) if hessian else None
            if np.any(active):
                point_ids = np.flatnonzero(active)
                query_frames = frame_ids[point_ids]
                relative = flat[point_ids]-region['origins'][query_frames]
                local = np.einsum('ni,nij->nj', relative, region['rotations'][query_frames])
                query_directions = region['directions'][query_frames]
                plane_values = np.einsum('ai,afi->af', local, query_directions)-region['offsets'][query_frames]
                maximum = plane_values.max(axis=1)
                # The raw plane values are dead after the maximum. Keep the
                # same subtract/divide/exp sequence while reusing its buffer.
                weights = plane_values
                weights -= maximum[:, None]
                weights /= self.plane_tau
                np.exp(weights, out=weights)
                total = weights.sum(axis=1)
                plane_distance = maximum+self.plane_tau*np.log(total/region['counts'][query_frames])
                local_normal = np.einsum('af,afi->ai', weights, query_directions)/total[:, None]
                rotation = region['rotations'][query_frames]
                plane_normal = np.einsum('nij,nj->ni', rotation, local_normal)
                # Plane maxima underestimate Euclidean distance outside thin
                # patch corners.  Intersect with the enclosing AABB using its
                # Euclidean outside distance, so distant patches cannot become
                # false union contacts merely through an oblique plane normal.
                box_d = box_distance[point_ids].copy()
                box_g = normal[point_ids].copy()
                box_probabilities = np.zeros((len(query_frames), 6)) if hessian else None
                outside = box_d > 1e-12
                inside = ~outside
                if np.any(inside):
                    point = flat[point_ids[inside]]
                    box_planes = np.concatenate((point-hi[point_ids[inside]], lo[point_ids[inside]]-point), axis=1)
                    box_max = box_planes.max(axis=1)
                    box_weights = box_planes
                    box_weights -= box_max[:, None]
                    box_weights /= self.plane_tau
                    np.exp(box_weights, out=box_weights)
                    box_total = box_weights.sum(axis=1)
                    box_d[inside] = box_max+self.plane_tau*np.log(box_total/6.)
                    box_g[inside] = (box_weights[:, :3]-box_weights[:, 3:])/box_total[:, None]
                    if hessian:
                        box_probabilities[inside] = box_weights/box_total[:, None]
                both_max = np.maximum(plane_distance, box_d)
                wa = np.exp((plane_distance-both_max)/self.plane_tau)
                wb = np.exp((box_d-both_max)/self.plane_tau)
                both_total = wa+wb; wa /= both_total; wb /= both_total
                distance[point_ids] = both_max+self.plane_tau*np.log(both_total/2.)
                normal[point_ids] = wa[:, None]*plane_normal+wb[:, None]*box_g
                if hessian:
                    wanted = hessian_wanted[point_ids].copy()
                    if hessian_limit is not None:
                        wanted &= distance[point_ids] <= hessian_limit[point_ids]+24.*self.union_tau
                    selected = np.flatnonzero(wanted)
                    if len(selected):
                        selected_directions = query_directions[selected]
                        second = np.einsum('af,afi,afj->aij', weights[selected], selected_directions, selected_directions, optimize=True)/total[selected, None, None]
                        selected_local_normal = local_normal[selected]
                        local_hessian = (second-np.einsum('ni,nj->nij', selected_local_normal, selected_local_normal))/self.plane_tau
                        selected_rotation = rotation[selected]
                        plane_h = selected_rotation@local_hessian@np.swapaxes(selected_rotation, -1, -2)
                        box_h = np.zeros((len(selected), 3, 3))
                        selected_outside = outside[selected]
                        if np.any(selected_outside):
                            ids = selected[selected_outside]
                            selected_vector = box_vector[point_ids[ids]]
                            diagonal = np.zeros((len(ids), 3, 3))
                            diagonal[:, np.arange(3), np.arange(3)] = np.abs(selected_vector)>0.
                            box_h[selected_outside] = (diagonal-np.einsum('ni,nj->nij', box_g[ids], box_g[ids]))/box_d[ids, None, None]
                        selected_inside = ~selected_outside
                        if np.any(selected_inside):
                            ids = selected[selected_inside]; probabilities = box_probabilities[ids]
                            diagonal = np.zeros((len(ids), 3, 3))
                            diagonal[:, np.arange(3), np.arange(3)] = probabilities[:, :3]+probabilities[:, 3:]
                            box_h[selected_inside] = (diagonal-np.einsum('ni,nj->nij', box_g[ids], box_g[ids]))/self.plane_tau
                        difference = plane_normal[selected]-box_g[selected]
                        curvature[point_ids[selected]] = (
                            wa[selected, None, None]*plane_h+wb[selected, None, None]*box_h
                            +wa[selected, None, None]*wb[selected, None, None]*np.einsum('ni,nj->nij', difference, difference)/self.plane_tau)
            if union_distance is None:
                union_distance = distance; union_normal = normal; union_hessian = curvature
                continue
            # An incremental log-sum-exp keeps peak storage independent of the
            # number of slabs, including when Hessians are requested.
            minimum = np.minimum(union_distance, distance)
            weight_a = np.exp((minimum-union_distance)/self.union_tau)
            weight_b = np.exp((minimum-distance)/self.union_tau)
            total = weight_a+weight_b
            weight_a /= total; weight_b /= total
            new_normal = weight_a[..., None]*union_normal+weight_b[..., None]*normal
            if hessian:
                wanted = hessian_wanted.copy()
                if hessian_limit is not None:
                    wanted &= minimum <= hessian_limit+24.*self.union_tau
                # A remote region's exact floating-point softmin weight often
                # underflows to zero.  Its Hessian contribution is then exactly
                # zero and the old Hessian is unchanged (weight_a == 1).  Avoid
                # rebuilding/gathering that dense nine-component array for all
                # points at every body slab; update only nonzero contributions.
                union_hessian[~wanted] = 0.
                ids = np.flatnonzero(wanted & (weight_b > 0.))
                if len(ids):
                    difference = union_normal[ids]-normal[ids]
                    covariance = (weight_a[ids, None, None]*weight_b[ids, None, None]
                                  *np.einsum('ni,nj->nij', difference, difference))
                    union_hessian[ids] = (weight_a[ids, None, None]*union_hessian[ids]
                        +weight_b[ids, None, None]*curvature[ids]-covariance/self.union_tau)
            union_distance = minimum-self.union_tau*np.log(total)
            union_normal = new_normal
        if hessian:
            if hessian_limit is not None:
                union_hessian[union_distance > hessian_limit] = 0.
            return union_distance, union_normal, union_hessian
        return union_distance, union_normal
