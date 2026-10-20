"""Compact, animation-aware palm and finger collision spheres.

This module uses NumPy only; it does not inspect or change a Blender scene.
The original finger channels are retained.  Every sphere centre is the mean
of a fixed set of skinned mesh vertices, so its path is continuous whenever
the original animation is continuous.  Do not refit or recluster per frame.

``build_hand_geometry(geometry, meta, matrices)`` returns ``HandGeometry``.
``original_points`` has shape [frame, side, sphere, xyz] and ``radii`` has
shape [side, sphere].  Side order is L, R.  A solver should move these points
with ``apply_rigid_transform(corrected_wrists, total_world_rotations)``.
The rotation is the accumulated upper-arm, forearm AND wrist correction;
the translation is the resulting wrist origin.  Rotations act on the offset
from the *original* wrist.  No finger corrections are introduced.

Spheres cover the complete selected surface triangles at captured poses.
This is a conservative proxy, not proof of surface separation between keys.
Frame/subframe surface validation remains necessary after baking.
"""
from dataclasses import dataclass
import copy
import numpy as np


def _descendants(parents, root):
    result = {int(root)}
    changed = True
    while changed:
        changed = False
        for child, parent in enumerate(parents):
            if parent in result and child not in result:
                result.add(child)
                changed = True
    return result


def _palm_labels(points, count):
    """Deterministic 2-D Lloyd partition, retaining full palm thickness."""
    xy = np.asarray(points[:, :2], np.float64)
    centers = [xy[np.argmin(np.sum((xy-xy.mean(0))**2, axis=1))]]
    for _ in range(1, count):
        distance = np.min(np.sum((xy[:, None]-np.asarray(centers))**2, axis=2), axis=1)
        centers.append(xy[np.argmax(distance)])
    centers = np.asarray(centers)
    for _ in range(35):
        labels = np.argmin(np.sum((xy[:, None]-centers)**2, axis=2), axis=1)
        updated = centers.copy()
        for j in range(count):
            if np.any(labels == j):
                updated[j] = xy[labels == j].mean(0)
        if np.max(np.abs(updated-centers)) < 1e-10:
            break
        centers = updated
    return labels


def _palm_surface_labels(projected, count):
    """One fixed rest-pose partition with separate palm/back surface cells.

    projected has actual PCA plane coordinates first and thickness last.
    Separating the two surface sheets lets an ellipsoid represent a thin
    patch rather than spanning the complete hand thickness at every cell.
    This split is NEVER recomputed from animated poses.
    """
    if count < 2:
        return _palm_labels(projected, count)
    front = projected[:, 2] >= np.median(projected[:, 2])
    counts = (count//2, count-count//2)
    result = np.empty(len(projected), np.int32)
    for mask, number, offset in ((~front, counts[0], 0),
                                 (front, counts[1], counts[0])):
        if np.count_nonzero(mask) < number:
            return _palm_labels(projected, count)
        result[mask] = _palm_labels(projected[mask], number)+offset
    return result


def _skin(vertices, joint, weight, inverse_rest, poses):
    """Linear blend skin a small frame batch in armature coordinates."""
    transform = poses @ inverse_rest
    result = np.zeros((len(poses), len(vertices), 3), np.float64)
    for k in range(joint.shape[1]):
        m = transform[:, joint[:, k]]
        result += (np.einsum('fvij,vj->fvi', m[:, :, :3, :3], vertices)
                   + m[:, :, :3, 3]) * weight[None, :, k, None]
    return result


def smooth_radius_upper_envelope(required, half_window=4):
    """Smooth in time while retaining a provable per-frame upper bound.

    First dilate by the maximum over +/- ``half_window`` frames, then convolve
    with a positive Gaussian kernel with exactly the same support. Every
    maximum entering frame f's weighted average contains required[f], hence
    their convex combination is also >= required[f].  This avoids an
    after-the-fact maximum with the raw radius, which reintroduces its kinks.
    Edge extension is constant and preserves the same proof at clip ends.
    Only fixed sample trajectories are processed; no temporal reclustering.
    """
    required = np.asarray(required, np.float64)
    if required.ndim < 1 or not len(required):
        raise ValueError('A temporal radius array must contain a frame axis')
    half_window = int(half_window)
    if half_window < 0:
        raise ValueError('Radius smoothing window must be nonnegative')
    if half_window == 0:
        return required.copy()
    pads = ((half_window, half_window),) + ((0, 0),)*(required.ndim-1)
    padded = np.pad(required, pads, mode='edge')
    maximum = np.maximum.reduce([padded[j:j+len(required)]
                                 for j in range(2*half_window+1)])
    kernel_x = np.arange(-half_window, half_window+1, dtype=np.float64)
    sigma = max(half_window*.5, .5)
    kernel = np.exp(-.5*(kernel_x/sigma)**2)
    kernel /= kernel.sum()
    padded = np.pad(maximum, pads, mode='edge')
    smooth = sum(float(w)*padded[j:j+len(required)]
                 for j, w in enumerate(kernel))
    # Numerical roundoff can undershoot by < 1e-17; a constant epsilon keeps
    # the conservative inequality strict without introducing temporal kinks.
    return smooth + np.finfo(np.float64).eps*np.maximum(1., np.max(required))


@dataclass
class HandGeometry:
    original_points: np.ndarray
    radii: np.ndarray
    bone_ids: np.ndarray
    part_ids: np.ndarray
    near_wrist: np.ndarray
    valid: np.ndarray
    wrist_ids: tuple
    wrist_origins: np.ndarray
    sample_names: tuple
    vertex_ids: tuple
    triangles: tuple
    triangle_spheres: tuple
    calibration: dict
    wrist_offsets: np.ndarray
    per_frame_radii: np.ndarray = None
    wrist_weight_fraction: np.ndarray = None
    ellipsoid_axes: np.ndarray = None
    ellipsoid_basis_local: np.ndarray = None
    envelope_base_axes: np.ndarray = None
    envelope_base_radii: np.ndarray = None
    envelope_refit_factors: np.ndarray = None
    envelope_sphere_refit_factors: np.ndarray = None

    @property
    def relative_points(self):
        return self.original_points-self.wrist_origins[:, :, None, :]

    def apply_rigid_transform(self, wrist_positions, transforms):
        """Transform the captured hand without changing finger articulation.

        wrist_positions: [frame, 2, 3]; transforms: [frame, 2, 3, 3].
        Identity transforms and original wrist_positions reproduce the input.
        """
        return (np.asarray(wrist_positions)[:, :, None, :]
                + np.einsum('fsij,fspj->fspi', transforms, self.relative_points))

    def apply_corrections(self, wrist_positions, upstream_rotations, wrist_rotations,
                          wrist_translation_world=None):
        """Move sphere centres with exact wrist skin-weight blending.

        Each rotation has shape [frame, 2, 3, 3].  upstream_rotations is the
        accumulated upper/forearm world correction. wrist_rotations is the
        wrist-only world correction *before* applying upstream_rotations.
        Most hand vertices have 100% wrist-subtree influence and simply move
        rigidly.  At the wrist seam, a small forearm-weight contribution must
        only follow the upstream rotation; otherwise twisting the wrist moves
        the predicted seam too far.  This decomposition handles that case
        exactly for fixed linear skin weights at every sample centre.

        wrist_positions is W0, the wrist origin after only upstream arm
        corrections. Optional wrist_translation_world [frame, 2, 3] is a
        separate wrist-only displacement expressed in the final armature
        coordinate system, after upstream rotation. Do not include this
        displacement in W0 as well. Only ``wrist_weight_fraction`` of that
        translation is added to a blended centre; the forearm-weight part
        remains fixed. If a caller already has the fully translated wrist
        position, subtract wrist_translation_world from it to obtain W0.
        """
        rotated = np.einsum('fsij,fspj->fspi', wrist_rotations, self.wrist_offsets)
        relative = self.relative_points-self.wrist_offsets+rotated
        result = (np.asarray(wrist_positions)[:, :, None, :]
                  + np.einsum('fsij,fspj->fspi', upstream_rotations, relative))
        if wrist_translation_world is not None:
            if self.wrist_weight_fraction is None:
                raise ValueError('Wrist translation needs wrist subtree weight fractions')
            result += (self.wrist_weight_fraction[None, :, :, None]
                       * np.asarray(wrist_translation_world)[:, :, None, :])
        return result

    def own_forearm_pairs(self, forearm_t, cutoff=.80):
        """Allowed hand/own-forearm contacts, shape [side, sphere, t].

        Exclude only the palm spheres adjoining the wrist and the last 20%
        of their own forearm.  Those surfaces have a permanent anatomical
        connection; penalising them would force an artificial wrist gap.
        Fingers, all upper-arm contacts, body contacts and opposite-side
        arm/hand contacts remain enabled.  Do not use this mask between sides.
        Same-hand sphere pairs must also be skipped: the original finger pose
        is preserved and adjacent skin is intentionally connected.
        """
        return (self.valid[:, :, None]
                & ~(self.near_wrist[:, :, None]
                    & (np.asarray(forearm_t)[None, None, :] >= cutoff)))


def build_hand_geometry(geometry, meta, matrices, palm_spheres=12,
                        finger_spheres=3, radius_padding=.00035,
                        batch_size=24, progress=None, dynamic_radii=True,
                        radius_smoothing_window=4, ellipsoids=True,
                        palm_surface_layers=True):
    """Fit tight sphere covers to the original animated hand surfaces.

    Geometry keys are ``vertices,joint,weight,dominant,rest,triangles``.
    Meta needs ``bones,parents`` and optionally ``wrist_ids``.  Materials are
    not filtered: nails and any other skin attached to hand bones participate.
    A hand bone with dominant vertices becomes a finger region; nondeforming
    control/helper bones do not create duplicate spheres. Palm uses 12 fixed
    cells on its two surface sheets (6 palm/6 back by default), projected onto
    the actual two largest whole-palm PCA axes. Set palm_surface_layers=False
    to retain a single set of 12 cells spanning both sheets. Every deforming
    finger segment uses 3 longitudinal cells
    by default (57 spheres per hand for a standard 15-segment MMD hand).

    Radius is the maximum over *all provided frames* and all corners of each
    assigned triangle.  Thus every selected triangle lies in one sphere at
    every captured pose, including finger bends and skin-weight blending.
    Small padding absorbs roundoff, not a large hand-wide clearance margin.
    Provide the complete solve interval here; fitting a short test clip and
    reusing its radii for a different articulation is not conservative.
    ``radii`` retains the compatible, static worst-case [2, sphere] array.
    By default ``per_frame_radii`` additionally contains a tighter temporally
    smooth upper cover [frame, 2, sphere]. Set ``dynamic_radii=False`` to omit
    that array. A +/-4-frame dilation followed by a positive Gaussian filter
    preserves coverage while easing local radius changes. This parameter is
    measured in captured samples (normally one sample per animation frame).
    With ``ellipsoids=True``, palm cells additionally receive conservative
    oriented ellipsoids. ``ellipsoid_basis_local[2, sphere, 3, 3]`` stores
    fixed basis COLUMN vectors in original wrist-bone local XYZ. Their world
    orientation is ``original_or_corrected_wrist_orientation @ local_basis``.
    ``ellipsoid_axes[frame, 2, sphere, 3]`` stores positive half-axis lengths.
    Palm aspect ratios are fixed and a single smooth scalar enlarges all
    three axes per frame to cover each assigned surface corner. Fingers stay
    spheres, with identity basis and three equal dynamic radii.
    """
    vertices = np.asarray(geometry['vertices'], np.float64)
    joint = np.asarray(geometry['joint'], np.int32)
    weight = np.asarray(geometry['weight'], np.float64)
    dominant = np.asarray(geometry['dominant'], np.int32)
    rest = np.asarray(geometry['rest'], np.float64)
    triangles = np.asarray(geometry['triangles'], np.int32)
    matrices = np.asarray(matrices, np.float64)
    names, parents = meta['bones'], meta['parents']
    wrists = tuple(meta.get('wrist_ids') or
                   (names.index('手首.L'), names.index('手首.R')))
    if len(wrists) != 2:
        raise ValueError('Hand collision geometry needs two wrist bone IDs')
    if not len(matrices):
        raise ValueError('Hand collision geometry needs at least one pose')
    inverse_rest = np.linalg.inv(rest)
    specs = []
    for side, wrist in enumerate(wrists):
        subtree = _descendants(parents, wrist)
        is_hand = np.isin(dominant, list(subtree))
        selected_tri = triangles[np.sum(is_hand[triangles], axis=1) >= 2]
        if len(selected_tri) < 20:
            raise ValueError('Too few hand triangles for ' + names[wrist])
        selected_ids = np.unique(selected_tri)
        primary_groups, owners, parts, labels = [], [], [], []
        wrist_ids = np.flatnonzero(dominant == wrist)
        if len(wrist_ids) < palm_spheres:
            raise ValueError('Too few palm vertices for ' + names[wrist])
        wrist_local = ((vertices[wrist_ids]-rest[wrist, :3, 3])
                       @ rest[wrist, :3, :3])
        palm_centered = wrist_local-wrist_local.mean(0)
        palm_eigenvalues, palm_basis = np.linalg.eigh(palm_centered.T@palm_centered)
        palm_normal = palm_basis[:, 0]
        # Bone roll is arbitrary: determine the actual palm plane from the
        # two widest principal directions rather than assuming local XY.
        palm_projected = palm_centered @ palm_basis[:, [2, 1, 0]]
        assignment = (_palm_surface_labels(palm_projected, palm_spheres)
                      if palm_surface_layers else _palm_labels(palm_projected, palm_spheres))
        for label in range(palm_spheres):
            ids = wrist_ids[assignment == label]
            if len(ids):
                primary_groups.append(ids)
                owners.append(wrist)
                parts.append(0)
                labels.append(names[wrist] + ':palm:' + str(label))
        finger_bones = sorted(b for b in subtree if b != wrist
                              and np.count_nonzero(dominant == b) >= finger_spheres)
        # Follow each deforming segment's ancestors to its first finger root.
        finger_roots = []
        for bone in finger_bones:
            root = bone
            while parents[root] != wrist and parents[root] in subtree:
                root = parents[root]
            finger_roots.append(root)
        root_ids = {b: i+1 for i, b in enumerate(sorted(set(finger_roots)))}
        for bone, root in zip(finger_bones, finger_roots):
            ids = np.flatnonzero(dominant == bone)
            local_y = (vertices[ids]-rest[bone, :3, 3]) @ rest[bone, :3, 1]
            spread = max(float(np.ptp(local_y)), 1e-10)
            assignments = np.minimum(((local_y-local_y.min())/spread*finger_spheres)
                                     .astype(np.int32), finger_spheres-1)
            for label in range(finger_spheres):
                group = ids[assignments == label]
                if len(group):
                    primary_groups.append(group)
                    owners.append(bone)
                    parts.append(root_ids[root])
                    labels.append(names[bone] + ':' + str(label))
        centers = np.array([vertices[ids].mean(0) for ids in primary_groups])
        # Whole triangles go to one sphere. Covering all three corners by the
        # same convex sphere also covers the entire intervening surface.
        centroids = vertices[selected_tri].mean(1)
        distances = np.sum((centroids[:, None]-centers)**2, axis=2)
        # Anatomical ownership matters at a knuckle: a finger-root triangle
        # can be spatially closer to a palm sphere in the rest pose, but then
        # move centimetres away when the finger curls.  Prefer the triangle's
        # majority deforming bone so that one palm sphere does not inflate to
        # enclose another articulated region over the complete clip.
        ownership = np.sum(dominant[selected_tri][:, None, :]
                           == np.asarray(owners)[None, :, None], axis=2)
        best_ownership = ownership.max(1, keepdims=True)
        allowed = ownership == best_ownership
        distances = np.where(allowed, distances, np.inf)
        triangle_assignment = np.argmin(distances, axis=1)
        fitting_groups = [np.unique(selected_tri[triangle_assignment == j])
                          for j in range(len(centers))]
        # Keep groups with no nearest triangle alive for reliable sample shape.
        fitting_groups = [group if len(group) else primary_groups[j]
                          for j, group in enumerate(fitting_groups)]
        all_ids = np.unique(np.concatenate([selected_ids] + primary_groups))
        remap = np.full(len(vertices), -1, np.int32)
        remap[all_ids] = np.arange(len(all_ids))
        primary_local = [remap[ids] for ids in primary_groups]
        fitting_local = [remap[ids] for ids in fitting_groups]
        bases = np.broadcast_to(np.eye(3), (len(centers), 3, 3)).copy()
        base_axes = np.ones((len(centers), 3), np.float64)
        near_wrist = ((np.asarray(parts) == 0)
                      & (np.linalg.norm(centers-rest[wrist, :3, 3], axis=1) < .035))
        specs.append(dict(ids=all_ids, primary=primary_local, fitting=fitting_local,
                          owners=owners, parts=parts, labels=labels,
                          near_wrist=near_wrist, triangles=selected_tri,
                          triangle_assignment=triangle_assignment,
                          subtree=subtree, bases=bases, base_axes=base_axes,
                          palm_normal=palm_normal,
                          palm_pca_spans=np.ptp(palm_projected, axis=0),
                          palm_clouds={j: [] for j, part in enumerate(parts) if part == 0}))
    count = max(len(s['primary']) for s in specs)
    points = np.zeros((len(matrices), 2, count, 3), np.float64)
    wrist_offsets = np.zeros_like(points)
    wrist_weight_fraction = np.zeros((2, count), np.float64)
    radii = np.zeros((2, count), np.float64)
    frame_required = np.zeros((len(matrices), 2, count), np.float64)
    ellipsoid_scale = np.zeros_like(frame_required) if ellipsoids else None
    ellipsoid_required_normal = np.zeros_like(frame_required) if ellipsoids else None
    ellipsoid_base_axes = np.ones((2, count, 3), np.float64)
    ellipsoid_basis_local = np.broadcast_to(np.eye(3), (2, count, 3, 3)).copy() if ellipsoids else None
    valid = np.zeros((2, count), bool)
    bone_ids = np.full((2, count), -1, np.int32)
    part_ids = np.full((2, count), -1, np.int32)
    near_wrist = np.zeros((2, count), bool)
    for side, s in enumerate(specs):
        ids = s['ids']
        number = len(s['primary'])
        valid[side, :number] = True
        bone_ids[side, :number] = s['owners']
        part_ids[side, :number] = s['parts']
        near_wrist[side, :number] = s['near_wrist']
        if ellipsoids:
            ellipsoid_base_axes[side, :number] = s['base_axes']
            ellipsoid_basis_local[side, :number] = s['bases']
        hand_weight = weight[ids]*np.isin(joint[ids], list(s['subtree']))
        cluster_hand_weight = [hand_weight[primary].sum(1).mean()
                               for primary in s['primary']]
        wrist_weight_fraction[side, :number] = cluster_hand_weight
        for start in range(0, len(matrices), batch_size):
            end = min(start+batch_size, len(matrices))
            posed = _skin(vertices[ids], joint[ids], weight[ids], inverse_rest,
                          matrices[start:end])
            wrist_part = _skin(vertices[ids], joint[ids], hand_weight, inverse_rest,
                               matrices[start:end])
            if ellipsoids:
                wrist_rotation = matrices[start:end, wrists[side], :3, :3].copy()
                wrist_rotation /= np.linalg.norm(wrist_rotation, axis=-2, keepdims=True)
            for j, (primary, fitting) in enumerate(zip(s['primary'], s['fitting'])):
                centers = posed[:, primary].mean(1)
                points[start:end, side, j] = centers
                wrist_offsets[start:end, side, j] = (wrist_part[:, primary].mean(1)
                    - cluster_hand_weight[j]*matrices[start:end, wrists[side], :3, 3])
                radius = np.linalg.norm(posed[:, fitting]-centers[:, None], axis=-1).max(1)
                frame_required[start:end, side, j] = radius
                radii[side, j] = max(radii[side, j], radius.max())
                if ellipsoids and s['parts'][j] == 0:
                    # Row-vector coordinates in the posed wrist's local
                    # frame, followed by the fixed local principal basis.
                    relative = posed[:, fitting]-centers[:, None]
                    local = np.einsum('fvi,fij->fvj', relative, wrist_rotation)
                    s['palm_clouds'][j].append(local)
            if progress:
                progress('hands', side*len(matrices)+end, 2*len(matrices))
        if number < count:
            points[:, side, number:] = matrices[:, wrists[side], None, :3, 3]
        if ellipsoids:
            # Fitting only rest-pose aspect ratios can amplify a small
            # articulated tangential deviation into unnecessary palm depth.
            # Fit one fixed basis/aspect to a coarse temporal coreset, then
            # calibrate its scalar against ALL frames and ALL surface corners.
            for j in list(s['palm_clouds']):
                cloud = np.concatenate(s['palm_clouds'].pop(j), axis=0)
                # The normal of the complete palm, rather than the shortest
                # axis of a tiny cell, identifies anatomical palm thickness.
                # A cell's shortest PCA axis can instead run tangentially
                # across that cell. Keep the common thickness axis explicit.
                normal = s['palm_normal']
                ellipsoid_required_normal[:, side, j] = np.abs(cloud@normal).max(1)
                tangent = np.eye(3)[np.argmin(np.abs(normal))]
                tangent = tangent-normal*np.dot(tangent, normal)
                tangent /= np.linalg.norm(tangent)
                bitangent = np.cross(normal, tangent)
                basis = np.column_stack((tangent, bitangent, normal))
                principal_coreset = (cloud[::8]@basis)
                xy2 = np.sum(principal_coreset[:, :, :2]**2, axis=2)
                z2 = principal_coreset[:, :, 2]**2
                sphere_support = np.sqrt(xy2+z2).max(1)
                # Search a bounded family with equal tangential axes. Every
                # candidate's normal support is <= the enclosing sphere's:
                # a*sqrt(xy^2+z^2/a^2) <= sqrt(xy^2+z^2), 0<a<=1.
                # Penalise tangential growth to avoid trading a slightly
                # smaller depth for an excessively broad hand silhouette.
                candidates = np.linspace(.25, 1., 31)
                costs = []
                for aspect in candidates:
                    support = np.sqrt(xy2+z2/(aspect*aspect)).max(1)
                    cost = np.mean(aspect*support+.4*(support-sphere_support))
                    # Never trade a flatter palm for a huge lateral envelope.
                    if np.max(support/np.maximum(sphere_support, 1e-12)) > 1.35:
                        cost = np.inf
                    costs.append(cost)
                aspect = float(candidates[np.argmin(costs)])
                principal = cloud@basis
                # The coreset chooses a useful aspect, but the lateral
                # growth cap must hold at every pose, including short events
                # between coreset samples. Solve its exact lower bound on a:
                # xy^2+z^2/a^2 <= (1.35*r_sphere)^2 at every fitted corner.
                xy_square = np.sum(principal[:, :, :2]**2, axis=-1)
                z_square = principal[:, :, 2]**2
                limit_square = (1.35*frame_required[:, side, j])**2
                minimum_aspect = np.sqrt(np.max(z_square/np.maximum(limit_square[:, None]-xy_square, 1e-30)))
                aspect = min(1., max(aspect, float(minimum_aspect)))
                axes = np.array([1., 1., aspect])
                scale = np.sqrt(np.sum((principal/axes)**2, axis=-1)).max(1)
                ellipsoid_scale[:, side, j] = scale
                ellipsoid_base_axes[side, j] = axes
                ellipsoid_basis_local[side, j] = basis
                s['bases'][j] = basis
    radii[valid] += radius_padding
    frame_required[:, valid] += radius_padding
    per_frame_radii = (smooth_radius_upper_envelope(frame_required, radius_smoothing_window)
                       if dynamic_radii else None)
    if per_frame_radii is not None:
        per_frame_radii[:, ~valid] = 0.
    ellipsoid_axes = None
    if ellipsoids:
        smooth_scale = smooth_radius_upper_envelope(ellipsoid_scale, radius_smoothing_window)
        ellipsoid_axes = smooth_scale[..., None]*ellipsoid_base_axes[None]
        # Adding padding to every half axis keeps the larger concentric
        # ellipsoid conservative and avoids a zero-thickness numerical case.
        ellipsoid_axes += radius_padding
        finger = (part_ids > 0) & valid
        sphere_radii = per_frame_radii if per_frame_radii is not None else np.broadcast_to(radii, frame_required.shape)
        ellipsoid_axes[:, finger] = sphere_radii[:, finger, None]
        ellipsoid_axes[:, ~valid] = 0.
    calibration = {
        'frames': len(matrices),
        'spheres_per_hand': [int(s.sum()) for s in valid],
        'triangles_per_hand': [len(s['triangles']) for s in specs],
        'radius_min': float(radii[valid].min()),
        'radius_max': float(radii[valid].max()),
        'radius_padding': float(radius_padding),
        'coverage': 'Each assigned hand triangle lies inside one sphere at each captured pose',
        'near_wrist_spheres': [int(s.sum()) for s in near_wrist],
        'same_side_forearm_cutoff': .80,
        'palm_partition_surface_layers': bool(palm_surface_layers),
    }
    if per_frame_radii is not None:
        active = per_frame_radii[:, valid]
        static = np.broadcast_to(radii, frame_required.shape)[:, valid]
        calibration.update({
            'dynamic_radius_smoothing_half_window': int(radius_smoothing_window),
            'dynamic_radius_mean': float(active.mean()),
            'dynamic_radius_p95': float(np.quantile(active, .95)),
            'static_radius_mean': float(static.mean()),
            'dynamic_radius_mean_reduction': float((static-active).mean()),
            'dynamic_radius_mean_reduction_fraction': float(1-active.mean()/static.mean()),
            'dynamic_radius_minimum_coverage_slack': float((per_frame_radii-frame_required)[:, valid].min()),
            'raw_radius_maximum_frame_step': float(np.max(np.abs(np.diff(frame_required[:, valid], axis=0)))) if len(matrices)>1 else 0.,
            'dynamic_radius_maximum_frame_step': float(np.max(np.abs(np.diff(active, axis=0)))) if len(matrices)>1 else 0.,
            'raw_radius_maximum_second_difference': float(np.max(np.abs(np.diff(frame_required[:, valid], n=2, axis=0)))) if len(matrices)>2 else 0.,
            'dynamic_radius_maximum_second_difference': float(np.max(np.abs(np.diff(active, n=2, axis=0)))) if len(matrices)>2 else 0.,
        })
    if ellipsoid_axes is not None:
        palm = (part_ids == 0) & valid
        thin_support = ellipsoid_axes[:, palm].min(-1)
        normal_projection = np.zeros((2, count, 3), np.float64)
        for side, s in enumerate(specs):
            number = len(s['primary'])
            normal_projection[side, :number] = np.einsum('pij,i->pj', s['bases'], s['palm_normal'])
        thickness_support = np.sqrt(np.sum((ellipsoid_axes[:, palm]*normal_projection[palm])**2, axis=-1))
        old_support = (per_frame_radii if per_frame_radii is not None else np.broadcast_to(radii, frame_required.shape))[:, palm]
        calibration.update({
            'palm_ellipsoid_thin_support_mean': float(thin_support.mean()),
            'palm_ellipsoid_thin_support_p95': float(np.quantile(thin_support, .95)),
            'palm_sphere_support_mean': float(old_support.mean()),
            'palm_thin_support_mean_reduction_fraction': float(1-thin_support.mean()/old_support.mean()),
            'palm_ellipsoid_thickness_support_mean': float(thickness_support.mean()),
            'palm_ellipsoid_thickness_support_p95': float(np.quantile(thickness_support, .95)),
            'palm_thickness_support_mean_reduction_fraction': float(1-thickness_support.mean()/old_support.mean()),
            'palm_thickness_normals_wrist_local': [s['palm_normal'].tolist() for s in specs],
            'palm_pca_spans_descending': [s['palm_pca_spans'].tolist() for s in specs],
            'palm_required_thickness_support_mean': float(ellipsoid_required_normal[:, palm].mean()),
            'palm_required_thickness_support_p95': float(np.quantile(ellipsoid_required_normal[:, palm], .95)),
            'palm_ellipsoid_tangent_support_growth_cap': 1.35,
        })
    return HandGeometry(points, radii, bone_ids, part_ids, near_wrist, valid,
                        wrists, np.take(matrices, wrists, axis=1)[:, :, :3, 3],
                        tuple(tuple(s['labels']) for s in specs),
                        tuple(s['ids'] for s in specs),
                        tuple(s['triangles'] for s in specs),
                        tuple(s['triangle_assignment'] for s in specs), calibration,
                        wrist_offsets, per_frame_radii, wrist_weight_fraction,
                        ellipsoid_axes, ellipsoid_basis_local)


def _rotation_vectors(vectors):
    """NumPy-only SO(3) exponential, matching the model's local corrections."""
    vectors = np.asarray(vectors, np.float64)
    theta = np.linalg.norm(vectors, axis=-1)
    k = np.zeros(vectors.shape[:-1]+(3, 3), np.float64)
    k[..., 0, 1] = -vectors[..., 2]
    k[..., 0, 2] = vectors[..., 1]
    k[..., 1, 0] = vectors[..., 2]
    k[..., 1, 2] = -vectors[..., 0]
    k[..., 2, 0] = -vectors[..., 1]
    k[..., 2, 1] = vectors[..., 0]
    return (np.eye(3)+np.sinc(theta/np.pi)[..., None, None]*k
            +(.5*np.sinc(theta/(2*np.pi))**2)[..., None, None]*(k@k))


def refit_corrected_hand_envelopes(hand, model, delta, geometry,
                                   meta=None, matrices=None,
                                   smoothing_window=4, batch_size=24,
                                   progress=None):
    """Freeze tighter conservative hand envelopes around a candidate motion.

    Returns ``(new_hand, report)`` without mutating hand/model/geometry or any
    Blender scene. No files or shared caches are read or written here.
    ``geometry`` supplies vertices/joint/weight/rest in the same indexing as
    hand.triangles. ``meta`` and ``matrices`` default to model.meta/matrices.
    They must describe the same captured frames as hand (slice the full clip
    BEFORE calling this function).

    Model protocol is FullLimbModel: state(delta), rot, target_ids, shoulders,
    arm_count, meta, matrices. The six corrected target IDs are ordered
    upper.L/fore.L/wrist.L/upper.R/fore.R/wrist.R. delta's first six vectors
    are their local rotation corrections. model.state supplies candidate
    centres/orientations and corrected elbow/wrist origins, including the
    independent wrist translations. Each original pose bone is transformed
    by its nearest corrected ancestor; fixed vertex weights then generate
    candidate LBS positions, including the wrist seam's mixed contributions.
    This follows the model's FK/LBS contract; actual Blender constraints,
    modifiers and subframe geometry must still pass final surface validation.

    Centres and basis directions never change. Every complete assigned
    triangle is enclosed by a single ellipsoid after a local uniform factor
    >=1; the same positive-kernel temporal upper envelope smooths that factor.
    Compatible sphere radii are independently refitted to those same corners.
    ``envelope_base_axes`` and ``envelope_base_radii`` retain the ORIGINAL
    pre-refit baseline. Repeated calls always recompute relative to it rather
    than repeatedly multiplying already enlarged axes. A returned hand can
    therefore be frozen into a second optimization pass; further candidate
    changes require another refit or the final real-surface validation gate.
    """
    metadata = model.meta if meta is None else meta
    poses = np.asarray(model.matrices if matrices is None else matrices, np.float64)
    x = np.asarray(delta, np.float64)
    frame_count, _, sample_count, _ = hand.original_points.shape
    if len(poses) != frame_count or x.shape[0] != frame_count:
        raise ValueError('Hand, matrices and correction must contain the same captured frames')
    targets = list(model.target_ids)
    if len(targets) != 6 or x.ndim != 3 or x.shape[1] < 6 or x.shape[2] != 3:
        raise ValueError('Hand refit needs six ordered limb targets and [frame, parameter, 3] corrections')
    if len(metadata['parents']) != poses.shape[1]:
        raise ValueError('Bone metadata and pose matrices do not match')
    valid = np.asarray(hand.valid, bool)
    saved_axes = getattr(hand, 'envelope_base_axes', None)
    saved_radii = getattr(hand, 'envelope_base_radii', None)
    source_radii = getattr(hand, 'per_frame_radii', None)
    if source_radii is None:
        source_radii = np.broadcast_to(hand.radii, (frame_count,)+hand.radii.shape)
    base_radii = np.array(source_radii if saved_radii is None else saved_radii,
                          np.float64, copy=True)
    source_axes = getattr(hand, 'ellipsoid_axes', None)
    if source_axes is None:
        source_axes = np.repeat(base_radii[..., None], 3, axis=-1)
    base_axes = np.array(source_axes if saved_axes is None else saved_axes,
                         np.float64, copy=True)
    if base_axes.shape != (frame_count, 2, sample_count, 3) or base_radii.shape != (frame_count, 2, sample_count):
        raise ValueError('Stored hand envelope baseline does not match the captured frames')
    if np.any(base_axes[:, valid] <= 0) or np.any(base_radii[:, valid] <= 0):
        raise ValueError('Valid hand samples need positive envelope axes and radii')
    vertices = np.asarray(geometry['vertices'], np.float64)
    joint = np.asarray(geometry['joint'], np.int32)
    weight = np.asarray(geometry['weight'], np.float64)
    inverse_rest = np.linalg.inv(np.asarray(geometry['rest'], np.float64))
    points, orientation, elbows, wrists = model.state(x)
    offset = int(model.arm_count)
    centers = np.asarray(points[:, :, offset:offset+sample_count], np.float64)
    bases = np.asarray(orientation[:, :, offset:offset+sample_count], np.float64)
    if centers.shape != hand.original_points.shape:
        raise ValueError('The model hand sample count/order does not match HandGeometry')
    target_set = set(targets)
    closest = np.full(poses.shape[1], -1, np.int32)
    for bone in range(len(closest)):
        ancestor = bone
        while ancestor >= 0 and ancestor not in target_set:
            ancestor = metadata['parents'][ancestor]
        if ancestor in target_set:
            closest[bone] = targets.index(ancestor)
    affected = np.flatnonzero(closest >= 0)
    local = _rotation_vectors(x[:, :6])
    original_rotations = np.asarray(model.rot, np.float64)
    world = original_rotations@local@original_rotations.swapaxes(-1, -2)
    upper = world[:, (0, 3)]
    upstream = upper@world[:, (1, 4)]
    hand_rotation = upstream@world[:, (2, 5)]
    corrections = np.empty((frame_count, 6, 3, 3), np.float64)
    origins = np.empty((frame_count, 6, 3), np.float64)
    corrections[:, (0, 3)] = upper
    corrections[:, (1, 4)] = upstream
    corrections[:, (2, 5)] = hand_rotation
    origins[:, (0, 3)] = model.shoulders
    origins[:, (1, 4)] = elbows
    origins[:, (2, 5)] = wrists
    original_targets = np.take(poses, targets, axis=1)
    changed_targets = original_targets.copy()
    changed_targets[:, :, :3, :3] = corrections@original_targets[:, :, :3, :3]
    changed_targets[:, :, :3, 3] = origins
    transforms = changed_targets@np.linalg.inv(original_targets)
    raw_metric = np.zeros((frame_count, 2, sample_count), np.float64)
    raw_sphere_metric = np.zeros_like(raw_metric)
    fitted_corner_counts = np.zeros((2, sample_count), np.int32)
    # Work by fixed triangle partitions, so this includes each triangle's
    # complete surface rather than just its centroid or a sparse vertex set.
    for side in range(2):
        triangles = np.asarray(hand.triangles[side], np.int32)
        assignment = np.asarray(hand.triangle_spheres[side], np.int32)
        ids = np.unique(triangles)
        if np.any(ids < 0) or np.any(ids >= len(vertices)):
            raise ValueError('Hand triangle indices do not match supplied geometry')
        remap = np.full(len(vertices), -1, np.int32)
        remap[ids] = np.arange(len(ids))
        fitting = [remap[np.unique(triangles[assignment == j])]
                   for j in range(sample_count)]
        fitted_corner_counts[side] = [len(group) for group in fitting]
        for start in range(0, frame_count, batch_size):
            end = min(start+batch_size, frame_count)
            changed = poses[start:end].copy()
            changed[:, affected] = transforms[start:end, closest[affected]]@changed[:, affected]
            actual = _skin(vertices[ids], joint[ids], weight[ids], inverse_rest, changed)
            for j, group in enumerate(fitting):
                if not valid[side, j] or not len(group):
                    continue
                relative = actual[:, group]-centers[start:end, side, j, None]
                principal = np.einsum('fvi,fij->fvj', relative, bases[start:end, side, j])
                metric = np.sqrt(np.sum((principal/base_axes[start:end, side, j, None])**2, axis=-1)).max(1)
                raw_metric[start:end, side, j] = metric
                raw_sphere_metric[start:end, side, j] = (
                    np.linalg.norm(relative, axis=-1).max(1)/base_radii[start:end, side, j])
            if progress:
                progress('refit_hands', side*frame_count+end, 2*frame_count)
    required = np.maximum(1., raw_metric)
    required_sphere = np.maximum(1., raw_sphere_metric)
    factors = smooth_radius_upper_envelope(required, smoothing_window)
    sphere_factors = smooth_radius_upper_envelope(required_sphere, smoothing_window)
    factors[:, ~valid] = 1.
    sphere_factors[:, ~valid] = 1.
    new_hand = copy.copy(hand)
    new_hand.envelope_base_axes = base_axes
    new_hand.envelope_base_radii = base_radii
    new_hand.envelope_refit_factors = factors
    new_hand.envelope_sphere_refit_factors = sphere_factors
    new_hand.ellipsoid_axes = base_axes*factors[..., None]
    new_hand.per_frame_radii = base_radii*sphere_factors
    new_hand.radii = new_hand.per_frame_radii.max(0)
    # For a spherical input, make the inherited identity basis explicit.
    if getattr(new_hand, 'ellipsoid_basis_local', None) is None:
        new_hand.ellipsoid_basis_local = np.broadcast_to(np.eye(3), (2, sample_count, 3, 3)).copy()
    clusters = []
    for side in range(2):
        for j in range(sample_count):
            if not valid[side, j]:
                continue
            name = hand.sample_names[side][j] if j < len(hand.sample_names[side]) else str(j)
            clusters.append({
                'side': side, 'sample': j, 'name': name,
                'fitted_corner_count': int(fitted_corner_counts[side, j]),
                'required_factor_max': float(required[:, side, j].max()),
                'required_factor_p95': float(np.quantile(required[:, side, j], .95)),
                'frozen_factor_max': float(factors[:, side, j].max()),
                'frozen_factor_p95': float(np.quantile(factors[:, side, j], .95)),
            })
    post_metric = raw_metric[:, valid]/factors[:, valid]
    post_sphere = raw_sphere_metric[:, valid]/sphere_factors[:, valid]
    report = {
        'frames': frame_count,
        'smoothing_half_window': int(smoothing_window),
        'baseline_retained': saved_axes is not None,
        'required_factor_max': float(required[:, valid].max()),
        'required_factor_p95': float(np.quantile(required[:, valid], .95)),
        'frozen_factor_max': float(factors[:, valid].max()),
        'frozen_factor_p95': float(np.quantile(factors[:, valid], .95)),
        'enlarged_sample_count': int(np.sum(np.any(required > 1.+1e-8, axis=0) & valid)),
        'maximum_pre_refit_quadratic': float(np.max(raw_metric[:, valid]**2)),
        'maximum_post_refit_quadratic': float(np.max(post_metric**2)),
        'maximum_post_refit_normalized_excess': float(np.maximum(post_metric-1., 0.).max()),
        'maximum_post_refit_sphere_normalized_excess': float(np.maximum(post_sphere-1., 0.).max()),
        'clusters': clusters,
        'scope': 'Captured candidate FK/LBS poses and fixed assigned complete triangles; final Blender mesh/subframe validation is still required',
    }
    new_hand.calibration = dict(hand.calibration)
    new_hand.calibration['corrected_envelope_refit'] = {k: v for k, v in report.items() if k != 'clusters'}
    return new_hand, report
