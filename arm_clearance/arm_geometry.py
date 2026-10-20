"""Rest-skin cross sections for directional arm collision support.

This module is NumPy-only and does not read or change a Blender scene.  Each
sample is centred on the same arm centreline as ``ClipModel``.  Its transverse
ellipse encloses the selected rest-skin transverse projections by default; it is not a claim
that a moving skinned surface is enclosed between those samples.

``fit_arm_ellipses`` returns a dict with these integration fields:
  radii_axes: [2, 23, 3], semi-axes (major transverse, axial, minor transverse).
  basis_local: [2, 23, 3, 3], columns are those axes in the segment bone's local
      XYZ coordinates.  Apply the *corrected* bone world rotation to this basis.
  bone_ids / segment_ids: [2, 23], the corresponding bone / upper(0), fore(1).
  t: [23], the existing .34..1 upper and .04...99 forearm sample parameters.

For a unit contact normal n in the segment bone's local coordinates, support
is sqrt(sum((basis_local.T @ n * radii_axes)**2)).  For self contact, use this
expression separately for the two arms along opposite world contact normals,
then add their supports and the chosen clearance.  Using a circular maximum
instead would discard the improvement from fitting these sections.
"""

import numpy as np


def ellipsoid_support_radius(direction_local, radii_axes, basis_local=None):
    """Directional support, broadcasting over frames or other leading axes.

    Input directions need not be normalized.  The returned quantity is the
    support for a unit direction; a zero direction returns zero.  With the
    standard fitted shapes, direction_local can be [N, 2, 23, 3].  To query all
    left/right pairs, insert singleton sample axes into each fitted array.
    """
    direction = np.asarray(direction_local, dtype=np.float64)
    length = np.linalg.norm(direction, axis=-1, keepdims=True)
    unit = direction / np.maximum(length, 1e-15)
    if basis_local is not None:
        unit = np.einsum('...ij,...i->...j', np.asarray(basis_local), unit)
    return np.sqrt(np.sum((unit * np.asarray(radii_axes)) ** 2, axis=-1))


def _centred_enclosing_ellipse(points, tolerance=1e-3, iterations=400):
    """Minimum-area origin-centred ellipse, with explicit final enclosure.

    The centred Khachiyan dual uses dimension 2 (no free centre coordinate).
    The final expansion compensates for early convergence or roundoff, so the
    numerical optimization tolerance cannot silently leave vertices outside.
    """
    points = np.asarray(points, dtype=np.float64)
    weights = np.full(len(points), 1. / len(points))
    scale = max(float(np.max(np.linalg.norm(points, axis=1))), 1e-8)
    normalized = points / scale
    covariance = normalized.T @ (normalized * weights[:, None])
    for _ in range(iterations):
        covariance = normalized.T @ (normalized * weights[:, None])
        inverse = np.linalg.inv(covariance + np.eye(2) * 1e-12)
        leverage = np.einsum('ni,ij,nj->n', normalized, inverse, normalized)
        index = int(np.argmax(leverage)); maximum = float(leverage[index])
        if maximum <= 2. * (1. + tolerance):
            break
        step = (maximum - 2.) / (2. * (maximum - 1.))
        weights *= 1. - step
        weights[index] += step
    values, vectors = np.linalg.eigh(2. * covariance)
    order = np.argsort(values)[::-1]
    axes = np.sqrt(np.maximum(values[order], 1e-12)) * scale
    vectors = vectors[:, order]
    rho = np.sqrt(np.sum(((points @ vectors) / axes) ** 2, axis=1))
    axes *= max(1., float(rho.max())) * (1. + 1e-9)
    return axes, vectors


def _ellipse(points, coverage_percentile):
    """Fit a robust core, then expand to the requested empirical coverage."""
    if coverage_percentile >= 100.:
        axes, vectors = _centred_enclosing_ellipse(points)
    else:
        covariance = points.T @ points / len(points)
        inv = np.linalg.inv(covariance + np.eye(2) * 1e-14)
        radius = np.einsum('ni,ij,nj->n', points, inv, points)
        core = points[radius <= np.percentile(radius, coverage_percentile)]
        axes, vectors = _centred_enclosing_ellipse(core)
        rho = np.sqrt(np.sum(((points @ vectors) / axes) ** 2, axis=1))
        axes *= float(np.percentile(rho, coverage_percentile)) * (1. + 1e-9)
    projected = points @ vectors
    rho = np.sqrt(np.sum((projected / axes) ** 2, axis=1))
    # Radial overrun is measured along the ray from the fixed centreline.
    # It is an upper bound for shortest Euclidean point-to-ellipse distance.
    overrun = np.linalg.norm(points, axis=1) * np.maximum(1. - 1. / np.maximum(rho, 1e-15), 0.)
    return axes, vectors, rho, overrun


def _arm_classes(metadata):
    names = metadata['bones']; parents = metadata['parents']
    wrist_ids = metadata.get('wrist_ids') or [names.index('手首.L'), names.index('手首.R')]
    classes = np.full(len(names), -1, dtype=np.int32)
    for bone in range(len(names)):
        for side in range(2):
            upper, fore = metadata['target_ids'][side*2:side*2+2]
            ancestor = bone
            while ancestor >= 0:
                if ancestor == wrist_ids[side]:
                    break
                if ancestor == fore:
                    classes[bone] = side*2+1; break
                if ancestor == upper:
                    classes[bone] = side*2; break
                ancestor = parents[ancestor]
    return classes, wrist_ids


def fit_arm_ellipses(geometry, metadata, samples_upper=10, samples_lower=13,
                     coverage_percentile=100., slab_halfwidth=.12,
                     minimum_vertices=12, axial_fraction=0.,
                     include_joint_neighbors=True):
    """Fit directional supports to the two arms without palm/finger vertices.

    ``slab_halfwidth`` is relative to segment length and matches the former
    circular-radius estimator's .12 neighbourhood.  At a joint, skin vertices
    dominated by either adjacent arm segment are eligible; this prevents a
    sparsely weighted elbow endpoint from shrinking to just two vertices.
    Sparse slabs expand only until minimum_vertices are available.

    coverage_percentile=100 encloses all selected main arm surface transverse projections.
    A 99 or 95 fit is available but its omitted-vertex count and worst radial
    overrun are explicitly reported.  This does not silently clamp the radius.
    axial_fraction=0 preserves the original point sampling.  A value of 1 gives
    an axial semi-axis of half a sample spacing; axial support increases the
    conservative envelope near contact normals aligned with the arm.
    """
    if not 50. <= coverage_percentile <= 100.:
        raise ValueError('coverage_percentile must be between 50 and 100')
    if samples_upper < 2 or samples_lower < 2:
        raise ValueError('Each segment needs at least two samples')
    if slab_halfwidth <= 0. or minimum_vertices < 4 or axial_fraction < 0.:
        raise ValueError('Invalid slab, minimum vertex count or axial fraction')
    vertices = np.asarray(geometry['vertices'], dtype=np.float64)
    rest = np.asarray(geometry['rest'], dtype=np.float64)
    classes, wrist_ids = _arm_classes(metadata)
    vertex_classes = classes[np.asarray(geometry['dominant'])]
    upper_t = np.linspace(.34, 1., samples_upper)
    fore_t = np.linspace(.04, .99, samples_lower)
    count = samples_upper + samples_lower
    axes_all = np.zeros((2, count, 3), dtype=np.float64)
    basis_all = np.empty((2, count, 3, 3), dtype=np.float64)
    bone_ids = np.empty((2, count), dtype=np.int32)
    segment_ids = np.tile(np.r_[np.zeros(samples_upper, int), np.ones(samples_lower, int)], (2, 1))
    samples = []
    for side in range(2):
        upper, fore = metadata['target_ids'][side*2:side*2+2]
        for segment, (a, b, ts, offset) in enumerate(((upper, fore, upper_t, 0),
                                                     (fore, wrist_ids[side], fore_t, samples_upper))):
            head = rest[a, :3, 3]; vector = rest[b, :3, 3] - head
            length = float(np.linalg.norm(vector)); longitudinal = vector / length
            rotation = rest[a, :3, :3].copy()
            rotation /= np.maximum(np.linalg.norm(rotation, axis=0), 1e-15)
            transverse = rotation[:, 0] - longitudinal * np.dot(rotation[:, 0], longitudinal)
            transverse /= np.linalg.norm(transverse)
            other = np.cross(transverse, longitudinal)
            projection = (vertices-head) @ longitudinal / length
            relative = vertices-head-projection[:, None]*vector
            projected = np.column_stack((relative@transverse, relative@other))
            eligible = (vertex_classes//2 == side) & (vertex_classes >= 0)
            if not include_joint_neighbors:
                eligible &= vertex_classes == side*2+segment
            eligible_ids = np.flatnonzero(eligible)
            if len(eligible_ids) < minimum_vertices:
                raise ValueError('Insufficient main arm surface vertices for '+metadata['bones'][a])
            for index, t in enumerate(ts):
                distance = np.abs(projection[eligible_ids] - t)
                width = max(slab_halfwidth, float(np.partition(distance, minimum_vertices-1)[minimum_vertices-1])+1e-10)
                selected = eligible_ids[distance <= width]
                cloud = projected[selected]
                axes, vectors, rho, overrun = _ellipse(cloud, coverage_percentile)
                robust_axes, _, robust_rho, robust_overrun = _ellipse(cloud, 99.)
                principal = transverse*vectors[0, 0] + other*vectors[1, 0]
                if np.dot(principal, transverse) < 0.:
                    principal = -principal
                final_other = np.cross(principal, longitudinal)
                world_basis = np.column_stack((principal, longitudinal, final_other))
                j = offset+index
                axial = .5 * (ts[1]-ts[0]) * length * axial_fraction
                axes_all[side, j] = (axes[0], axial, axes[1])
                basis_all[side, j] = rotation.T @ world_basis
                bone_ids[side, j] = a
                # Uniform angular statistics expose anisotropy without assuming
                # a future contact normal.  Actual body-side support depends on pose.
                angle = np.arange(180)*np.pi/180.
                support = np.sqrt((axes[0]*np.cos(angle))**2 + (axes[1]*np.sin(angle))**2)
                samples.append({'side':side, 'segment':segment, 'sample':j, 't':float(t),
                    'vertices':int(len(selected)), 'slab_halfwidth':float(width),
                    'axes_mm':(axes*1000).tolist(),
                    'circle95_mm':float(np.percentile(np.linalg.norm(cloud, axis=1),95)*1000),
                    'circle_max_mm':float(np.linalg.norm(cloud, axis=1).max()*1000),
                    'outside_count':int(np.sum(rho>1.+1e-7)),
                    'max_radial_overrun_mm':float(overrun.max()*1000),
                    'robust99_axes_mm':(robust_axes*1000).tolist(),
                    'robust99_outside_count':int(np.sum(robust_rho>1.+1e-7)),
                    'robust99_max_radial_overrun_mm':float(robust_overrun.max()*1000),
                    'angular_support_min_mean_max_mm':[float(support.min()*1000),float(support.mean()*1000),float(support.max()*1000)]})
    report = {'coverage_percentile':float(coverage_percentile),
        'all_selected_vertices_enclosed':bool(all(v['outside_count']==0 for v in samples)),
        'max_radial_overrun_mm':max(v['max_radial_overrun_mm'] for v in samples),
        'robust99_max_radial_overrun_mm':max(v['robust99_max_radial_overrun_mm'] for v in samples),
        'scope':'Rest skin in overlapping axial slabs; wrists/fingers excluded; moving full-resolution mesh still requires verification.',
        'samples':samples}
    return {'radii_axes':axes_all, 'basis_local':basis_all, 'bone_ids':bone_ids,
            'segment_ids':segment_ids, 't':np.r_[upper_t, fore_t],
            'vertices_class':vertex_classes, 'report':report}
