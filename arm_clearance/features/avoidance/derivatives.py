"""Analytical virtual-work FK gradient for the existing six-bone limb model.

NumPy-only.  This changes neither the model nor the objective.  Force is dE/dp,
torque is dE/d(world left rotation), and ef/wf are the extra elbow/wrist point
gradients.  Wrist blended-skin offsets and translation weights follow exactly
``HandGeometry.apply_corrections``; fingers keep their original local motion.
"""

import numpy as np


def _skew(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    matrix = np.zeros(vectors.shape[:-1]+(3, 3), dtype=np.float64)
    matrix[..., 0, 1] = -vectors[..., 2]
    matrix[..., 0, 2] = vectors[..., 1]
    matrix[..., 1, 0] = vectors[..., 2]
    matrix[..., 1, 2] = -vectors[..., 0]
    matrix[..., 2, 0] = -vectors[..., 1]
    matrix[..., 2, 1] = vectors[..., 0]
    return matrix


def so3_left_jacobian(vectors):
    """Spatial left Jacobian of exp([x]); stable at exactly zero rotation."""
    vectors = np.asarray(vectors, dtype=np.float64)
    theta2 = np.sum(vectors*vectors, axis=-1)
    theta = np.sqrt(theta2)
    first = .5*np.sinc(theta/(2.*np.pi))**2
    # 1-sinc(theta) loses precision around zero; use its analytic series there.
    second = np.empty_like(theta)
    small = theta2 < 1e-6
    second[small] = 1./6.-theta2[small]/120.+theta2[small]**2/5040.
    second[~small] = (1.-np.sinc(theta[~small]/np.pi))/theta2[~small]
    skew = _skew(vectors)
    return np.eye(3)+first[..., None, None]*skew+second[..., None, None]*(skew@skew)


def _exp(vectors):
    theta = np.linalg.norm(vectors, axis=-1)
    skew = _skew(vectors)
    return (np.eye(3)+np.sinc(theta/np.pi)[..., None, None]*skew
            +(.5*np.sinc(theta/(2.*np.pi))**2)[..., None, None]*(skew@skew))


def analytical_fk_gradient(model, x, points, orientation, elbows, wrists,
                           force, torque, ef, wf, palm_matrix_gradient=None):
    """Return [N,8,3] geometry gradient, excluding temporal/parameter costs.

    The passed points/elbows/wrists must be the same ``model.state(x)`` result
    used to compute the forces.  orientation is accepted for a drop-in call at
    that site; torque already summarizes its contribution, so no orientation
    differentiation or extra state evaluation is needed here.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (model.n, 8, 3):
        raise ValueError('Expected the model\'s [frames,8,3] parameter array')
    points = np.asarray(points); force = np.asarray(force); torque = np.asarray(torque)
    elbows = np.asarray(elbows); wrists = np.asarray(wrists)
    ef = np.asarray(ef); wf = np.asarray(wf)
    rot = np.asarray(model.rot)
    jacobian = so3_left_jacobian(x[:, :6])
    local = _exp(x[:, :6])
    world = rot@local@np.swapaxes(rot, -1, -2)
    upper = world[:, (0, 3)]
    fore = world[:, (1, 4)]
    wrist = world[:, (2, 5)]
    upstream = upper@fore
    result = np.zeros_like(x)

    # Every downstream centre, including the fractionally translated seam,
    # rotates about the shoulder under the upper-arm world rotation.
    upper_moment = (np.cross(points-model.shoulders[:, :, None], force).sum(axis=2)
                    +torque.sum(axis=2)
                    +np.cross(elbows-model.shoulders, ef)
                    +np.cross(wrists-model.shoulders, wf))
    upper_axes = rot[:, (0, 3)]@jacobian[:, (0, 3)]
    result[:, (0, 3)] = np.einsum('nsij,nsi->nsj', upper_axes, upper_moment)

    arm_count = int(model.arm_count)
    fore_start = len(model.tu)
    fore_moment = (np.cross(points[:, :, fore_start:]-elbows[:, :, None],
                            force[:, :, fore_start:]).sum(axis=2)
                   +torque[:, :, fore_start:].sum(axis=2)
                   +np.cross(wrists-elbows, wf))
    fore_axes = upper@rot[:, (1, 4)]@jacobian[:, (1, 4)]
    result[:, (1, 4)] = np.einsum('nsij,nsi->nsj', fore_axes, fore_moment)

    # The wrist lever is only its weighted subtree component.  Using p-W
    # here would incorrectly rotate the forearm-weight portion at the seam.
    wrist_lever = np.einsum('nsij,nspj->nspi', upstream@wrist,
                            np.asarray(model.hand.wrist_offsets))
    wrist_moment = (np.cross(wrist_lever, force[:, :, arm_count:]).sum(axis=2)
                    +torque[:, :, arm_count:].sum(axis=2))
    wrist_axes = upstream@rot[:, (2, 5)]@jacobian[:, (2, 5)]
    result[:, (2, 5)] = np.einsum('nsij,nsi->nsj', wrist_axes, wrist_moment)

    fraction = model.hand.wrist_weight_fraction
    if fraction is None:
        raise ValueError('Analytical wrist translation needs wrist subtree weight fractions')
    translation_force = (force[:, :, arm_count:]*np.asarray(fraction)[None, :, :, None]).sum(axis=2)+wf
    translation_axes = model.length*(upstream@np.asarray(model.location_bases))
    result[:, 6:] = np.einsum('nsij,nsi->nsj', translation_axes, translation_force)
    if palm_matrix_gradient is not None:
        # Exact pullback of the first hand-frame matrix:
        # R = (Au Eu Au.T)(Af Ef Af.T)(Ah Eh Ah.T) Ah B.
        # Imported source matrices are not perfectly orthonormal. Differentiate
        # A exp(x) A.T as written, using the already computed local/world/J data,
        # instead of assuming that A.T@A is exactly identity.
        matrix_gradient=np.asarray(palm_matrix_gradient,dtype=np.float64)
        base=rot[:,(2,5)]@np.asarray(model.hand_basis)[:,0]
        upper_a=rot[:,(0,3)];fore_a=rot[:,(1,4)];hand_a=rot[:,(2,5)]
        tails=(local[:,(0,3)]@upper_a.swapaxes(-1,-2)@fore@wrist@base,
               local[:,(1,4)]@fore_a.swapaxes(-1,-2)@wrist@base,
               local[:,(2,5)]@hand_a.swapaxes(-1,-2)@base)
        prefixes=(upper_a,upper@fore_a,upstream@hand_a)
        for ids,prefix,tail in zip(((0,3),(1,4),(2,5)),prefixes,tails):
            column_gradient=prefix.swapaxes(-1,-2)@matrix_gradient
            moment=np.cross(tail.swapaxes(-1,-2),column_gradient.swapaxes(-1,-2)).sum(axis=-2)
            result[:,ids]+=np.einsum('nsij,nsi->nsj',jacobian[:,ids],moment)
    return result
