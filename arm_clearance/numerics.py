"""Shared numerical primitives; exact v0.2.12 statements, no Blender state."""
import numpy as np

def rotation_vectors(v):
    """SO(3) exponential map, including a stable zero-angle limit."""
    theta = np.linalg.norm(v, axis=-1)
    a = np.sinc(theta / np.pi)
    b = .5 * np.sinc(theta / (2 * np.pi)) ** 2
    k = np.zeros(v.shape[:-1] + (3, 3), dtype=np.float64)
    k[..., 0, 1] = -v[..., 2]; k[..., 0, 2] = v[..., 1]
    k[..., 1, 0] = v[..., 2]; k[..., 1, 2] = -v[..., 0]
    k[..., 2, 0] = -v[..., 1]; k[..., 2, 1] = v[..., 0]
    return np.eye(3) + a[..., None, None] * k + b[..., None, None] * (k @ k)


def quaternion_multiply(a, b):
    w = a[..., :1]*b[..., :1] - np.sum(a[..., 1:]*b[..., 1:], axis=-1, keepdims=True)
    xyz = a[..., :1]*b[..., 1:] + b[..., :1]*a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate((w, xyz), axis=-1)


def corrected_quaternions(original, delta):
    angle = np.linalg.norm(delta, axis=-1, keepdims=True)
    q = np.concatenate((np.cos(angle/2), .5*np.sinc(angle/(2*np.pi))*delta), axis=-1)
    result = quaternion_multiply(original, q)
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    # A quaternion sign change describes the same pose but breaks interpolation.
    for f in range(1, len(result)):
        result[f] *= np.where(np.sum(result[f]*result[f-1], axis=-1, keepdims=True)<0, -1., 1.)
    return result


def skin_vertices(vertices, joints, weights, rest, pose):
    transform = pose @ np.linalg.inv(rest)
    out = np.zeros((len(vertices), 3), np.float64)
    for k in range(joints.shape[1]):
        m = transform[joints[:, k]]
        out += (np.einsum('nij,nj->ni', m[:, :3, :3], vertices) + m[:, :3, 3]) * weights[:, k, None]
    return out


def project_parameters(model, values):
    """Copy parameter frames into the model's hard admissible subspace.

    Fixed channels remain in the legacy array format for compatibility. They
    are not optimization variables and are removed before FK/contact caching.
    """
    result=np.asarray(values,dtype=np.float64).copy()
    parameters=getattr(model,'parameter_count',4)
    if result.shape!=(model.n,parameters,3):
        raise ValueError('parameters must have shape [frames, parameter_count, 3]')
    fixed=tuple(getattr(model,'fixed_parameter_indices',()))
    if fixed:
        if any(index<0 or index>=parameters for index in fixed):
            raise ValueError('fixed parameter index is out of range')
        result[:,fixed]=0.
    return result
