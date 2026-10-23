"""Exact-order temporal projection without quadratic frame-space matrices.

The frame derivative Hessian is five-diagonal. Project its nonzero diagonals
in source-frame order, then skip the zero rows in each spline column. This
keeps the original left-associated (B.T @ C) @ B evaluation order on the
generic NumPy BLAS used by Blender, including its tiny roundoff asymmetry.
"""
import numpy as np


def temporal_matrix(basis, velocity, acceleration):
    """Return B.T @ (2*v*D1.T@D1 + 2*a*D2.T@D2) @ B.

    basis is the existing dense, compact-support cubic spline basis. No
    frame-by-frame square array is allocated. A single controls-by-frames
    workspace replaces the old dense finite-difference matrices. In
    particular, do not symmetrize the result: eigh historically consumes
    its lower triangle and roundoff is part of the existing solver path.
    """
    basis = np.asarray(basis, dtype=np.float64)
    if basis.ndim != 2 or not basis.shape[0] or not basis.shape[1]:
        raise ValueError('basis must be a nonempty two-dimensional array')
    frames, controls = basis.shape
    velocity = float(velocity)
    acceleration = float(acceleration)
    first_diagonal = np.zeros(frames, dtype=np.float64)
    first_diagonal[:-1] += 1.
    first_diagonal[1:] += 1.
    second_diagonal = np.zeros(frames, dtype=np.float64)
    second_diagonal[:-2] += 1.
    second_diagonal[1:-1] += 4.
    second_diagonal[2:] += 1.
    second_adjacent = np.zeros(max(frames-1, 0), dtype=np.float64)
    second_adjacent[:-1] -= 2.
    second_adjacent[1:] -= 2.
    diagonal = 2*velocity*first_diagonal + 2*acceleration*second_diagonal
    adjacent = -2*velocity + 2*acceleration*second_adjacent
    outer = np.full(max(frames-2, 0), 2*acceleration, dtype=np.float64)

    # For any output frame, source frames are added in ascending order.
    left = np.zeros((controls, frames), dtype=np.float64)
    left[:, 2:] = basis[:-2].T * outer
    left[:, 1:] += basis[:-1].T * adjacent
    left += basis.T * diagonal
    left[:, :-1] += basis[1:].T * adjacent
    left[:, :-2] += basis[2:].T * outer
    projected = np.empty((controls, controls), dtype=np.float64)
    for column in range(controls):
        active = np.flatnonzero(basis[:, column])
        if active.size:
            projected[:, column] = left[:, active] @ basis[active, column]
        else:
            projected[:, column] = 0.
    return projected
