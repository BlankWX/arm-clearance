"""Temporal projection arithmetic, boundary and whole-clip regression checks."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'temporal_standalone', ROOT/'arm_clearance/temporal.py')
TEMPORAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TEMPORAL)


def spline_basis(frames, spacing):
    t = np.arange(frames)/spacing
    k = np.floor(t).astype(int)
    u = t-k
    weights = np.stack(((1-u)**3, 3*u**3-6*u*u+4,
                        -3*u**3+3*u*u+3*u+1, u**3), axis=1)/6
    basis = np.zeros((frames, int(k.max())+4))
    basis[np.arange(frames)[:, None], k[:, None]+np.arange(4)] = weights
    return basis


def legacy_temporal(basis, velocity, acceleration):
    frames = len(basis)
    first = np.diff(np.eye(frames), axis=0)
    second = np.diff(np.eye(frames), n=2, axis=0)
    return basis.T@(2*velocity*(first.T@first)
                    + 2*acceleration*(second.T@second))@basis


class TemporalProjectionTests(unittest.TestCase):
    def assert_same(self, basis, velocity, acceleration):
        expected = legacy_temporal(basis, velocity, acceleration)
        actual = TEMPORAL.temporal_matrix(basis, velocity, acceleration)
        # Generic NumPy supplied with Blender is bitwise equal. Other
        # optimized BLAS implementations may reduce products differently.
        np.testing.assert_allclose(actual, expected, rtol=8e-14, atol=8e-12)
        config = getattr(np.__config__, 'CONFIG', {})
        blas = config.get('Build Dependencies', {}).get('blas', {})
        if blas.get('name') == 'auto':
            # Blender's generic reductions must retain every legacy bit.
            np.testing.assert_array_equal(actual.view(np.uint64),
                                          expected.view(np.uint64))
        self.assertLessEqual(float(np.max(np.abs(actual-actual.T))),
                             max(1., float(np.max(np.abs(actual))))*2e-13)
        return expected, actual

    def test_short_clips_boundaries_spacings_and_weights(self):
        for frames in (1, 2, 3, 4, 5, 16, 80, 174):
            for spacing in (1, 2, 3, 8):
                basis = spline_basis(frames, spacing)
                for velocity, acceleration in ((10., 3000.), (0., 3000.),
                                                (10., 0.), (0., 0.), (.13, .7)):
                    with self.subTest(frames=frames, spacing=spacing,
                                      velocity=velocity, acceleration=acceleration):
                        self.assert_same(basis, velocity, acceleration)

    def test_eight_hundred_frame_projection(self):
        self.assert_same(spline_basis(800, 8), 10., 3000.)

    def test_constant_signal_nullspace_and_nonnegative_energy(self):
        rng = np.random.default_rng(119)
        for frames, spacing in ((3, 1), (174, 8), (800, 8)):
            basis = spline_basis(frames, spacing)
            projected = TEMPORAL.temporal_matrix(basis, 10., 3000.)
            np.testing.assert_allclose(projected@np.ones(basis.shape[1]),
                                       0., rtol=0., atol=2e-11)
            values = np.linalg.eigvalsh(projected)
            self.assertGreaterEqual(float(values.min()), -2e-10)
            controls = rng.normal(size=(basis.shape[1], 9))
            signal = basis@controls
            expected = (20*np.sum(np.diff(signal, axis=0)**2)
                        + 6000*np.sum(np.diff(signal, n=2, axis=0)**2))
            actual = float(np.sum(controls*(projected@controls)))
            self.assertAlmostEqual(actual/expected, 1., places=12)

    def test_eigendecomposition_and_inverse_hessian(self):
        rng = np.random.default_rng(419)
        for frames, spacing in ((3, 1), (80, 3), (174, 8)):
            expected, actual = self.assert_same(
                spline_basis(frames, spacing), 10., 3000.)
            old_values, old_vectors = np.linalg.eigh(expected)
            new_values, new_vectors = np.linalg.eigh(actual)
            q = rng.normal(size=(len(old_values), 24))
            old = old_vectors@((old_vectors.T@q)/(1000+old_values[:, None]))
            new = new_vectors@((new_vectors.T@q)/(1000+new_values[:, None]))
            np.testing.assert_allclose(new, old, rtol=2e-11, atol=2e-14)

    def test_build_does_not_construct_frame_identity(self):
        basis = spline_basis(800, 8)
        with patch.object(np, 'eye', side_effect=AssertionError(
                'frame-square finite difference construction regressed')):
            projected = TEMPORAL.temporal_matrix(basis, 10., 3000.)
        self.assertEqual(projected.shape, (basis.shape[1], basis.shape[1]))

    def test_unused_control_and_invalid_shape(self):
        basis = np.column_stack((spline_basis(16, 8), np.zeros(16)))
        self.assert_same(basis, 10., 3000.)
        for basis in (np.zeros(4), np.zeros((0, 4)), np.zeros((4, 0))):
            with self.assertRaises(ValueError):
                TEMPORAL.temporal_matrix(basis, 10., 3000.)


if __name__ == '__main__':
    unittest.main()
