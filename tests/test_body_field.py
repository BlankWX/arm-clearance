"""Offline analytic derivative and conservative triangle-cover checks."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('body_field_standalone', ROOT/'arm_clearance/body_field.py')
BF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BF)


def cube_fixture():
    vertices = np.array([[x, y, z] for x in (-.1, .1) for y in (-.08, .08) for z in (0., .7)])
    triangles = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                          [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                          [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    collider = {'vertices':vertices, 'triangles':triangles,
                'joint':np.zeros((8, 1), dtype=int), 'weight':np.ones((8, 1)),
                'rest':np.eye(4)[None]}
    matrices = np.broadcast_to(np.eye(4), (2, 1, 4, 4)).copy()
    angle = .17
    matrices[1, 0, :3, :3] = [[np.cos(angle), -np.sin(angle), 0.],
                              [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
    return collider, matrices


class TightBodyFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collider, cls.matrices = cube_fixture()
        cls.field = BF.TightBodyField(cls.collider, cls.matrices, {'bones':['body']}, .2)
        for _ in cls.field.prepare():
            pass

    def test_clipped_triangle_surface_is_conservatively_covered(self):
        rng = np.random.default_rng(42)
        triangles = self.collider['triangles']
        ids = rng.integers(0, len(triangles), 100)
        bary = rng.random((100, 3)); bary /= bary.sum(axis=1, keepdims=True)
        rest = np.einsum('ni,nij->nj', bary, self.collider['vertices'][triangles[ids]])
        posed = np.stack([rest@matrix[0, :3, :3].T for matrix in self.matrices])
        points = np.repeat(posed[:, None], 2, axis=1)
        distance, _ = self.field.distances(points)
        self.assertLessEqual(float(distance.max()), 1e-10)
        self.assertEqual(self.field.report['collider_triangles_covered'], len(triangles))
        self.assertTrue(self.field.report['all_region_vertices_inside_support_planes'])
        self.assertLess(self.field.report['total_smoothing_expansion_bound'], .001)
        for region in self.field.regions:
            self.assertTrue(np.allclose(region['barycentric'].sum(axis=1), 1.))
            self.assertGreater(float(region['barycentric'].min()), -1e-10)

    def test_contact_gradient_and_hessian_match_finite_differences(self):
        rest = np.array([[.1002, .0803, .123], [.101, .0797, .348],
                         [-.1003, -.0802, .57], [.0995, -.0802, .23]])
        posed = np.stack([rest@matrix[0, :3, :3].T for matrix in self.matrices])
        points = np.repeat(posed[:, None], 2, axis=1)
        distance, gradient, hessian = self.field.distances(points, hessian=True)
        self.assertTrue(np.allclose(distance, self.field.distances(points)[0], atol=1e-14))
        self.assertGreater(float(np.abs(hessian).max()), 10.)
        self.assertLess(float(np.abs(hessian-np.swapaxes(hessian, -1, -2)).max()), 1e-9)
        epsilon = 1e-7
        for axis in range(3):
            offset = np.zeros_like(points); offset[..., axis] = epsilon
            plus_d, plus_g = self.field.distances(points+offset)
            minus_d, minus_g = self.field.distances(points-offset)
            self.assertLess(float(np.abs((plus_d-minus_d)/(2*epsilon)-gradient[..., axis]).max()), 2e-6)
            self.assertLess(float(np.abs((plus_g-minus_g)/(2*epsilon)-hessian[..., :, axis]).max()), .01)

    def test_active_planes_remain_conservative_and_restore_exactly(self):
        rest = np.array([[.1002, .0803, .123], [.101, .0797, .348],
                         [-.1003, -.0802, .57], [.0995, -.0802, .23]])
        posed = np.stack([rest@matrix[0, :3, :3].T for matrix in self.matrices])
        points = np.repeat(posed[:, None], 2, axis=1)
        full_d, full_g, full_h = self.field.distances(points, hessian=True)
        shifted = points+np.array([.004, -.003, .002])
        full_shifted = self.field.distances(shifted)[0]
        try:
            report = self.field.update_active_planes(points)
            active_d, active_g, active_h = self.field.distances(points, hessian=True)
            self.assertLessEqual(float(np.max(active_d-full_d)), 1e-12)
            self.assertLess(float(np.max(full_d-active_d)), report['distance_loss_bound_at_update_points']+1e-10)
            self.assertLess(float(np.abs(active_g-full_g).max()), 1e-4)
            self.assertLess(float(np.abs(active_h-full_h).max()), 1.)
            self.assertLessEqual(float(np.max(self.field.distances(shifted)[0]-full_shifted)), 1e-12)
            epsilon = 1e-7
            for axis in range(3):
                offset = np.zeros_like(points); offset[..., axis] = epsilon
                plus_d, plus_g = self.field.distances(points+offset)
                minus_d, minus_g = self.field.distances(points-offset)
                self.assertLess(float(np.abs((plus_d-minus_d)/(2*epsilon)-active_g[..., axis]).max()), 2e-6)
                self.assertLess(float(np.abs((plus_g-minus_g)/(2*epsilon)-active_h[..., :, axis]).max()), .01)
        finally:
            self.field.restore_full_planes()
        self.assertTrue(np.array_equal(self.field.distances(points)[0], full_d))

    def test_hessian_mask_and_distance_limit_preserve_contact_curvature(self):
        rest = np.array([[.1002, .0803, .123], [.101, .0797, .348],
                         [-.1003, -.0802, .57], [.0995, -.0802, .23]])
        posed = np.stack([rest@matrix[0, :3, :3].T for matrix in self.matrices])
        points = np.repeat(posed[:, None], 2, axis=1)
        full_d, full_g, full_h = self.field.distances(points, hessian=True)
        mask = np.broadcast_to(np.array([True, False, True, False]), points.shape[:-1])
        limited_d, limited_g, limited_h = self.field.distances(
            points, hessian=True, hessian_mask=mask,
            hessian_distance_limit=full_d+.001)
        self.assertTrue(np.array_equal(full_d, limited_d))
        self.assertTrue(np.array_equal(full_g, limited_g))
        self.assertEqual(float(np.abs(limited_h[~mask]).max()), 0.)
        self.assertLess(float(np.abs(limited_h[mask]-full_h[mask]).max()), 1e-7)
        zero_h = self.field.distances(points, hessian=True, hessian_distance_limit=-1.)[2]
        self.assertEqual(float(np.abs(zero_h).max()), 0.)


if __name__ == '__main__':
    unittest.main(argv=[__file__], verbosity=2)
