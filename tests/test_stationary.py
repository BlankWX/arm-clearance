"""Held-pose detection and constrained optimizer regressions (Blender Python)."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_clearance.stationary import stationary_spans, StationaryControls, MIN_HOLD_FRAMES, transition_basis
from test_temporal import spline_basis
from test_progress_solver import QuadraticFixture


def identity_pose(n=80):
    q = np.zeros((n, 6, 4))
    q[..., 0] = 1.
    return q, np.zeros((n, 2, 3))


class DetectionTests(unittest.TestCase):
    def test_static_bilateral_pose(self):
        q, loc = identity_pose()
        self.assertEqual(stationary_spans(q, loc), [(p, 0, 79) for p in range(8)])

    def test_signs_and_normalization(self):
        q, loc = identity_pose()
        q[1::2] *= -2.
        self.assertEqual(stationary_spans(q, loc), [(p, 0, 79) for p in range(8)])

    def test_short_pause_not_locked(self):
        q, loc = identity_pose(MIN_HOLD_FRAMES-1)
        self.assertEqual(stationary_spans(q, loc), [])

    def test_slow_intentional_motion(self):
        q, loc = identity_pose()
        angle = np.arange(len(q))*5e-8
        q[:, 0, 0], q[:, 0, 1] = np.cos(angle), np.sin(angle)
        self.assertEqual(stationary_spans(q, loc), [])

    def test_isolated_stationary_joint_remains_free(self):
        q, loc = identity_pose()
        angle = np.arange(len(q))*.01
        q[:, 0, 0], q[:, 0, 1] = np.cos(angle), np.sin(angle)
        self.assertEqual(stationary_spans(q, loc), [])

    def test_moving_wrist_position_remains_free(self):
        q, loc = identity_pose()
        loc[:, 0, 0] = np.arange(len(q))*.001
        self.assertEqual(stationary_spans(q, loc), [])

    def test_separate_hold_boundaries(self):
        q, loc = identity_pose()
        angle = np.zeros(80)
        angle[30:50] = np.linspace(.01, .4, 20)
        angle[50:] = .4
        q[:, 0, 0], q[:, 0, 1] = np.cos(angle), np.sin(angle)
        self.assertEqual(stationary_spans(q, loc),
                         [(p, a, b) for p in range(8) for a, b in [(0, 29), (49, 79)]])

    def test_bad_input(self):
        q, loc = identity_pose()
        for invalid in [q[:,:,:3], q*np.nan, q*0]:
            with self.assertRaises(ValueError): stationary_spans(invalid, loc)
        with self.assertRaises(ValueError): stationary_spans(q, loc[:20])


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.basis = spline_basis(80, 8)
        self.projector = StationaryControls(self.basis, 8, [(0, 0, 39), (6, 0, 39)])
        self.rng = np.random.default_rng(1964)

    def test_projection_is_orthogonal(self):
        a = self.rng.normal(size=(self.basis.shape[1], 24))
        b = self.rng.normal(size=a.shape)
        pa, pb = self.projector.project(a.copy()), self.projector.project(b.copy())
        np.testing.assert_allclose(self.projector.project(pa.copy()), pa, atol=1e-15)
        self.assertAlmostEqual(float(np.sum(pa*b)), float(np.sum(a*pb)), places=12)

    def test_hold_constant_and_other_parameters_unchanged(self):
        x = self.rng.normal(size=(self.basis.shape[1], 24))
        y = self.projector.project(x.copy())
        frames = (self.basis@y).reshape(80, 8, 3)
        np.testing.assert_allclose(frames[:40, 0], np.broadcast_to(frames[0, 0], (40, 3)), atol=1e-14)
        np.testing.assert_array_equal(y[:,3:18], x[:,3:18])
        self.assertGreater(float(np.max(np.abs(frames[60:, 0]-frames[0, 0]))), .01)

    def test_mask_releases_entire_tied_group(self):
        mask = np.zeros((self.basis.shape[1], 24), bool)
        indices, columns = self.projector.groups[0]
        mask[indices[0],columns] = True
        self.projector.expand_free(mask)
        self.assertTrue(mask[indices,columns].all())
        self.assertFalse(mask[:,3:].any())

    def test_shared_reduction_is_orthonormal_and_matches_projection(self):
        spans=[(parameter,0,39) for parameter in range(8)]
        projector=StationaryControls(self.basis,8,spans)
        reduction=projector.shared_reduction()
        np.testing.assert_allclose(reduction.T@reduction,np.eye(reduction.shape[1]),atol=1e-14)
        x=self.rng.normal(size=(self.basis.shape[1],24))
        np.testing.assert_allclose(reduction@(reduction.T@x),projector.project(x.copy()),atol=1e-14)
        self.assertIsNone(self.projector.shared_reduction())

    def test_overlapping_control_groups_merge(self):
        p = StationaryControls(self.basis, 8, [(0,0,20),(0,25,45)])
        self.assertEqual(len(p.groups), 1)


class TransitionBasisTests(unittest.TestCase):
    def test_refinement_retains_existing_spline_space(self):
        old=spline_basis(80,8)
        refined,added=transition_basis(80,8,[(p,0,39) for p in range(8)])
        self.assertTrue(added)
        self.assertTrue(np.all(refined>=0))
        np.testing.assert_allclose(refined.sum(1),1.,atol=1e-14)
        fit=np.linalg.lstsq(refined,old,rcond=None)[0]
        np.testing.assert_allclose(refined@fit,old,atol=1e-13)

    def test_whole_clip_hold_needs_no_extra_knots(self):
        basis,added=transition_basis(80,8,[(p,0,79) for p in range(8)])
        self.assertIsNone(basis);self.assertEqual(added,())

    def test_refined_hold_has_constant_pose_and_smooth_release(self):
        spans=[(p,0,39) for p in range(8)]
        basis,added=transition_basis(80,8,spans)
        projection=StationaryControls(basis,8,spans)
        controls=projection.project(np.random.default_rng(2026).normal(size=(basis.shape[1],24)))
        frames=basis@controls
        np.testing.assert_allclose(frames[:40],np.broadcast_to(frames[:1],(40,24)),atol=1e-14)
        fine,_=transition_basis(800,80,[(p,0,390) for p in range(8)])
        values=fine@controls
        np.testing.assert_allclose(values[::10],frames,atol=1e-14)
        jump=np.max(np.abs(values[391]-values[390]))
        larger=np.max(np.abs(values[392]-values[390]))
        self.assertGreater(jump,1e-10)
        self.assertAlmostEqual(larger/jump,8.,places=6)


class OptimizerTests(unittest.TestCase):
    def test_no_hold_path_is_unchanged(self):
        a, b = QuadraticFixture(), QuadraticFixture()
        b.stationary_spans = []
        for solver in [a,b]:
            for _ in solver.optimize_staged(30, stage_iterations=15): pass
        np.testing.assert_array_equal(a.result, b.result)
        self.assertEqual(a.objective_values, b.objective_values)

    def test_constant_under_staged_and_masked_warm_repair(self):
        solver = QuadraticFixture()
        solver.stationary_spans = [(0,0,39),(6,0,39)]
        for _ in solver.optimize_staged(80, stage_iterations=40): pass
        for parameter in [0,6]:
            self.assertLess(float(np.max(np.abs(solver.result[:40,parameter]-solver.result[0,parameter]))),1e-13)
        mask = np.zeros(80,bool);mask[5]=True
        for _ in solver.optimize(15, initial=solver.result.copy(), free_frames=mask): pass
        self.assertEqual(solver.last_optimization_scope['locked_control_drift_max_abs'],0.)
        self.assertLess(solver.last_optimization_scope['stationary_max_correction_excursion'],1e-13)

    def test_shared_preconditioner_preserves_static_and_transition(self):
        solver=QuadraticFixture();solver.stationary_spans=[(p,0,39) for p in range(8)]
        for _ in solver.optimize_staged(80,stage_iterations=40):pass
        self.assertTrue(hasattr(solver,'_stationary_eigen_cache'))
        self.assertLess(float(np.max(np.abs(solver.result[:40]-solver.result[:1]))),1e-13)
        self.assertGreater(float(np.max(np.abs(solver.result[60:]-solver.result[:1]))),.01)
        self.assertLess(solver.objective_values[-1],solver.objective_values[0])

    def test_fully_frozen_warm_pass(self):
        solver = QuadraticFixture();solver.stationary_spans = [(p,0,79) for p in range(8)]
        for _ in solver.optimize(30): pass
        before=solver.result.copy()
        for _ in solver.optimize(10, initial=before, free_frames=np.zeros(80,bool)): pass
        np.testing.assert_allclose(solver.result,before,atol=1e-14,rtol=0)
        self.assertEqual(solver.last_optimization_scope['locked_control_drift_max_abs'],0.)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
