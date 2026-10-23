"""Real C2 optimizer progress regression with an 80-frame convex fixture.

Uses the real ClipSolver.optimize and FullLimbSolver.optimize_staged methods.
Only the contact objective and plane cache are deterministic synthetic fixtures;
no existing Blender scene, character, action or MCP connection is accessed.

  blender -b --factory-startup --python-exit-code 1 --python tests/test_progress_solver.py
"""
import copy
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/test-results/progress-solver'
sys.path.insert(0, str(ROOT))
from arm_clearance.solver import ClipSolver
from arm_clearance.solver_v2 import FullLimbSolver


class PlaneFixture:
    def __init__(self):
        self.updates = []
        self.restores = 0

    def update_active_planes(self, points):
        # A fixed convex objective has no contact-plane state to change.
        self.updates.append(np.asarray(points).copy())

    def restore_full_planes(self):
        self.restores += 1


class QuadraticFixture(FullLimbSolver):
    """Use production optimizers with an independent positive quadratic."""
    def __init__(self, free_frames=None, optimize=None, optimize_staged=None):
        self.events = []
        self.objective_values = []
        self.model = SimpleNamespace(n=80, parameter_count=8, points=lambda value: value)
        ClipSolver.__init__(self, self.model, {}, np.zeros((80, 1, 4, 4)), self.record)
        self.velocity = .7
        self.acceleration = 4.
        self.body = PlaneFixture()
        self.free_frames = free_frames
        self.optimization_stage_index = 1
        self.optimization_stage_count = 1
        times = np.linspace(0., 2.*np.pi, self.model.n)[:, None, None]
        phases = np.arange(24).reshape(1, 8, 3)*.13
        self.target = .15*np.sin(times*1.4 + phases) + .035*np.cos(times*3.1-phases)
        self.weights = np.linspace(.8, 2.3, 24).reshape(1, 8, 3)
        if optimize is not None:
            self.optimize = MethodType(optimize, self)
        if optimize_staged is not None:
            self.optimize_staged = MethodType(optimize_staged, self)

    def record(self, phase, current=0, total=0, *info):
        self.events.append({'phase': phase, 'current': int(current), 'total': int(total),
                            'group': self.optimization_stage_index,
                            'groups': self.optimization_stage_count,
                            'info': copy.deepcopy(info)})

    def objective(self, flat):
        x = np.asarray(flat).reshape(80, 8, 3)
        difference = x-self.target
        d1, d2 = np.diff(x, axis=0), np.diff(x, n=2, axis=0)
        energy = (np.sum(self.weights*difference*difference)
                  + self.velocity*np.sum(d1*d1) + self.acceleration*np.sum(d2*d2))
        gradient = 2.*self.weights*difference
        gradient[:-1] -= 2.*self.velocity*d1
        gradient[1:] += 2.*self.velocity*d1
        gradient[:-2] += 2.*self.acceleration*d2
        gradient[1:-1] -= 4.*self.acceleration*d2
        gradient[2:] += 2.*self.acceleration*d2
        self.evaluations += 1
        self.last_metrics = {'energy': float(energy),
                             'gradient_max': float(np.max(np.abs(gradient)))}
        self.objective_values.append(float(energy))
        return float(energy), gradient.ravel()


def run_call(solver, name, iterations, initial=None, stage_iterations=None):
    event_start, history_start = len(solver.events), len(solver.history)
    evaluation_start, value_start = solver.evaluations, len(solver.objective_values)
    update_start, restore_start = len(solver.body.updates), solver.body.restores
    if stage_iterations is None:
        solver.optimization_stage_index = solver.optimization_stage_count = 1
        generator = solver.optimize(iterations, initial=initial, free_frames=solver.free_frames)
    else:
        generator = solver.optimize_staged(iterations, initial=initial,
            stage_iterations=stage_iterations, free_frames=solver.free_frames)
    yielded = [value.copy() for value in generator]
    return {'name': name, 'iterations': iterations, 'stage_iterations': stage_iterations,
            'result': solver.result.copy(), 'yields': yielded,
            'events': copy.deepcopy(solver.events[event_start:]),
            'evaluations': solver.evaluations-evaluation_start,
            'objective_values': solver.objective_values[value_start:].copy(),
            'history': copy.deepcopy(solver.history[history_start:]),
            'scope': copy.deepcopy(solver.last_optimization_scope),
            'plane_updates': [v.copy() for v in solver.body.updates[update_start:]],
            'plane_restores': solver.body.restores-restore_start,
            'cache_present': getattr(solver, '_temporal_eigen_cache', None) is not None}


def run_sequences(optimize=None, optimize_staged=None):
    partial = np.zeros(80, dtype=bool)
    partial[29:43] = True
    cases = {}
    for mask_name, mask in (('all', None), ('partial', partial),
                            ('frozen', np.zeros(80, dtype=bool))):
        solver = QuadraticFixture(mask, optimize, optimize_staged)
        calls = []
        calls.append(run_call(solver, 'cold', 13))
        cache = solver._temporal_eigen_cache
        calls.append(run_call(solver, 'warm_cached', 9, initial=calls[-1]['result']))
        assert solver._temporal_eigen_cache is cache, 'warm solve did not reuse the matrix cache'
        calls.append(run_call(solver, 'staged_warm_cached', 17,
                              initial=calls[-1]['result'], stage_iterations=6))
        assert solver._temporal_eigen_cache is cache, 'staged warm solve changed the matrix cache'
        calls.append(run_call(solver, 'staged_cold_cached', 11, stage_iterations=4))
        assert solver._temporal_eigen_cache is cache, 'staged cold solve changed the matrix cache'
        cases[mask_name] = calls
    return cases


def check_events(call):
    events = call['events']
    group_count = (1 if call['stage_iterations'] is None else
                   (call['iterations']+call['stage_iterations']-1)//call['stage_iterations'])
    grouped = []
    for group in range(1, group_count+1):
        group_events = [event for event in events if event['group'] == group]
        assert group_events, (call['name'], group)
        assert all(event['groups'] == group_count for event in group_events)
        phases = [event['phase'] for event in group_events]
        prefix = ['solve_prepare', 'solve_initial', 'solve_temporal']
        if call['stage_iterations'] is not None:
            prefix.insert(0, 'solve_stage')
        assert phases[:len(prefix)] == prefix, phases
        assert phases[-1] == 'solve_done', phases
        iteration_events = [event for event in group_events if event['phase'] == 'solve_iteration']
        assert iteration_events
        assert [event['current'] for event in iteration_events] == list(range(len(iteration_events)))
        for iteration in iteration_events:
            index = iteration['current']
            trials = [event for event in group_events
                      if event['phase'] == 'solve_trial' and event['current'] == index]
            assert trials, (call['name'], group, index)
            assert [event['info'][0] for event in trials] == list(range(1, len(trials)+1))
            assert len(trials) <= 16
            assert all(event['total'] == iteration['total'] for event in trials)
            first_trial = group_events.index(trials[0])
            assert group_events.index(iteration) < first_trial
        done = group_events[-1]
        assert 0 <= done['current'] <= done['total']
        expected_budget = (call['iterations'] if call['stage_iterations'] is None else
                           min(call['stage_iterations'], call['iterations']-(group-1)*call['stage_iterations']))
        assert done['total'] == expected_budget
        accepted_events = [event for event in group_events if event['phase'] == 'solve']
        assert all(isinstance(event['info'][0], dict) for event in accepted_events)
        assert all(event['current'] % 5 == 0 for event in accepted_events)
        accepted_energies = [event['info'][0]['energy'] for event in accepted_events]
        assert all(b <= a+1e-10 for a, b in zip(accepted_energies, accepted_energies[1:]))
        grouped.append({'group': group, 'budget': done['total'],
                        'accepted_steps': done['current'], 'iteration_events': len(iteration_events),
                        'trial_events': sum(event['phase'] == 'solve_trial' for event in group_events)})
    assert len(call['yields']) == sum(event['phase'] == 'solve' for event in events)+group_count
    assert call['evaluations'] == len(call['objective_values'])
    assert call['objective_values'][-1] <= call['objective_values'][0]+1e-9
    assert call['cache_present']
    assert len(call['plane_updates']) == (group_count if call['stage_iterations'] else 0)
    assert call['plane_restores'] == (group_count if call['stage_iterations'] else 0)
    assert np.isfinite(call['result']).all()
    assert all(np.isfinite(value).all() for value in call['yields'])
    np.testing.assert_array_equal(call['result'], call['yields'][-1])
    return grouped


def check_cases(cases):
    summary = {}
    for mask_name, calls in cases.items():
        summary[mask_name] = []
        for call in calls:
            groups = check_events(call)
            scope = call['scope']
            if mask_name == 'all':
                assert not scope['masked'] and scope['editable_frame_count'] == 80
            else:
                assert scope['masked'] and scope['locked_control_drift_max_abs'] == 0.
                assert scope['editable_frame_count'] < 80
                if mask_name == 'frozen':
                    assert scope['free_controls'] == scope['editable_frame_count'] == 0
                    assert all(group['accepted_steps'] == 1 for group in groups)
            summary[mask_name].append({'name': call['name'], 'groups': groups,
                                      'evaluations': call['evaluations'], 'yields': len(call['yields']),
                                      'initial_energy': call['objective_values'][0],
                                      'last_energy': call['objective_values'][-1], 'scope': scope})
    return summary


def main():
    cases = run_sequences()
    summary = check_cases(cases)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'passed': True, 'frames': 80, 'parameters': 8,
              'objective': 'positive quadratic with velocity and acceleration costs',
              'optimizer_methods': ['ClipSolver.optimize', 'FullLimbSolver.optimize_staged'],
              'scope': 'synthetic numeric fixture; no user scene or animation acceptance',
              'cases': summary}
    (OUT/'solver-progress.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'passed': True, 'mask_cases': len(cases),
                      'calls': sum(map(len, cases.values())),
                      'evaluations': sum(call['evaluations'] for calls in cases.values() for call in calls),
                      'report': str(OUT/'solver-progress.json')}))


if __name__ == '__main__':
    main()
