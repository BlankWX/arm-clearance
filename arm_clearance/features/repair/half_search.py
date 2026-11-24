"""Bounded local rotation search for isolated failed interpolated samples."""
import math
import numpy as np


def isolated_half_frames(previous_surface):
    failures = set(previous_surface['problem_frames'])
    return sorted(f for f in failures if f % 1 == .5 and
                      f-.5 not in failures and f+.5 not in failures)


def contact_sides(surface):
    sides = set()
    for record in surface['records']:
        for details in record['categories'].values():
            for pair in details['pairs']:
                for part in (pair['part_a'], pair['part_b']):
                    if part.endswith(('.L', '.R')):
                        sides.add(int(part.endswith('.R')))
            for item in details['containments']:
                for part in (item['source_part'], item['target_part']):
                    if part.endswith(('.L', '.R')):
                        sides.add(int(part.endswith('.R')))
    return sorted(sides)


def contact_score(surface):
    categories = surface['categories'].values()
    return (surface['collision_samples'], sum(len(v['problem_frames']) for v in categories),
            sum(v['intersection_pair_count'] for v in categories))


def search_half_pose(keys, baseline, evaluate, initial=None):
    sides = contact_sides(baseline)
    zero = np.zeros((6, 3))
    best = zero.copy() if initial is None else initial.copy()
    best_score = (*contact_score(baseline), float(np.sum(best**2)), float("inf"))
    allowed = set(baseline['problem_frames'])
    seen = set()

    def consider(vector):
        nonlocal best, best_score
        lengths = np.linalg.norm(vector, axis=1)
        if np.any(lengths > np.asarray(keys.rotation_limits)+1e-7):
            return
        signature = np.round(vector, 6).tobytes()
        if signature in seen:
            return
        seen.add(signature)
        keys.write(vector)
        surface, wrist_error, smooth = evaluate()
        if not smooth or not set(surface['problem_frames']) <= allowed:
            return
        score = (*contact_score(surface), float(np.sum(lengths**2)), wrist_error)
        if score < best_score:
            best_score, best = score, vector.copy()

    # A symmetric shoulder detour can separate both hands when either
    # individual arm would still be blocked by the other.
    for degree in ((5, 12, 22, 35, 48) if initial is None else ()):
        for side_group in ([sides] + [[side] for side in sides] if len(sides)>1 else [sides]):
            vector = zero.copy()
            for side in side_group:
                i = 3*side
                vector[i] = keys.swing_axes[side]*min(math.radians(degree), keys.rotation_limits[i])
            consider(vector)
            yield None
        if best_score[0] == 0:
            break

    for degree in (18, 9, 4, 2):
        if not best_score[0]:
            break
        for _ in range(3):
            start = best.copy()
            for side in sides:
                for bone in range(3*side, 3*side+3):
                    for axis in range(3):
                        for sign in (-1, 1):
                            vector = start.copy()
                            vector[bone,axis] += sign*math.radians(degree)
                            consider(vector)
                            yield None
                        if not best_score[0]:
                            break
                    if not best_score[0]:
                        break
                if not best_score[0]:
                    break
            if np.array_equal(start,best) or not best_score[0]:
                break
    if best_score[0] == 0:
        # Reduce the accepted displacement along its ray, preserving clearance.
        solution = best.copy()
        low, high = 0., 1.
        for _ in range(6):
            middle = (low+high)/2
            consider(solution*middle)
            yield None
            if np.allclose(best,solution*middle,atol=1e-9,rtol=0):
                high = middle
            else:
                low = middle
        # A component needed to find the detour may be redundant once both
        # arms have moved. Drop or shorten it before emitting rotation keys.
        for bone in np.argsort(-np.linalg.norm(best, axis=1)):
            if not np.any(best[bone]):
                continue
            initial = best[bone].copy()
            for fraction in (0., .25, .5, .75, .9):
                candidate = best.copy()
                candidate[bone] = initial*fraction
                consider(candidate)
                yield None
                if np.array_equal(best,candidate):
                    break
    keys.write(best)
    return best
