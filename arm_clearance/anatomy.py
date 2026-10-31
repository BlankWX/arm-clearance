"""Shared bone identity/classification contract; no feature dependencies."""
import numpy as np

def _bone_ids(meta):
    names = list(meta['bones'])
    index = {name: i for i, name in enumerate(names)}
    target = list(meta.get('arm_target_ids', ()))
    all_targets = list(meta.get('target_ids', ()))
    if len(target) != 4:
        if len(all_targets) == 4:
            target = all_targets
        elif len(all_targets) == 6:
            target = [all_targets[i] for i in (0, 1, 3, 4)]
        else:
            target = [index[name] for name in ('腕.L', 'ひじ.L', '腕.R', 'ひじ.R')]
    wrists = list(meta.get('wrist_ids', ()))
    if len(wrists) != 2:
        wrists = ([all_targets[2], all_targets[5]] if len(all_targets) == 6
                  else [index[name] for name in ('手首.L', '手首.R')])
    ids = target + wrists
    if len(set(ids)) != 6 or any(i < 0 or i >= len(names) for i in ids):
        raise ValueError('Full surface verification requires six distinct arm/wrist bones')
    return target, wrists


def classify_bones(meta):
    """Return tree-derived ``side`` (-1/body, 0/L, 1/R) and ``region`` arrays.

    Regions are 0/body, 1/upper arm, 2/forearm, 3/hand and every wrist
    descendant.  Twist/helper descendants are included by ancestry.  ``finger``
    identifies a named finger branch only for optional same-side diagnostics;
    it does not control inclusion in the hand collision surface.
    """
    names = list(meta['bones'])
    parents = np.asarray(meta['parents'], dtype=np.int32)
    if len(parents) != len(names):
        raise ValueError('Bone parent count does not match bone count')
    arm_ids, wrist_ids = _bone_ids(meta)
    roots = {}
    for side in range(2):
        upper, fore = arm_ids[2 * side:2 * side + 2]
        wrist = wrist_ids[side]
        roots.update({upper: (side, 1), fore: (side, 2), wrist: (side, 3)})
    side = np.full(len(names), -1, dtype=np.int8)
    region = np.zeros(len(names), dtype=np.int8)
    finger = np.full(len(names), -1, dtype=np.int32)
    for bone in range(len(names)):
        chain = []
        current = bone
        seen = set()
        while current >= 0:
            if current >= len(names) or current in seen:
                raise ValueError('Invalid or cyclic bone hierarchy')
            seen.add(current)
            chain.append(current)
            if current in roots:
                side[bone], region[bone] = roots[current]
                if region[bone] == 3:
                    # The highest named finger ancestor below the wrist is
                    # stable across all phalanges and driven helper bones.
                    candidates = [i for i in chain[:-1] if '指' in names[i]]
                    if candidates:
                        finger[bone] = candidates[-1]
                break
            current = int(parents[current])
    return {'side': side, 'region': region, 'finger': finger,
            'arm_ids': arm_ids, 'wrist_ids': wrist_ids}
