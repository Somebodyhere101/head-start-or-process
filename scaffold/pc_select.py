"""SCAFFOLD Stage-2 addendum: choose the PC-H2 amplification factor on the PILOT B seeds (1-3) with the Stage-2 budgets, before any Stage-2 run.
Rule (frozen in PREREGISTRATION_STAGE2.md): the smallest factor in {2, 4, 8} whose mean G gain over base (same pilot seeds, same budget) is
>= 0.10 on BOTH B rules. If none qualifies, PC-H2 is recorded as failed calibration and the Stage-2 sensitivity condition cannot be met
(all verdicts 'uninterpretable'), which is reported. Writes results/pc_calib/factor.txt and results/pc_calib/selection.json."""
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results')


def G(tag, arm):
    out = {}
    for f in glob.glob(os.path.join(R, tag, '*.json')):
        r = json.load(open(f))
        if arm in r.get('arms', {}): out[(r['rule'], r['bseed'])] = r['arms'][arm]['G']
    return out


base = G('pc_calib_base', 'base'); sel = {'base': {'%s_%d' % k: v for k, v in base.items()}, 'factors': {}}; chosen = None
for f in (2, 4, 8):
    pc = G('pc_calib_f%d' % f, 'PC_H2'); gains = {}
    for rule in ('bind', 'bext'):
        d = [pc[k] - base[k] for k in pc if k[0] == rule and k in base]; gains[rule] = sum(d) / len(d) if d else None
    sel['factors'][f] = gains
    if chosen is None and all(g is not None and g >= 0.10 for g in gains.values()): chosen = f
sel['chosen'] = chosen; os.makedirs(os.path.join(R, 'pc_calib'), exist_ok=True)
json.dump(sel, open(os.path.join(R, 'pc_calib', 'selection.json'), 'w'), indent=1)
open(os.path.join(R, 'pc_calib', 'factor.txt'), 'w').write(str(chosen if chosen is not None else 'none')); print(json.dumps(sel))
