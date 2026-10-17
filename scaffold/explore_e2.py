"""SCAFFOLD exploratory E2 (DEVIATIONS.md): is A's start x network interaction specific to the abstraction, or generic to saturated adapters?
Interaction(adapter) = (G(raw) - G(prior)) - (G(matched) - G(base)), per B seed, for A adapters (6) and for the shuf and single controls (3 each).
Reading fixed in advance (E2): shuf interaction >= 0.5 x A's -> 'generic saturation'; <= 0.25 x A's with its 95 % CI excluding A's point
estimate -> 'abstraction-specific'; otherwise 'unresolved'. single reported alongside.   python explore_e2.py"""
import json, os
import explore_analyze as EX          # loads S2 (stage2 cells) and prints E1 again
import analyze as AN
HERE = os.path.dirname(os.path.abspath(__file__)); S2 = EX.S2; out = {}
for rule in ('bind', 'bext'):
    base = {bs: S2[(rule, bs, 'ST2_20260919')]['base']['G'] for bs in (101, 102, 103)}
    def inter(names):
        C = {}
        for ad in names:
            for bs in (101, 102, 103):
                a = S2.get((rule, bs, ad), {})
                if all(k in a for k in ('A_raw', 'prior', 'A_matched')): C.setdefault(ad, {})[bs] = (a['A_raw']['G'] - a['prior']['G']) - (a['A_matched']['G'] - base[bs])
        return C
    A = inter(EX.ADS); est = {'A': AN.hboot(A)}
    for k in ('shuf', 'single'):
        C = inter(['%s_%d' % (k, s) for s in (20260919, 20260921, 20260922)])
        if C: est[k] = AN.hboot(C); est[k]['n_cells'] = sum(len(v) for v in C.values())
    a = est['A']['est']; s = est.get('shuf')
    if s is None: verdict = 'pending'
    elif s['est'] >= 0.5 * a: verdict = 'generic saturation'
    elif s['est'] <= 0.25 * a and (a > s['ci95'][1] or a < s['ci95'][0]): verdict = 'abstraction-specific'
    else: verdict = 'unresolved'
    out[rule] = {'estimates': est, 'verdict': verdict}
    print(rule, verdict, {k: (round(v['est'], 3), [round(x, 3) for x in v['ci95']]) for k, v in est.items()})
json.dump(out, open(os.path.join(HERE, 'results', 'EXPLORE_E2.json'), 'w'), indent=1)
