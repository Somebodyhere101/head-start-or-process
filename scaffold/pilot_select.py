"""SCAFFOLD Stage 3 budget rule (PREREGISTRATION_STAGE3.md section 5), applied mechanically to the pilot.
Pilot cells: results/stage3_pilot_<model>/<rule>_b{1,2,3}_<adapter>.json (base arm, 6 passes, eval every 96 of 192-situation passes).
p* = smallest pass count at which >= 1 seed reaches balanced accuracy >= 0.85; if p* = 1 and all 3 seeds reach it within 1 pass: half a pass
(96 situations); if never within 6 passes: 6 passes, flagged 'hard'. Writes results/stage3_budgets.json: {model_rule: passes, n_train, eval_every, flag}."""
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results'); out = {}
for model in ('q06', 'q17'):
    for f in glob.glob(os.path.join(R, 'stage3_pilot_%s' % model, '*.json')):
        r = json.load(open(f)); key = '%s_%s' % (model, r['rule'])
        if 'base' not in r['arms']: continue
        first = next((t['seen'] for t in r['arms']['base']['trace'] if t['bal'] >= 0.85), None)
        out.setdefault(key, {'first_reach': {}})['first_reach'][r['bseed']] = first
B = {}
for key, d in out.items():
    fr = d['first_reach']; reach = {p: sum(1 for v in fr.values() if v is not None and v <= 192 * p) for p in range(1, 7)}
    pstar = next((p for p in range(1, 7) if reach[p] >= 1), None)
    if pstar is None: B[key] = {'passes': 6, 'n_train': 192, 'eval_every': 144, 'flag': 'hard'}
    elif pstar == 1 and reach[1] == 3: B[key] = {'passes': 1, 'n_train': 96, 'eval_every': 12, 'flag': 'half_pass'}
    else: B[key] = {'passes': pstar, 'n_train': 192, 'eval_every': 24 * pstar, 'flag': ''}
    B[key].update(first_reach=fr, reach_by_pass=reach)
json.dump(B, open(os.path.join(R, 'stage3_budgets.json'), 'w'), indent=1); print(json.dumps(B, indent=1))
