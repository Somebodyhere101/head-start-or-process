"""SCAFFOLD exploratory E1 (DEVIATIONS.md), post hoc: start x network factorial and the learning-rate sensitivity benchmark.
Network effect given A's start  = G(A_raw) - G(prior)      (A's network vs base network, both starting from A's predictions)
Network effect given base start = G(A_matched) - G(base)   (= Delta_K)
Interaction = the difference of the two. LR benchmark: G(base_lrX) - G(base) on the same B seeds (results/explore_lr), and the replicate floor
(results/stage2_rep vs stage2 base). Same hierarchical bootstrap as analyze.py.   python explore_analyze.py"""
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results')
import analyze as AN


def load(tag):
    out = {}
    for f in glob.glob(os.path.join(R, tag, '*.json')):
        if os.path.basename(f) == 'ANALYSIS.json': continue
        r = json.load(open(f)); out.setdefault((r['rule'], r['bseed'], r['adapter']), {}).update(r['arms'])
    return out


S2 = load('stage2'); LR = load('explore_lr'); REP = load('stage2_rep'); res = {}
ADS = ['ST2_20260919', 'ST2_20260921', 'ST2_20260922', 'ST2b_20260920', 'P1_20260923', 'P1_20260924']
for rule in ('bind', 'bext'):
    base = {bs: S2[(rule, bs, 'ST2_20260919')]['base']['G'] for bs in (101, 102, 103)}
    C = {'net_given_Astart': {}, 'net_given_basestart': {}, 'interaction': {}}
    for ad in ADS:
        for bs in (101, 102, 103):
            a = S2[(rule, bs, ad)]; x = a['A_raw']['G'] - a['prior']['G']; y = a['A_matched']['G'] - base[bs]
            C['net_given_Astart'].setdefault(ad, {})[bs] = x; C['net_given_basestart'].setdefault(ad, {})[bs] = y; C['interaction'].setdefault(ad, {})[bs] = x - y
    est = {k: AN.hboot(v) for k, v in C.items()}
    for mult in ('2', '4'):
        arm = 'base_lr' + mult; d = {bs: LR[(rule, bs, 'ST2_20260919')][arm]['G'] - base[bs] for bs in (101, 102, 103) if (rule, bs, 'ST2_20260919') in LR and arm in LR[(rule, bs, 'ST2_20260919')]}
        if d: est['lr_x' + mult] = {'per_bseed': d, 'mean': sum(d.values()) / len(d)}
    rep = REP.get((rule, 101, 'ST2_20260919'), {}).get('base')
    if rep: est['replicate_floor_b101'] = {'G_rep': rep['G'], 'G_orig': base[101], 'diff': rep['G'] - base[101]}
    est['consistency'] = {k: sum(1 for ad in ADS for bs in (101, 102, 103) if C[k][ad][bs] > 0) for k in C}
    res[rule] = est
    print(rule, {k: (round(v['est'], 3), [round(x, 3) for x in v['ci95']]) if 'est' in v else v for k, v in est.items()})
json.dump(res, open(os.path.join(R, 'EXPLORE_E1.json'), 'w'), indent=1)
