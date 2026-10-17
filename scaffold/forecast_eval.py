"""Forecast evaluation (stdlib only).
1. Preregistered (PREREGISTRATION_STAGE2.md section 6): Spearman of F_res, F_sign and CKA_type at t = 0 with G, across arm x cell. forecast.py
   was run on B seed 101 only, so the cells are (adapter, arm) pairs at seed 101, per rule. Criterion: Spearman >= 0.5.
2. E4 (exploratory, post hoc; prompted by review): across adapters, does a t = 0 kernel statistic of the output-matched frame rank the mean
   output-matched advantage Delta_K (mean over B seeds; the shared base cancels in the ranking)? All four statistics are reported, for the six
   A adapters (exact permutation p over 720 orderings, two-sided) and for all 15 networks (A + shuf/single/perm; Monte Carlo p).
   python forecast_eval.py  ->  results/FORECAST_EVAL.json"""
import glob, itertools, json, os, random
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results')
ADS = ['ST2_20260919', 'ST2_20260921', 'ST2_20260922', 'ST2b_20260920', 'P1_20260923', 'P1_20260924']; SEEDS = (101, 102, 103)
CTRL = ['%s_%d' % (k, s) for k in ('shuf', 'single', 'perm') for s in (20260919, 20260921, 20260922)]
STATS = ['F_res', 'F_sign', 'CKA_type', 'raw_scale']


def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y); mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float('nan')


def perm_p(x, y, exact=True, n=20000):
    obs = abs(spearman(x, y))
    if exact:
        ps = list(itertools.permutations(y)); return sum(abs(spearman(x, list(p))) >= obs - 1e-12 for p in ps) / len(ps)
    rng = random.Random(0); y = list(y); c = 0
    for _ in range(n): rng.shuffle(y); c += abs(spearman(x, y)) >= obs - 1e-12
    return c / n


S = {}
for f in glob.glob(os.path.join(R, 'stage2', '*.json')):
    if 'ANALYSIS' in f: continue
    r = json.load(open(f)); S.setdefault((r['rule'], r['bseed'], r['adapter']), {}).update(r['arms'])
out = {}
for rule in ('bind', 'bext'):
    F = {}
    for f in glob.glob(os.path.join(R, 'forecast', '%s_b101_*.json' % rule)):
        r = json.load(open(f)); name = os.path.splitext(os.path.basename(r['adapter']))[0]
        for arm, st in r['arms'].items(): F[('base' if arm == 'base' else name, arm)] = st
    # 1. preregistered: arm x cell at seed 101
    pairs = []
    for (name, arm), st in F.items():
        cell = S.get((rule, 101, 'ST2_20260919' if name == 'base' else name), {})
        if arm in cell: pairs.append((st, cell[arm]['G'], name, arm))
    pre = {k: spearman([p[0][k] for p in pairs], [p[1] for p in pairs]) for k in STATS}
    # 2. E4: per-network mean Delta_K vs t = 0 statistic in the matched frame
    dK = lambda n: sum(S[(rule, s, n)]['A_matched']['G'] - S[(rule, s, 'ST2_20260919')]['base']['G'] for s in SEEDS) / 3
    e4 = {}
    for grp, names, exact in (('A6', ADS, True), ('all15', ADS + CTRL, False)):
        names = [n for n in names if (n, 'A_matched') in F]; y = [dK(n) for n in names]
        e4[grp] = {k: {'rho': spearman([F[(n, 'A_matched')][k] for n in names], y), 'p': perm_p([F[(n, 'A_matched')][k] for n in names], y, exact)} for k in STATS}
        e4[grp]['n'] = len(names)
    out[rule] = {'prereg_n_pairs': len(pairs), 'prereg_spearman': pre, 'prereg_pass': {k: v >= 0.5 for k, v in pre.items() if k != 'raw_scale'}, 'E4': e4,
                 'F_sign_A': {n: F[(n, 'A_matched')]['F_sign'] for n in ADS}, 'F_sign_base': F[('base', 'base')]['F_sign'], 'dK_A': {n: dK(n) for n in ADS}}
    print(rule, 'prereg (n=%d):' % len(pairs), {k: round(v, 3) for k, v in pre.items()})
    for grp in ('A6', 'all15'):
        print('   E4 %-5s' % grp, {k: (round(v['rho'], 3), round(v['p'], 4)) for k, v in e4[grp].items() if k != 'n'})
    print('   F_sign A:', {n: round(v, 1) for n, v in out[rule]['F_sign_A'].items()}, 'base', round(out[rule]['F_sign_base'], 1))
json.dump(out, open(os.path.join(R, 'FORECAST_EVAL.json'), 'w'), indent=1)
