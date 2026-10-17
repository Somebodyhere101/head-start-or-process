"""E3 (exploratory, post hoc; DEVIATIONS.md): robustness of the Stage-2 contrasts to (i) the crossed adapter x B-seed design, (ii) the
single shared base trajectory per B seed, and (iii) the small number of clusters.
Reference R(s): mean G of the base run and the three output-matched perm runs at B seed s. perm adapters are norm-matched permutations of A's
factors with letter KL to base <= 0.01 and a base-like kernel (Appendix tables), so they serve as numerically perturbed replicates of base.
Contrasts per (adapter a, seed s): arm(a,s) - R(s). Intervals: (1) three-way bootstrap (adapters, B seeds, reference runs within seed; 20000);
(2) t interval over adapter means (df = 5); (3) stratified permutation test (labels permuted within seed between A runs and reference runs).
Also: trajectory-noise SD among reference runs, and final-accuracy contrasts.   python robustness.py"""
import glob, json, math, os, random, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results')
ADS = ['ST2_20260919', 'ST2_20260921', 'ST2_20260922', 'ST2b_20260920', 'P1_20260923', 'P1_20260924']; SEEDS = (101, 102, 103)
PERM = ['perm_20260919', 'perm_20260921', 'perm_20260922']
T5 = {0.95: 2.015, 0.975: 2.571}                                              # Student t quantiles, df = 5
S = {}
for f in glob.glob(os.path.join(R, 'stage2', '*.json')):
    if 'ANALYSIS' in f: continue
    r = json.load(open(f)); S.setdefault((r['rule'], r['bseed'], r['adapter']), {}).update(r['arms'])


def val(arm, metric):
    return arm['G'] if metric == 'G' else arm['final_own']['bal']


def refs(rule, s, metric):
    return [val(S[(rule, s, 'ST2_20260919')]['base'], metric)] + [val(S[(rule, s, p)]['A_matched'], metric) for p in PERM]


def table(rule, metric, fn, units=ADS):
    """fn(cell arms) -> value for one (unit, seed); returns {unit: {seed: value}}"""
    return {a: {s: fn(S[(rule, s, a)]) for s in SEEDS} for a in units}


def boot3(rule, metric, vals, coef_ref=-1.0, n=20000, seed=0):
    rng = random.Random(seed); units = sorted(vals); out = []
    for _ in range(n):
        us = [units[rng.randrange(len(units))] for _ in units]; ss = [SEEDS[rng.randrange(3)] for _ in SEEDS]
        Rs = {}
        for s in set(ss):
            rr = refs(rule, s, metric); Rs[s] = sum(rr[rng.randrange(len(rr))] for _ in rr) / len(rr)
        out.append(sum(vals[u][s] + coef_ref * Rs[s] for u in us for s in ss) / (len(us) * len(ss)))
    out.sort(); q = lambda p: out[int(p * (len(out) - 1))]
    return [q(0.025), q(0.975)], [q(0.05), q(0.95)]


def summarise(rule, metric, vals, coef_ref=-1.0, perm_test=False):
    Rm = {s: st.mean(refs(rule, s, metric)) for s in SEEDS}
    per_unit = {u: st.mean(vals[u][s] + coef_ref * Rm[s] for s in SEEDS) for u in vals}
    est = st.mean(per_unit.values()); out = {'est': est, 'per_unit': per_unit}
    ci95, ci90 = boot3(rule, metric, vals, coef_ref); out['boot3_ci95'] = ci95; out['boot3_ci90'] = ci90
    if len(per_unit) == 6:
        se = st.stdev(per_unit.values()) / math.sqrt(6)
        out['t_ci95'] = [est - T5[0.975] * se, est + T5[0.975] * se]; out['t_ci90'] = [est - T5[0.95] * se, est + T5[0.95] * se]
    if perm_test:                                                                 # H0: A runs exchangeable with reference runs within seed
        rng = random.Random(1); obs = est; cnt_ge = cnt_le = 0; N = 20000
        pools = {s: [vals[u][s] for u in vals] + refs(rule, s, metric) for s in SEEDS}; nA = len(vals)
        for _ in range(N):
            d = 0.0
            for s in SEEDS:
                p = pools[s][:]; rng.shuffle(p); d += st.mean(p[:nA]) - st.mean(p[nA:])
            d /= 3; cnt_ge += d >= obs - 1e-12; cnt_le += d <= obs + 1e-12
        out['perm_p_greater'] = cnt_ge / N; out['perm_p_less'] = cnt_le / N
    return out


res = {}
for rule in ('bind', 'bext'):
    res[rule] = {}
    for metric in ('G', 'final'):
        rr = {s: refs(rule, s, metric) for s in SEEDS}
        noise = {'ref_runs': rr, 'within_seed_sd': {s: st.stdev(v) for s, v in rr.items()}, 'base_minus_ref_mean': {s: v[0] - st.mean(v) for s, v in rr.items()},
                 'pooled_within_sd': math.sqrt(st.mean([st.variance(v) for v in rr.values()]))}
        d = {'noise': noise}
        arm = lambda name: (lambda c: val(c[name], metric))
        d['K'] = summarise(rule, metric, table(rule, metric, arm('A_matched')), perm_test=True)
        d['raw'] = summarise(rule, metric, table(rule, metric, arm('A_raw')))
        d['start'] = summarise(rule, metric, table(rule, metric, arm('prior')))
        if metric == 'G':
            d['interaction'] = summarise(rule, metric, table(rule, metric, lambda c: c['A_raw']['G'] - c['prior']['G'] - c['A_matched']['G']), coef_ref=+1.0)
        for k in ('shuf', 'single'):
            units = ['%s_%d' % (k, s) for s in (20260919, 20260921, 20260922)]
            d['K_' + k] = summarise(rule, metric, table(rule, metric, arm('A_matched'), units), perm_test=True)
        res[rule][metric] = d
        print('%s %-5s noise sd %.3f | base-ref %s' % (rule, metric, noise['pooled_within_sd'], {s: round(v, 3) for s, v in noise['base_minus_ref_mean'].items()}))
        for k in [x for x in d if x != 'noise']:
            v = d[k]; extra = ('  t90 [%+.3f, %+.3f]' % tuple(v['t_ci90'])) if 't_ci90' in v else ''
            pp = ('  perm p(>=) %.3f p(<=) %.3f' % (v['perm_p_greater'], v['perm_p_less'])) if 'perm_p_greater' in v else ''
            print('   %-12s %+.3f  boot95 [%+.3f, %+.3f] boot90 [%+.3f, %+.3f]%s%s' % (k, v['est'], *v['boot3_ci95'], *v['boot3_ci90'], extra, pp))

# Components of the E2 interaction (reference-free): network effect given A's own start, raw - prior (same starting predictions, A's vs base's
# network). Group comparison A (6 adapters) vs each control kind (3 adapters): two-way bootstrap over units within group and shared B seeds.
def comp(rule, units):
    return {u: {s: S[(rule, s, u)]['A_raw']['G'] - S[(rule, s, u)]['prior']['G'] for s in SEEDS} for u in units}


def boot_diff(X, Y, n=20000, seed=2):
    rng = random.Random(seed); ux, uy = sorted(X), sorted(Y); out = []
    for _ in range(n):
        ss = [SEEDS[rng.randrange(3)] for _ in SEEDS]; a = [ux[rng.randrange(len(ux))] for _ in ux]; b = [uy[rng.randrange(len(uy))] for _ in uy]
        out.append(st.mean(X[u][s] for u in a for s in ss) - st.mean(Y[u][s] for u in b for s in ss))
    out.sort(); q = lambda p: out[int(p * (len(out) - 1))]
    return [q(0.025), q(0.975)]


for rule in ('bind', 'bext'):
    A = comp(rule, ADS); c = {'A': st.mean(st.mean(v.values()) for v in A.values()), 'A_ci95': boot_diff(A, {'z': {s: 0.0 for s in SEEDS}})}
    for k in ('shuf', 'single'):
        C = comp(rule, ['%s_%d' % (k, s) for s in (20260919, 20260921, 20260922)])
        c[k] = st.mean(st.mean(v.values()) for v in C.values()); c['A_minus_' + k] = c['A'] - c[k]; c['A_minus_%s_ci95' % k] = boot_diff(A, C)
    res[rule]['raw_minus_prior'] = c
    print('%s raw-prior: A %+.3f %s | shuf %+.3f, A-shuf %+.3f %s | single %+.3f, A-single %+.3f %s' % (
        rule, c['A'], [round(x, 3) for x in c['A_ci95']], c['shuf'], c['A_minus_shuf'], [round(x, 3) for x in c['A_minus_shuf_ci95']],
        c['single'], c['A_minus_single'], [round(x, 3) for x in c['A_minus_single_ci95']]))

# Scaffold contrasts against the reference. For base and perm, theta0 + phi (A unmerged) is the network itself (perm: approximately), so the
# reference scaffold value is each reference run's scaffold evaluation. S_ref = scaffold(A_matched) - R_scaf(s); S_fade_ref uses the fade arm's
# final evaluation (A fully removed). dependence = final_own(A_matched) - scaffold(A_matched): what removing A costs the learned B (reference-free).
def ref_scaf(rule, s):
    return [S[(rule, s, 'ST2_20260919')]['base']['scaffold']['bal']] + [S[(rule, s, p)]['A_matched']['scaffold']['bal'] for p in PERM]


for rule in ('bind', 'bext'):
    Rs = {s: st.mean(ref_scaf(rule, s)) for s in SEEDS}; sc = {}
    for name, fn in (('S_ref', lambda c, s: c['A_matched']['scaffold']['bal'] - Rs[s]), ('S_fade_ref', lambda c, s: c['A_matched_fade']['final_own']['bal'] - Rs[s]),
                     ('dependence', lambda c, s: c['A_matched']['final_own']['bal'] - c['A_matched']['scaffold']['bal'])):
        V = {a: {s: fn(S[(rule, s, a)], s) for s in SEEDS} for a in ADS}
        est = st.mean(st.mean(v.values()) for v in V.values())
        rng = random.Random(3); out = []
        for _ in range(20000):                                                     # adapters x seeds x reference runs (reference part only for *_ref)
            us = [ADS[rng.randrange(6)] for _ in ADS]; ss = [SEEDS[rng.randrange(3)] for _ in SEEDS]; shift = {}
            for s in set(ss):
                rr = ref_scaf(rule, s); shift[s] = (Rs[s] - sum(rr[rng.randrange(4)] for _ in rr) / 4) if name != 'dependence' else 0.0
            out.append(st.mean(V[u][s] + shift[s] for u in us for s in ss))
        out.sort(); sc[name] = {'est': est, 'ci95': [out[int(0.025 * 19999)], out[int(0.975 * 19999)]]}
    sc['ref_scaffold_sd'] = math.sqrt(st.mean([st.variance(ref_scaf(rule, s)) for s in SEEDS]))
    res[rule]['scaffold'] = sc
    print(rule, 'scaffold', {k: (round(v['est'], 3), [round(x, 3) for x in v['ci95']]) if isinstance(v, dict) else round(v, 3) for k, v in sc.items()})
json.dump(res, open(os.path.join(R, 'EXPLORE_E3.json'), 'w'), indent=1, default=str)
