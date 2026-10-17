"""Appendix tables for the SCAFFOLD paper, written from the result files (stdlib only) into paper/tab_*.tex.
   python paper/make_tables.py"""
import glob, json, math, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(os.path.dirname(HERE), 'results')
ADS = ['ST2_20260919', 'ST2_20260921', 'ST2_20260922', 'ST2b_20260920', 'P1_20260923', 'P1_20260924']; BS = (101, 102, 103)
S2 = {}
for f in glob.glob(os.path.join(R, 'stage2', '*.json')):
    if 'ANALYSIS' in f: continue
    r = json.load(open(f)); S2.setdefault((r['rule'], r['bseed'], r['adapter']), {}).update(r['arms'])
tex = lambda s: s.replace('_', r'\_')
m = lambda xs: sum(xs) / len(xs)
sg = lambda x: '$%+.3f$' % x                                                    # signed, with a true minus sign
def sci(x):                                                                     # 3 significant digits as a x 10^k
    k = int(math.floor(math.log10(abs(x)))); a = x / 10 ** k
    return '$%s$' % ('%.3g' % x if 0 <= k <= 2 else r'%.3g\times10^{%d}' % (a, k))

# 1. per-adapter contrasts (mean over B seeds)
rows = []
for ad in ADS:
    cells = []
    for rule in ('bind', 'bext'):
        d = {k: [] for k in ('raw', 'start', 'K', 'S')}
        for bs in BS:
            a = S2[(rule, bs, ad)]; b = S2[(rule, bs, 'ST2_20260919')]['base']
            d['raw'].append(a['A_raw']['G'] - b['G']); d['start'].append(a['prior']['G'] - b['G']); d['K'].append(a['A_matched']['G'] - b['G'])
            d['S'].append(a['A_matched']['scaffold']['bal'] - b['scaffold']['bal'])
        cells += [sg(m(d[k])) for k in ('raw', 'start', 'K', 'S')]
    rows.append(tex(ad) + ' & ' + ' & '.join(cells) + r' \\')
open(os.path.join(HERE, 'tab_peradapter.tex'), 'w').write('\n'.join(rows) + '\n')

# 2. control adapters
rows = []
for k in ('shuf', 'single', 'perm'):
    for s in (20260919, 20260921, 20260922):
        r = json.load(open(os.path.join(R, 'controls', '%s_%d.json' % (k, s))))
        Gs = {rule: m([S2[(rule, bs, '%s_%d' % (k, s))]['A_matched']['G'] - S2[(rule, bs, 'ST2_20260919')]['base']['G'] for bs in BS]) for rule in ('bind', 'bext')}
        rows.append('%s\\_%d & %.3f & %.2f & %s & %s \\\\' % (k, s, r['software_interpret_meaning_acc']['control'], r['normalised_letter_kl_to_base']['control'], sg(Gs['bind']), sg(Gs['bext'])))
open(os.path.join(HERE, 'tab_controls.tex'), 'w').write('\n'.join(rows) + '\n')

# 3. kernel statistics at t = 0 (B seed 101): base, A_matched per adapter (range), controls (range)
rows = []
for rule in ('bind', 'bext'):
    F = {}
    for f in glob.glob(os.path.join(R, 'forecast', '%s_b101_*.json' % rule)):
        r = json.load(open(f)); name = os.path.splitext(os.path.basename(r['adapter']))[0]
        for arm, st in r['arms'].items(): F[(name if arm != 'base' else 'base', arm)] = st
    groups = {'base': [v for (n, a), v in F.items() if a == 'base'], '$A$ (matched frame)': [v for (n, a), v in F.items() if a == 'A_matched' and n in ADS],
              'shuf / single (matched)': [v for (n, a), v in F.items() if a == 'A_matched' and n.split('_')[0] in ('shuf', 'single')], 'perm (matched)': [v for (n, a), v in F.items() if a == 'A_matched' and n.split('_')[0] == 'perm']}
    for g, vs in groups.items():
        if not vs: continue
        rng = lambda key, fmt: (fmt(min(v[key] for v in vs))) + ('' if len(vs) == 1 else '--' + fmt(max(v[key] for v in vs)))
        rows.append('%s & %s & %s & %s & %s & %s \\\\' % ('B\\_ind' if rule == 'bind' else 'B\\_ext', g, rng('raw_scale', sci), rng('F_res', lambda x: '%.1f' % x),
                                                     rng('F_sign', lambda x: '$%.1f$' % x), rng('CKA_type', lambda x: '%.2f' % x)))
open(os.path.join(HERE, 'tab_kernel.tex'), 'w').write('\n'.join(rows) + '\n')

# 5. E3 robustness: contrasts against the four-run reference, three interval methods
E3 = json.load(open(os.path.join(R, 'EXPLORE_E3.json')))
ci = lambda v: '[%s, %s]' % tuple(('$%+.3f$' % x) for x in v)
LAB = [('K', r'$\Delta_K$ ($A$)'), ('K_shuf', r'$\Delta_K$ (shuf)'), ('K_single', r'$\Delta_K$ (single)'), ('raw', r'$\Delta_{\text{raw}}$'),
       ('start', r'$\Delta_{\text{start}}$'), ('interaction', 'Interaction')]
rows = []
for rule, name in (('bind', 'B\\_ind'), ('bext', 'B\\_ext')):
    G = E3[rule]['G']
    rows.append(r'\multicolumn{5}{l}{\emph{%s} (reference SD of $G$ within seed %.3f)}\\' % (name, G['noise']['pooled_within_sd']))
    for k, lab in LAB:
        v = G[k]; p = ('%.3f' % min(1.0, 2 * min(v['perm_p_greater'], v['perm_p_less']))) if 'perm_p_greater' in v else '--'
        rows.append('%s & %s & %s & %s & %s \\\\' % (lab, sg(v['est']), ci(v['boot3_ci95']), ci(v['t_ci90']) if 't_ci90' in v else '--', p))
    sc = E3[rule]['scaffold']
    for k, lab in (('S_ref', r'$S$ (abrupt unmerge)'), ('S_fade_ref', r'$S_{\text{fade}}$'), ('dependence', r'with $A$ $-$ without $A$')):
        rows.append('%s & %s & %s & -- & -- \\\\' % (lab, sg(sc[k]['est']), ci(sc[k]['ci95'])))
open(os.path.join(HERE, 'tab_robust.tex'), 'w').write('\n'.join(rows) + '\n')

# 6. reference runs: G of base and the three perm replicates per seed
rows = []
for rule, name in (('bind', 'B\\_ind'), ('bext', 'B\\_ext')):
    for s in ('101', '102', '103'):
        rr = E3[rule]['G']['noise']['ref_runs'][s]; ff = E3[rule]['final']['noise']['ref_runs'][s]
        rows.append('%s & %s & %s & %s \\\\' % (name, s, ' & '.join('%.3f' % x for x in rr), ' & '.join('%.2f' % x for x in ff)))
open(os.path.join(HERE, 'tab_refruns.tex'), 'w').write('\n'.join(rows) + '\n')

# 4. PC-H2 calibration
sel = json.load(open(os.path.join(R, 'pc_calib', 'selection.json')))
rows = ['$\\times$%s & %s & %s \\\\' % (f, sg(g['bind']), sg(g['bext'])) for f, g in sel['factors'].items()]
open(os.path.join(HERE, 'tab_pccalib.tex'), 'w').write('\n'.join(rows) + '\n')
print('tables written')
