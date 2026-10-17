"""Figure data for the SCAFFOLD paper (stdlib only).
1. paper/fig/curves_<tag>_<rule>.dat: mean balanced held-out accuracy at each evaluation point, per arm, averaged over A adapters and B seeds.
   Columns: frac base A_raw prior A_matched ref ref_lo ref_hi (frac = share of budget seen). 'ref' is the mean of four reference
   configurations (base and the three output-matched perm runs; E3), each averaged over B seeds; ref_lo/ref_hi are their min and max.
2. paper/fig/refruns_<rule>_b<seed>.dat: the four individual reference trajectories at one B seed (columns: frac base p19 p21 p22).
   python paper/make_figs.py stage2"""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(os.path.dirname(HERE), 'results'); OUT = os.path.join(HERE, 'fig'); os.makedirs(OUT, exist_ok=True)
ARMS = ['base', 'A_raw', 'prior', 'A_matched']; CTRL = ('shuf', 'single', 'perm', 'q06shuf', 'q06single'); PERM = ['perm_20260919', 'perm_20260921', 'perm_20260922']
curve = lambda arm: [arm['t0']['bal']] + [t['bal'] for t in arm['trace']]
for tag in sys.argv[1:] or ['stage2']:
    cells = [json.load(open(f)) for f in glob.glob(os.path.join(R, tag, '*.json')) if 'ANALYSIS' not in f]
    for rule in sorted({c['rule'] for c in cells}):
        acc = {arm: {} for arm in ARMS}; byk = {}
        for c in cells:
            if c['rule'] != rule: continue
            byk[c['adapter'], c['bseed']] = c['arms']
            if c['adapter'].split('_')[0] in CTRL: continue
            for arm in ARMS:
                if arm not in c['arms']: continue
                for i, v in enumerate(curve(c['arms'][arm])): acc[arm].setdefault(i, []).append(v)
        if not acc['base']: continue
        seeds = sorted({s for (_, s) in byk}); npts = max(acc['base'])
        refcfg = [[curve(byk['ST2_20260919', s]['base']) for s in seeds]] + [[curve(byk[p, s]['A_matched']) for s in seeds] for p in PERM if (p, seeds[0]) in byk]
        refmean = [[sum(run[i] for run in cfg) / len(cfg) for i in range(npts + 1)] for cfg in refcfg]
        rows = []
        for i in range(npts + 1):
            arms = ' '.join('%.4f' % (sum(acc[a].get(i, [float('nan')])) / max(1, len(acc[a].get(i, [])))) for a in ARMS)
            col = [m[i] for m in refmean]
            rows.append('%.4f %s %.4f %.4f %.4f' % (i / npts, arms, sum(col) / len(col), min(col), max(col)))
        open(os.path.join(OUT, 'curves_%s_%s.dat' % (tag, rule)), 'w').write('frac ' + ' '.join(ARMS) + ' ref ref_lo ref_hi\n' + '\n'.join(rows) + '\n')
        for s in seeds:
            runs = [curve(byk['ST2_20260919', s]['base'])] + [curve(byk[p, s]['A_matched']) for p in PERM if (p, s) in byk]
            rows = ['%.4f ' % (i / npts) + ' '.join('%.4f' % r[i] for r in runs) for i in range(npts + 1)]
            open(os.path.join(OUT, 'refruns_%s_b%d.dat' % (rule, s)), 'w').write('frac base p19 p21 p22\n' + '\n'.join(rows) + '\n')
        print(tag, rule, 'points', npts + 1, 'reference configurations', len(refcfg))
