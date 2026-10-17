"""SCAFFOLD Stage-2 analysis (PREREGISTRATION.md sections 3 and 7). Reads results/<tag>/*.json written by h2run.py.
Per (A adapter, B rule, B seed): Delta_raw = G(A_raw) - G(base); Delta_start = G(prior) - G(base); Delta_K = G(A_matched) - G(base);
Delta_K_vs_ctrl = Delta_K(A) - Delta_K(control of the same training seed); S = scaffold(A_matched) - scaffold(base); S_fade = final(A_matched_fade) - final(base).
PC-H2 (sensitivity): G and scaffold of PC_H2 minus base. Hierarchical bootstrap (2000; adapters, then B seeds), 95 % CI for effects,
90 % CI for TOST equivalence at +-0.05, Holm across the four primary contrasts within each B rule. Verdict per the preregistered table.
   python analyze.py --tag stage2"""
import argparse, glob, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SESOI, EQ = 0.10, 0.05
CTRL_OF = lambda ad: None


def load(tag):
    rows = {}
    for f in glob.glob(os.path.join(HERE, 'results', tag, '*.json')):
        r = json.load(open(f)); key = (r['rule'], r['bseed'], r['adapter'])
        rows.setdefault(key, {}).update(r['arms'])
    return rows


def hboot(cells, fn=None, n=2000, seed=0):            # DEVIATIONS D5: 'fn' was an unused required argument (crash); now optional, logic unchanged
    """cells: {adapter: {bseed: value}}; resample adapters, then B seeds within adapter; returns point estimate and bootstrap list."""
    rng = random.Random(seed); ads = sorted(cells); point = sum(sum(v.values()) / len(v) for v in cells.values()) / len(cells); bs = []
    for _ in range(n):
        acc = []
        for _a in range(len(ads)):
            ad = ads[rng.randrange(len(ads))]; ks = sorted(cells[ad]); acc.append(sum(cells[ad][ks[rng.randrange(len(ks))]] for _k in range(len(ks))) / len(ks))
        bs.append(sum(acc) / len(acc))
    bs.sort(); q = lambda p: bs[int(p * (len(bs) - 1))]
    return {'est': point, 'ci95': [q(0.025), q(0.975)], 'ci90': [q(0.05), q(0.95)], 'p_le0': sum(b <= 0 for b in bs) / len(bs)}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(order): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='stage2'); ap.add_argument('--controls', default='shuf,single,perm'); a = ap.parse_args()
    R = load(a.tag); out = {}
    base = {(k[0], k[1]): v['base'] for k, v in R.items() if 'base' in v}
    pc = {(k[0], k[1]): v['PC_H2'] for k, v in R.items() if 'PC_H2' in v}
    for rule in ('bind', 'bext'):
        C = {'raw': {}, 'start': {}, 'K': {}, 'S': {}, 'S_fade': {}}; Cc = {c: {} for c in a.controls.split(',')}
        for (rl, bs, ad), arms in R.items():
            if rl != rule or (rl, bs) not in base or ad.split('_')[0] in a.controls.split(','): continue
            b = base[(rl, bs)]
            put = lambda name, val: C[name].setdefault(ad, {}).__setitem__(bs, val)
            if 'A_raw' in arms: put('raw', arms['A_raw']['G'] - b['G'])
            if 'prior' in arms: put('start', arms['prior']['G'] - b['G'])
            if 'A_matched' in arms: put('K', arms['A_matched']['G'] - b['G']); put('S', arms['A_matched']['scaffold']['bal'] - b['scaffold']['bal'])
            if 'A_matched_fade' in arms: put('S_fade', arms['A_matched_fade']['final_own']['bal'] - b['final_own']['bal'])
            seed = ad.split('_')[-1]
            for c in Cc:
                ck = (rl, bs, '%s_%s' % (c, seed))
                if ck in R and 'A_matched' in R[ck] and 'A_matched' in arms: Cc[c].setdefault(ad, {})[bs] = arms['A_matched']['G'] - R[ck]['A_matched']['G']
        est = {k: hboot(v) for k, v in C.items() if v}; est.update({'K_minus_' + c: hboot(v) for c, v in Cc.items() if v})
        prim = [k for k in ('K', 'K_minus_shuf', 'K_minus_single', 'S') if k in est]; adj = holm([est[k]['p_le0'] for k in prim])
        for k, p in zip(prim, adj): est[k]['p_holm'] = p
        eqK = 'K' in est and EQ * -1 < est['K']['ci90'][0] and est['K']['ci90'][1] < EQ
        pcs = {bs: v for (rl, bs), v in pc.items() if rl == rule}
        sens = {'G_minus_base': {bs: v['G'] - base[(rule, bs)]['G'] for bs, v in pcs.items() if (rule, bs) in base}, 'scaffold_minus_base': {bs: v['scaffold']['bal'] - base[(rule, bs)]['scaffold']['bal'] for bs, v in pcs.items() if (rule, bs) in base}}
        detected = bool(sens['G_minus_base']) and sum(sens['G_minus_base'].values()) / len(sens['G_minus_base']) >= SESOI
        verdict = 'uninterpretable (PC-H2 not detected)' if not detected else (
            'H2 (process)' if 'K' in est and est['K']['est'] >= SESOI and est['K']['ci95'][0] > 0 and all(est.get('K_minus_' + c, {'est': 0})['est'] >= SESOI for c in Cc) else
            'H1 (head start only)' if eqK else 'inconclusive')
        if verdict.startswith('H2') and 'S' in est and est['S']['est'] >= SESOI and est['S']['ci95'][0] > 0: verdict = 'H2-strong (scaffold)'
        out[rule] = {'estimates': est, 'equivalence_K': eqK, 'pc_h2': sens, 'pc_h2_detected': detected, 'verdict': verdict}
        print(rule, verdict, {k: (round(v['est'], 3), [round(x, 3) for x in v['ci95']]) for k, v in est.items()})
    json.dump(out, open(os.path.join(HERE, 'results', a.tag, 'ANALYSIS.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
