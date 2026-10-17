"""SCAFFOLD Stage 3 analysis (PREREGISTRATION_STAGE3.md section 6). Reads results/stage3_<model>/*.json (h2run.py cells).
Per model x task: Delta_raw, Delta_start, Delta_K (+ TOST), S, S_fade, Delta_K - Delta_K(shuf / single) (paired by training seed), the
sensitivity benchmark G(base_lr2) - G(base), the saturation interaction for A and controls, and the verdicts and confirmatory claims C1-C4.
   python analyze3.py"""
import glob, json, os
import analyze as AN
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results')
MODELS = {'q06': {'tag': 'stage3_q06', 'ads': ['ST2s_20260919', 'ST2s_20260921', 'ST2s_20260922'], 'ctrl': 'q06%s_%s', 'rules': ['bind', 'bext', 'bpara']},
          'q17': {'tag': 'stage3_q17', 'ads': ['ST2_20260919', 'ST2_20260921', 'ST2_20260922', 'ST2b_20260920', 'P1_20260923', 'P1_20260924'], 'ctrl': '%s_%s', 'rules': ['bpara']}}
SEEDS = ('20260919', '20260921', '20260922'); BS = (201, 202, 203)


def load(tag):
    out = {}
    for f in glob.glob(os.path.join(R, tag, '*.json')):
        r = json.load(open(f)); out.setdefault((r['rule'], r['bseed'], r['adapter']), {}).update(r['arms'])
    return out


def main():
    res = {}
    for model, M in MODELS.items():
        S = load(M['tag'])
        for rule in M['rules']:
            key = '%s_%s' % (model, rule); bc = {bs: S.get((rule, bs, M['ads'][0]), {}) for bs in BS}
            if not all('base' in bc[bs] for bs in BS): res[key] = {'status': 'incomplete'}; continue
            base = {bs: bc[bs]['base'] for bs in BS}
            C = {k: {} for k in ('raw', 'start', 'K', 'S', 'S_fade', 'inter')}; CC = {c: {'K': {}, 'inter': {}} for c in ('shuf', 'single')}; DK = {c: {} for c in ('shuf', 'single')}
            def put(d, ad, bs, v): d.setdefault(ad, {})[bs] = v
            for ad in M['ads']:
                for bs in BS:
                    a = S.get((rule, bs, ad), {})
                    if not all(k in a for k in ('A_raw', 'prior', 'A_matched', 'A_matched_fade')): continue
                    b = base[bs]; K = a['A_matched']['G'] - b['G']
                    put(C['raw'], ad, bs, a['A_raw']['G'] - b['G']); put(C['start'], ad, bs, a['prior']['G'] - b['G']); put(C['K'], ad, bs, K)
                    put(C['S'], ad, bs, a['A_matched']['scaffold']['bal'] - b['scaffold']['bal']); put(C['S_fade'], ad, bs, a['A_matched_fade']['final_own']['bal'] - b['final_own']['bal'])
                    put(C['inter'], ad, bs, (a['A_raw']['G'] - a['prior']['G']) - K)
                    sd = ad.split('_')[-1]
                    for c in ('shuf', 'single'):
                        cc = S.get((rule, bs, M['ctrl'] % (c, sd)), {})
                        if 'A_matched' in cc: put(DK[c], ad, bs, K - (cc['A_matched']['G'] - b['G']))
            for c in ('shuf', 'single'):
                for sd in SEEDS:
                    name = M['ctrl'] % (c, sd)
                    for bs in BS:
                        cc = S.get((rule, bs, name), {})
                        if all(k in cc for k in ('A_raw', 'prior', 'A_matched')):
                            Kc = cc['A_matched']['G'] - base[bs]['G']; put(CC[c]['K'], name, bs, Kc); put(CC[c]['inter'], name, bs, (cc['A_raw']['G'] - cc['prior']['G']) - Kc)
            est = {k: AN.hboot(v) for k, v in C.items() if v}
            for c in ('shuf', 'single'):
                if DK[c]: est['K_minus_' + c] = AN.hboot(DK[c])
                if CC[c]['inter']: est['inter_' + c] = AN.hboot(CC[c]['inter']); est['K_' + c] = AN.hboot(CC[c]['K'])
            lr = {bs: bc[bs]['base_lr2']['G'] - base[bs]['G'] for bs in BS if 'base_lr2' in bc[bs]}; sens = sum(lr.values()) / len(lr) if lr else None
            sens_ok = sens is not None and abs(sens) >= 0.07
            K = est.get('K'); eq = K is not None and -0.05 < K['ci90'][0] and K['ci90'][1] < 0.05
            h2 = K is not None and K['est'] >= 0.10 and K['ci95'][0] > 0 and all(est.get('K_minus_' + c, {'est': 0})['est'] >= 0.10 for c in ('shuf', 'single'))
            scaf = h2 and est['S']['est'] >= 0.10 and est['S']['ci95'][0] > 0
            verdict = 'H2-scaffold' if scaf else 'H2' if h2 else ('H1' if eq and sens_ok else ('uninterpretable (sensitivity not met)' if eq and not sens_ok else 'inconclusive'))
            ai = est.get('inter', {}).get('est'); si = est.get('inter_shuf')
            sat = 'absent' if ai is None or ai < 0.05 else ('generic saturation' if si and si['est'] >= 0.5 * ai else ('abstraction-specific' if si and si['est'] <= 0.25 * ai and not (si['ci95'][0] <= ai <= si['ci95'][1]) else 'unresolved'))
            res[key] = {'estimates': est, 'sensitivity_lr2': {'per_bseed': lr, 'mean': sens, 'met': sens_ok}, 'equivalence_K': eq, 'verdict': verdict, 'saturation': sat,
                        'no_scaffold': not (est.get('S', {'est': 0})['est'] >= 0.10 and est.get('S', {'ci95': [0, 0]})['ci95'][0] > 0)}
            print(key, verdict, '| sat:', sat, '| lr2 %.3f' % (sens if sens is not None else float('nan')), {k: (round(v['est'], 3), [round(x, 3) for x in v['ci95']]) for k, v in est.items()})
    claims = {'C1': all(res.get('q06_%s' % r, {}).get('verdict') == 'H1' for r in ('bind', 'bpara')), 'C2': res.get('q17_bpara', {}).get('verdict') == 'H1',
              'C3': all(v.get('no_scaffold', False) for v in res.values() if 'estimates' in v), 'C4': all(v.get('saturation') in ('absent', 'generic saturation') for v in res.values() if 'estimates' in v)}
    res['claims'] = claims; print('claims', claims)
    json.dump(res, open(os.path.join(R, 'STAGE3_ANALYSIS.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
