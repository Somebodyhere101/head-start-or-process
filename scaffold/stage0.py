"""SCAFFOLD Stage 0 (PREREGISTRATION.md section 5, as amended by DEVIATIONS D2): identify A's letter-agnostic representation r_A.
For one A adapter: fresh ced-world interpret items (plant + clinic = fit, software = test), each situation in all four cyclic option rotations
(letter identity decorrelated from the decision). Sites: 'final' (last prompt token), 'optS' (last token of "...shared cause affecting all three"),
'optP' (last token of "...separate causes"), 'opt2' (optS and optP jointly). Residuals after block l (hidden_states[l+1]); deltas d = h_A - h_0.
Candidates per site/layer: U_pca4 (top-4 PCs of d), U_res{k} (top-k PCs of d after removing per-correct-letter means), w_dec (class-mean
difference common - separate of the letter-residualised d, k=1); controls rand{k}, L_base (final site: base letter-identity subspace).
Interchange (selection test): in A, replace the candidate component at the site(s) by a partner's (opposite decision, different rotation, the
donor's chosen letter differs from the recipient's letter for the donor's meaning): meaning-flip vs letter-copy. Necessity (A minus P d) and
sufficiency (base plus P d) for the 3 best non-control candidates and the controls. Letter-span overlap for every candidate.
   python stage0.py --adapter ST2_20260919 [--n 40]"""
import argparse, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import h2lib as H, bworld as BW
CW = BW.CW
MEAN = {0: 'single', 1: 'separate', 2: 'record', 3: 'early'}
SITE_LAYERS = {'final': [16, 20, 24, 26], 'optS': [8, 12, 16, 20, 24], 'optP': [8, 12, 16, 20, 24], 'opt2': [8, 12, 16, 20, 24]}
CANDS = {'final': ['U_pca4', 'w_dec', 'U_res4', 'U_res16', 'rand4', 'L_base', 'full'], 'opt': ['U_pca4', 'w_dec', 'U_res2', 'U_res4', 'rand1', 'rand4', 'full']}   # 'full' = whole-residual swap at the site (upper bound, reference only)
OPT_TEXT = {'optS': BW.INTERPRET_OPTS[0], 'optP': BW.INTERPRET_OPTS[1]}


def build(seed, n):
    out = []
    for dom, off in (('plant', 3), ('clinic', 4), ('software', 5)):
        sits = CW.test_situations(dom, n, seed + off); rng = random.Random(seed + 50 + off)
        for si, s in enumerate(sits):
            base = list(range(4)); rng.shuffle(base)
            for r in range(4):
                order = base[r:] + base[:r]; opts = [BW.INTERPRET_OPTS[k] for k in order]
                q = BW.INTERPRET_Q + '\n' + '\n'.join('%s. %s' % ('ABCD'[j], o) for j, o in enumerate(opts)) + '\n\nAnswer with the letter only.'
                out.append({'dom': dom, 'sid': '%s%d' % (dom[0], si), 'rot': r, 'common': s['common'], 'order': order, 'prompt': 'Current situation. ' + s['text'] + '\n\n' + q,
                            'letter_of': {MEAN[k]: 'ABCD'[order.index(k)] for k in range(4)}, 'correct': 'ABCD'[order.index(0 if s['common'] else 1)]})
    return out


def option_positions(tok, items):
    """Token index (in the unpadded chat sequence) of the last token of each decision option's text."""
    P = {'optS': [], 'optP': []}
    for it in items:
        text = tok.apply_chat_template([{'role': 'system', 'content': H.SYSTEM}, {'role': 'user', 'content': it['prompt']}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True); offs = enc['offset_mapping']
        for site, otext in OPT_TEXT.items():
            end = text.index(otext) + len(otext) - 1                       # character index of the option text's last character
            hit = [i for i, (a, b) in enumerate(offs) if a <= end < b]
            P[site].append(hit[0] if hit else max(i for i, (a, b) in enumerate(offs) if a <= end))
        assert len(enc['input_ids']) == len(H.LM.chat_ids(tok, [{'role': 'system', 'content': H.SYSTEM}, {'role': 'user', 'content': it['prompt']}]))
    return P


class PosHook:
    """Adds vectors at given (row, position) pairs of block `layer`'s output; positions are in unpadded coordinates, shifted by the row's pad."""
    def __init__(self, model, layer):
        self.plan = None; self.h = model.model.layers[layer].register_forward_hook(self._fn)
    def _fn(self, module, inputs, output):
        o = output[0] if isinstance(output, tuple) else output
        if self.plan is None or o.shape[1] == 1: return output
        o = o.clone()
        for r, p, v in self.plan: o[r, p] = o[r, p] + v.to(o.dtype).to(o.device)
        return (o,) + tuple(output[1:]) if isinstance(output, tuple) else o
    def remove(self): self.h.remove()


def pad_offsets(enc, idx): L = max(len(enc.seqs[i]) for i in idx); return [L - len(enc.seqs[i]) for i in idx]


@torch.no_grad()
def forward(model, enc, n, lets, sites_pos, layers, batch=16, hook=None, plan_of=None):
    """Letter log-probs for all prompts; residuals at each site (final + option positions) for the given layers."""
    LP = torch.zeros(n, 4); R = {}
    for idx in H.batches_of(n, batch):
        ids, mask = enc.batch(idx); offs = pad_offsets(enc, idx)
        if hook is not None: hook.plan = plan_of(idx, offs)
        o = model(input_ids=ids, attention_mask=mask, output_hidden_states=bool(layers), logits_to_keep=1)
        if hook is not None: hook.plan = None
        LP[idx] = torch.log_softmax(o.logits[:, -1].float(), -1)[:, lets].cpu()
        for l in layers:
            hs = o.hidden_states[l + 1].float()
            for site in ('final', 'optS', 'optP'):
                key = (site, l)
                if key not in R: R[key] = torch.zeros(n, hs.shape[-1])
                for r, i in enumerate(idx): R[key][i] = hs[r, -1 if site == 'final' else sites_pos[site][i] + offs[r]].cpu()
    return LP, R


def meaning_acc(items, LP, idx):
    ok = {True: [], False: []}
    for i in idx: ok[items[i]['common']].append('ABCD'[int(LP[i].argmax())] == items[i]['correct'])
    return 0.5 * (sum(ok[True]) / len(ok[True]) + sum(ok[False]) / len(ok[False]))


def orth(M): return torch.linalg.qr(M)[0]


def fit_candidates(D, items, fit, names, seed, Hb=None, LPb=None):
    X = D[fit]; C = {}
    oh = torch.zeros(len(fit), 4)
    for r, i in enumerate(fit): oh[r, 'ABCD'.index(items[i]['correct'])] = 1.0
    Rz = X - oh @ torch.linalg.lstsq(oh, X).solution; Vr = torch.linalg.svd(Rz - Rz.mean(0), full_matrices=False)[2]
    com = torch.tensor([items[i]['common'] for i in fit])
    g = torch.Generator().manual_seed(seed)
    for nm in names:
        if nm == 'U_pca4': C[nm] = torch.linalg.svd(X - X.mean(0), full_matrices=False)[2][:4].T
        elif nm == 'w_dec': w = Rz[com].mean(0) - Rz[~com].mean(0); C[nm] = (w / w.norm())[:, None]
        elif nm.startswith('U_res'): C[nm] = Vr[:int(nm[5:])].T
        elif nm.startswith('rand'): C[nm] = orth(torch.randn(D.shape[1], int(nm[4:]), generator=g))
        elif nm == 'full': C[nm] = torch.eye(D.shape[1])
        elif nm == 'L_base':
            ch = LPb[fit].argmax(1); Hc = Hb[fit] - Hb[fit].mean(0); means = torch.stack([Hc[ch == c].mean(0) if (ch == c).any() else torch.zeros(Hc.shape[1]) for c in range(4)]); C[nm] = orth(means.T)[:, :3]
    return C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--adapter', required=True); ap.add_argument('--seed', type=int, default=777000); ap.add_argument('--n', type=int, default=40); ap.add_argument('--tag', default='stage0')
    ap.add_argument('--light', action='store_true', help='DEVIATIONS D4: full-residual swap at every site plus final-site U_pca4; the full candidate set is run only where a full swap flips >= 0.2'); a = ap.parse_args()
    if a.light: CANDS['final'] = ['U_pca4', 'full']; CANDS['opt'] = ['full']
    out = os.path.join(HERE, 'results', a.tag); os.makedirs(out, exist_ok=True); T0 = time.time(); logf = open(os.path.join(out, a.adapter + '.log'), 'a')
    def log(m): s = '[%6.0fs] %s' % (time.time() - T0, m); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    items = build(a.seed, a.n); n = len(items); fit = [i for i, it in enumerate(items) if it['dom'] != 'software']; test = [i for i, it in enumerate(items) if it['dom'] == 'software']
    tok, model = H.load(); lets = H.letters(tok); enc = H.Enc(tok, [it['prompt'] for it in items]); pos = option_positions(tok, items)
    layers = sorted(set(sum(SITE_LAYERS.values(), [])))
    LPb, Rb = forward(model, enc, n, lets, pos, layers)
    snap = {k: v.detach().clone().cpu() for k, v in model.state_dict().items() if 'q_proj' in k or 'v_proj' in k}
    H.merge_adapter(model, os.path.join(H.CED_RES, a.adapter, 'lora_none.pt'))
    LPa, Ra = forward(model, enc, n, lets, pos, layers)
    acc = {'base_fit': meaning_acc(items, LPb, fit), 'base_test': meaning_acc(items, LPb, test), 'A_fit': meaning_acc(items, LPa, fit), 'A_test': meaning_acc(items, LPa, test)}
    res = {'adapter': a.adapter, 'n_items': n, 'acc': acc, 'sites': {}}
    log('meaning accuracy base fit %.3f test %.3f | A fit %.3f test %.3f' % (acc['base_fit'], acc['base_test'], acc['A_fit'], acc['A_test']))
    rng = random.Random(a.seed + 9)
    def pairs(pool):
        P = []
        for i in pool:
            cand = [j for j in pool if items[j]['dom'] == items[i]['dom'] and items[j]['common'] != items[i]['common'] and items[j]['rot'] != items[i]['rot']]; rng.shuffle(cand)
            for j in cand:
                dmean = 'single' if items[j]['common'] else 'separate'
                if 'ABCD'[int(LPa[j].argmax())] != items[i]['letter_of'][dmean]: P.append((i, j)); break
        return P
    Pfit, Ptest = pairs(fit), pairs(test); Pall = Pfit + Ptest
    W = model.lm_head.weight[lets].float().cpu() * model.model.norm.weight.float().cpu(); Qlet = orth((W - W.mean(0)).T)[:, :3]
    BASES = {}

    def site_list(site): return ['optS', 'optP'] if site == 'opt2' else [site]

    def make_plan(L, site, vecfun):
        """vecfun(prompt_index, subsite) -> vector or None. Returns a plan_of(idx, offs) for PosHook."""
        def plan_of(idx, offs):
            plan = []
            for r, i in enumerate(idx):
                for ss in site_list(site):
                    v = vecfun(i, ss)
                    if v is not None: plan.append((r, -1 if ss == 'final' else pos[ss][i] + offs[r], v))
            return plan
        return plan_of

    def interchange(L, site, Us):
        donor = dict(Pall); hk = PosHook(model, L)
        vf = lambda i, ss: ((Ra[(ss, L)][donor[i]] - Ra[(ss, L)][i]) @ (Us[ss] @ Us[ss].T)) if i in donor else None
        LPx, _ = forward(model, enc, n, lets, pos, [], hook=hk, plan_of=make_plan(L, site, vf)); hk.remove(); r = {}
        for split, Pp in (('fit', Pfit), ('test', Ptest)):
            mf = lc = mf0 = lc0 = 0
            for i, j in Pp:
                ch = 'ABCD'[int(LPx[i].argmax())]; ch0 = 'ABCD'[int(LPa[i].argmax())]; dmean = 'single' if items[j]['common'] else 'separate'; dl = 'ABCD'[int(LPa[j].argmax())]
                mf += ch == items[i]['letter_of'][dmean]; lc += ch == dl; mf0 += ch0 == items[i]['letter_of'][dmean]; lc0 += ch0 == dl
            # DEVIATIONS D3: flip and copy are reported as excess over the unpatched rates on the same pairs (a null patch has copy ~0.22 by chance)
            r['flip_' + split] = (mf - mf0) / len(Pp); r['copy_' + split] = (lc - lc0) / len(Pp); r['raw_flip_' + split] = mf / len(Pp); r['raw_copy_' + split] = lc / len(Pp)
        return r

    for site, Ls in SITE_LAYERS.items():
        names = CANDS['final'] if site == 'final' else CANDS['opt']; res['sites'][site] = {}
        for L in Ls:
            per_sub = {ss: fit_candidates(Ra[(ss, L)] - Rb[(ss, L)], items, fit, names, a.seed + L, Rb[(ss, L)], LPb) for ss in site_list(site)}
            lay = {}
            for nm in names:
                Us = {ss: per_sub[ss][nm] for ss in site_list(site)}; BASES[(site, L, nm)] = Us
                r = interchange(L, site, Us); r['k'] = Us[site_list(site)[0]].shape[1]
                r['letter_span_overlap'] = {ss: float((Qlet.T @ U).pow(2).sum() / min(3, U.shape[1])) for ss, U in Us.items()}
                lay[nm] = r
            res['sites'][site][str(L)] = lay
            log('%s L%d: %s' % (site, L, {k: (round(v['flip_fit'], 2), round(v['copy_fit'], 2)) for k, v in lay.items()}))
    if a.light:                                                                # escalation: the full candidate set wherever a whole-residual swap moved meaning
        heavy = {'final': ['U_pca4', 'w_dec', 'U_res4', 'U_res16', 'rand4', 'L_base'], 'opt': ['U_pca4', 'w_dec', 'U_res2', 'U_res4', 'rand1', 'rand4']}
        for site, d in list(res['sites'].items()):
            for L, lay in list(d.items()):
                if lay.get('full', {}).get('flip_fit', 0) >= 0.2:
                    names = heavy['final'] if site == 'final' else heavy['opt']; per_sub = {ss: fit_candidates(Ra[(ss, int(L))] - Rb[(ss, int(L))], items, fit, names, a.seed + int(L), Rb[(ss, int(L))], LPb) for ss in site_list(site)}
                    for nm in names:
                        Us = {ss: per_sub[ss][nm] for ss in site_list(site)}; BASES[(site, int(L), nm)] = Us; r = interchange(int(L), site, Us); r['k'] = Us[site_list(site)[0]].shape[1]
                        r['letter_span_overlap'] = {ss: float((Qlet.T @ U).pow(2).sum() / min(3, U.shape[1])) for ss, U in Us.items()}; lay[nm] = r
                    log('ESCALATED %s L%s: %s' % (site, L, {k: (round(v['flip_fit'], 2), round(v['copy_fit'], 2)) for k, v in lay.items()}))
    # rank non-control candidates by fit-domain flip (copy <= 0.2); necessity / sufficiency for the top 3 and the controls at their sites
    ranked = sorted([(site, L, nm, r) for site, d in res['sites'].items() for L, lay in d.items() for nm, r in lay.items() if not nm.startswith('rand') and nm not in ('L_base', 'full')], key=lambda x: -x[3]['flip_fit'])
    top = [x for x in ranked if x[3]['copy_fit'] <= 0.2][:3]; probe = top + [x for x in ranked if x not in top][:1]
    ctrl = list({(s, L, nm): (s, L, nm, res['sites'][s][L][nm]) for (s, L, _n, _) in probe for nm in (['rand4'] + (['L_base'] if s == 'final' else [])) if nm in res['sites'][s][L]}.values())
    def nec_suf(model_, L, site, Us, sign):
        hk = PosHook(model_, int(L)); vf = lambda i, ss: sign * ((Ra[(ss, int(L))][i] - Rb[(ss, int(L))][i]) @ (Us[ss] @ Us[ss].T))
        LPx, _ = forward(model_, enc, n, lets, pos, [], hook=hk, plan_of=make_plan(int(L), site, vf)); hk.remove(); return LPx
    gain_t = acc['A_test'] - acc['base_test']; gain_f = acc['A_fit'] - acc['base_fit']
    for (site, L, nm, r) in probe + ctrl:
        LPx = nec_suf(model, L, site, BASES[(site, int(L), nm)], -1.0)
        r['necessity_test'] = (acc['A_test'] - meaning_acc(items, LPx, test)) / gain_t if abs(gain_t) > 1e-6 else None
        r['necessity_fit'] = (acc['A_fit'] - meaning_acc(items, LPx, fit)) / gain_f if abs(gain_f) > 1e-6 else None
    with torch.no_grad():
        sd = model.state_dict()
        for k, v in snap.items(): sd[k].copy_(v.to(sd[k].device))
    for (site, L, nm, r) in probe + ctrl:
        LPx = nec_suf(model, L, site, BASES[(site, int(L), nm)], +1.0)
        r['sufficiency_test'] = (meaning_acc(items, LPx, test) - acc['base_test']) / gain_t if abs(gain_t) > 1e-6 else None
    torch.save({str(k): v for k, v in BASES.items() if (k[0], str(k[1]), k[2]) in {(s, L, nm) for s, L, nm, _ in probe}}, os.path.join(out, a.adapter + '_bases.pt'))
    if top:
        site, L, nm, r = top[0]; gate = r['flip_test'] >= 0.5 and r['copy_test'] <= 0.2 and (r.get('necessity_test') or 0) >= 0.5
        res['selected'] = {'site': site, 'layer': int(L), 'candidate': nm, 'gate_pass': bool(gate), **r}
    else: res['selected'] = {'gate_pass': False, 'reason': 'no candidate with copy_fit <= 0.2'}
    res['top3'] = [{'site': s, 'layer': int(L), 'candidate': nm, **r} for s, L, nm, r in top]
    res['seconds'] = time.time() - T0; H.save_json(res, os.path.join(out, a.adapter + '.json')); log('selected %s' % json.dumps(res['selected'], default=str))


if __name__ == '__main__':
    main()
