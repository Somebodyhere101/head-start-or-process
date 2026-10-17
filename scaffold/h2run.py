"""SCAFFOLD runner (PREREGISTRATION.md sections 5-6). One process = one (B rule, B seed, A adapter) cell; runs the requested arms in order,
reloading clean base weights before each arm.
Arms: base | A_raw | prior (base + A's start offset) | A_matched (A's kernel, base start) | A_matched_rm (r_A removed during learning only) |
      A_matched_rand (random equal-k subspace removed during learning only). (A' controls = A_matched with --adapter pointing at a control.)
--check_only: precompute tables, verify output-matching exactness at t=0 (fresh zero adapter on the merged model), no training.
Per arm: balanced held-out interpret accuracy at t=0 and every --eval_every situations (own frame), then final evaluations: own frame,
scaffold (theta0 + phi, A unmerged), and for rm arms the intact-A matched frame (persistence). Writes results/<tag>/<cell>.json.
   python h2run.py --rule bind --bseed 1 --arms base,A_raw --adapter ST2_20260919 --passes 6 --tag pilot"""
import argparse, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import h2lib as H, bworld as BW, local_models as LM


def adapter_path(name): return name if os.path.exists(name) else os.path.join(H.CED_RES, name, 'lora_none.pt')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--rule', required=True, choices=['bind', 'bext', 'bpara']); ap.add_argument('--bseed', type=int, required=True)
    ap.add_argument('--arms', default='base,A_raw'); ap.add_argument('--adapter', default='ST2_20260919'); ap.add_argument('--passes', type=int, default=6)
    ap.add_argument('--n_train', type=int, default=192); ap.add_argument('--n_eval', type=int, default=96); ap.add_argument('--eval_every', type=int, default=96)
    ap.add_argument('--layer', type=int, default=24); ap.add_argument('--rbasis', default=''); ap.add_argument('--tag', default='pilot'); ap.add_argument('--check_only', action='store_true')
    ap.add_argument('--pc_factor', type=float, default=4.0, help='PC-H2 amplification factor (Stage-2 addendum: chosen on pilot seeds by pc_select.py)'); a = ap.parse_args()
    out_dir = os.path.join(HERE, 'results', a.tag); os.makedirs(out_dir, exist_ok=True); aname = os.path.basename(os.path.dirname(adapter_path(a.adapter))) if not os.path.exists(a.adapter) else os.path.splitext(os.path.basename(a.adapter))[0]
    cell = '%s_b%d_%s' % (a.rule, a.bseed, aname); T0 = time.time(); logf = open(os.path.join(out_dir, cell + '.log'), 'a')
    def log(m): s = '[%6.0fs] %s' % (time.time() - T0, m); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    # B items: seeds derived from the B seed only (items, option orders, item order, phi init); identical across arms and adapters
    train = BW.item_set(a.rule, a.n_train, 1000 * a.bseed + 1); random.Random(1000 * a.bseed + 2).shuffle(train)
    ev = BW.item_set(a.rule, a.n_eval, 1000 * a.bseed + 3); shift = BW.item_set(a.rule, a.n_eval, 1000 * a.bseed + 4, shift=True)
    phi_seed = 1000 * a.bseed + 5
    tok, model = H.load(); lets = H.letters(tok)
    encT, encE, encS = H.Enc(tok, [BW.prompt(it) for it in train]), H.Enc(tok, [BW.prompt(it) for it in ev]), H.Enc(tok, [BW.prompt(it) for it in shift])
    bT, bE, bS = H.batches_of(len(train), H.V1['batch']), H.batches_of(len(ev), 8), H.batches_of(len(shift), 8)
    # offset tables: base and A letter log-probs on every prompt, with the batch composition later used
    lp0 = {k: H.lp_table(model, e, b, lets) for k, e, b in (('T', encT, bT), ('E', encE, bE), ('S', encS, bS))}
    reload = lambda: H.load()[1]
    model = None; torch.cuda.empty_cache(); model = reload(); H.merge_adapter(model, adapter_path(a.adapter))
    lpA = {k: H.lp_table(model, e, b, lets) for k, e, b in (('T', encT, bT), ('E', encE, bE), ('S', encS, bS))}
    res = {'rule': a.rule, 'bseed': a.bseed, 'adapter': aname, 'passes': a.passes, 'n_train': a.n_train, 'n_eval': a.n_eval, 'arms': {}, 'prereg_sha': open(os.path.join(HERE, 'PREREGISTRATIONS_SHA256SUMS')).read().split()[0]}
    prev_path = os.path.join(out_dir, cell + '.json')
    if os.path.exists(prev_path) and not a.check_only:                     # several invocations may write arms of the same cell: keep earlier arms
        prev = json.load(open(prev_path))
        if prev.get('passes') == a.passes and prev.get('n_train') == a.n_train: res['arms'] = prev.get('arms', {})
    res['t0'] = {'base_eval': H.BW.balanced_acc(ev, ['ABCD'[int(k)] for k in lp0['E'].argmax(1)])[0], 'A_eval': H.BW.balanced_acc(ev, ['ABCD'[int(k)] for k in lpA['E'].argmax(1)])[0],
                 'A_minus_base_lp_absmean': float((lpA['E'] - lp0['E']).abs().mean())}
    log('cell %s: t0 balanced base %.3f, A %.3f' % (cell, res['t0']['base_eval'], res['t0']['A_eval']))
    # exactness check: fresh zero adapter on merged A leaves lp unchanged; matched lp = base lp
    H.add_phi(model, phi_seed)
    lpA_phi = H.lp_table(model, encT, bT, lets); matched = lpA_phi - (lpA['T'] - lp0['T'])
    res['exactness'] = {'phi_zero_maxdiff': float((lpA_phi - lpA['T']).abs().max()), 'matched_vs_base_maxdiff': float((matched - lp0['T']).abs().max())}
    log('exactness: zero-phi max |dlp| %.2e, matched vs base max |dlp| %.2e' % (res['exactness']['phi_zero_maxdiff'], res['exactness']['matched_vs_base_maxdiff']))
    H.strip(model)
    if a.check_only: H.save_json(res, os.path.join(out_dir, cell + '.json')); return
    U = torch.load(a.rbasis)['U'] if a.rbasis else None
    for arm in a.arms.split(','):
        t = time.time(); del model; torch.cuda.empty_cache(); model = reload(); hook = None; rm = None; off_fn = None; on_batch = None; step_hooks = None; extra = {}
        if arm in ('A_raw', 'A_matched', 'A_matched_rm', 'A_matched_rand'): H.merge_adapter(model, adapter_path(a.adapter))
        off = {'base': None, 'A_raw': None, 'prior': lpA['T'] - lp0['T'], 'A_matched': -(lpA['T'] - lp0['T']), 'A_matched_fade': -(lpA['T'] - lp0['T'])}.get(arm)
        offE = {'base': None, 'A_raw': None, 'prior': lpA['E'] - lp0['E'], 'A_matched': -(lpA['E'] - lp0['E'])}.get(arm)
        offS = {'prior': lpA['S'] - lp0['S'], 'A_matched': -(lpA['S'] - lp0['S'])}.get(arm)
        if arm in ('A_matched_rm', 'A_matched_rand'):
            raise NotImplementedError('removal arms are enabled by the Stage-2 addendum once r_A is fixed by Stage 0')
        if arm == 'A_matched_fade':
            # A as a scaled weight delta; scale 1 for the first two thirds of the budget, then linear to 0; offsets recomputed per scale so the
            # learner always starts from base's function plus phi's change (DEVIATIONS / ASSESSMENT 5.3b item 4)
            SD = H.ScaledDelta(model, adapter_path(a.adapter)); cs = [round(0.1 * k, 1) for k in range(11)]; tabT, tabE = {}, {}
            for c in cs:
                SD.set(c); tabT[c] = -(H.lp_table(model, encT, bT, lets) - lp0['T']); tabE[c] = -(H.lp_table(model, encE, bE, lets) - lp0['E'])
            SD.set(1.0); cur = {'c': 1.0}
            def on_batch(seen, total):
                c = 1.0 if seen < 2 * total / 3 else max(0.0, 1.0 - (seen - 2 * total / 3) / (total / 3)); SD.set(c); cur['c'] = min(cs, key=lambda x: abs(x - c))
            off_fn = lambda idx: tabT[cur['c']][idx]
            offE_fn = lambda: tabE[cur['c']]
        params = H.add_phi(model, phi_seed)
        if arm == 'PC_H2':
            cal = BW.item_set(a.rule, 32, 1000 * a.bseed + 6); encC = H.Enc(tok, [BW.prompt(it) for it in cal]); Ug = H.calib_basis(model, encC, cal, lets, k=8)
            step_hooks = H.make_amplifier(model, Ug, a.pc_factor); extra['pc_basis_dim'] = list(Ug.shape); extra['pc_factor'] = a.pc_factor
        cur_offE = (lambda: offE_fn()) if arm == 'A_matched_fade' else (lambda: offE)
        def on_eval(seen):
            r = H.evaluate(model, encE, ev, lets, off=cur_offE()); out = {'bal': r['bal'], 'by_type': r['by_type']}
            if arm == 'A_matched_fade': out['c'] = SD.c
            return out
        t0 = H.evaluate(model, encE, ev, lets, off=cur_offE())
        cfg = dict(H.V1, lr=H.V1['lr'] * float(arm[7:])) if arm.startswith('base_lr') else H.V1      # E1 exploratory: pure-process benchmark (same forward as base)
        trace = H.train_b(model, params, encT, train, lets, a.passes, off=off, hook=hook, rm=rm, on_eval=on_eval, eval_every=a.eval_every, log=log, step_hooks=step_hooks, off_fn=off_fn, on_batch=on_batch, cfg=cfg)
        if arm == 'A_matched_fade': SD.set(0.0); extra['final_c'] = SD.c
        own = H.evaluate(model, encE, ev, lets, off=cur_offE() if arm != 'A_matched_fade' else None); own_shift = H.evaluate(model, encS, shift, lets, off=offS if arm != 'A_matched_fade' else None)
        phi = H.phi_state(model)
        # scaffold evaluation: theta0 + phi (A unmerged)
        del model; torch.cuda.empty_cache(); model = reload(); H.add_phi(model, phi_seed); LM.load_lora_state(model, phi)
        scaf = H.evaluate(model, encE, ev, lets); scaf_shift = H.evaluate(model, encS, shift, lets); H.strip(model)
        res['arms'][arm] = {'t0': t0, 'trace': trace, 'final_own': own, 'final_own_shift': own_shift, 'scaffold': scaf, 'scaffold_shift': scaf_shift, 'G': sum(x['bal'] for x in trace) / max(1, len(trace)), 'seconds': time.time() - t, **extra}
        torch.save(phi, os.path.join(out_dir, '%s_%s_phi.pt' % (cell, arm)))
        log('%-12s t0 %.3f -> final %.3f | G %.3f | scaffold %.3f | shift own %.3f | curve %s | %.0f s' % (arm, t0['bal'], own['bal'], res['arms'][arm]['G'], scaf['bal'], own_shift['bal'], [round(x['bal'], 2) for x in trace], time.time() - t))
        H.save_json(res, os.path.join(out_dir, cell + '.json'))
    res['seconds'] = time.time() - T0; H.save_json(res, os.path.join(out_dir, cell + '.json')); log('done in %.0f s' % res['seconds'])


if __name__ == '__main__':
    main()
