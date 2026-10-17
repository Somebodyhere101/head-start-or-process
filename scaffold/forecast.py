"""SCAFFOLD forecast (PREREGISTRATION.md section 6; ASSESSMENT 5.3b items 1-2): the fresh B adapter's empirical tangent kernel on B's training
prompts, measured at t = 0 in each arm's own frame, before any B update. Output per item: f_i = log q_i(correct letter), q = softmax over the four
letters of (lp_model + offset). G = per-item gradients of f_i w.r.t. phi's B matrices (phi's A matrices have zero gradient at init).
Statistics (all frame-consistent; base and output-matched arms share residuals by construction, so differences come from the kernel):
  F_res  = d^T K d / (||d||^2 * mean diag K)   with d_i = 1 - q_i(correct)     (residual form, Deshpande et al. 2021; normalised)
  F_sign = d^T K_s d / (||d||^2 * mean diag K_s), K_s = G sign(G)^T             (sign-GD / Adam-like kernel, Malladi et al. 2023)
  CKA_type = centred alignment of K with the item-type indicator kernel (pos / near-miss types)  (Cortes et al. 2012)
  raw_scale = mean diag K (kept separately: scale must not be read as structure; Shan & Bordelon 2021)
   python forecast.py --rule bind --bseed 101 --arms base,A_matched --adapter ST2_20260919"""
import argparse, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import h2lib as H, bworld as BW, h2run as HR


def cka(K, L):
    n = K.shape[0]; Hc = torch.eye(n) - torch.full((n, n), 1.0 / n); Kc = Hc @ K @ Hc; Lc = Hc @ L @ Hc
    return float((Kc * Lc).sum() / (Kc.norm() * Lc.norm() + 1e-12))


def grads(model, enc, items, lets, off):
    Bs = H.phi_B_params(model); rows = []; dres = []
    for p in Bs: p.requires_grad_(True)
    for i in range(len(items)):
        for p in Bs: p.grad = None
        lp = H.letter_lp(model, enc, [i], lets)
        if off is not None: lp = lp + off[[i]].cuda()
        q = torch.log_softmax(lp, 1); c = 'ABCD'.index(items[i]['answer']); q[0, c].backward()
        rows.append(torch.cat([p.grad.flatten().float() for p in Bs]).cpu()); dres.append(1.0 - float(q[0, c].exp()))
    for p in Bs: p.requires_grad_(False); p.grad = None
    return torch.stack(rows), torch.tensor(dres)


def stats(G, d, types):
    K = G @ G.T; Ks = G @ torch.sign(G).T; md, mds = float(K.diag().mean()), float(Ks.diag().mean())
    T = torch.tensor([[1.0 if a == b else 0.0 for b in types] for a in types])
    return {'F_res': float(d @ K @ d) / (float(d @ d) * md + 1e-12), 'F_sign': float(d @ Ks @ d) / (float(d @ d) * mds + 1e-12), 'CKA_type': cka(K, T), 'raw_scale': md, 'mean_residual': float(d.mean())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--rule', required=True); ap.add_argument('--bseed', type=int, required=True); ap.add_argument('--arms', default='base,A_matched'); ap.add_argument('--adapter', default='ST2_20260919')
    ap.add_argument('--n_train', type=int, default=192); ap.add_argument('--tag', default='forecast'); a = ap.parse_args()
    out = os.path.join(HERE, 'results', a.tag); os.makedirs(out, exist_ok=True); T0 = time.time()
    train = BW.item_set(a.rule, a.n_train, 1000 * a.bseed + 1); random.Random(1000 * a.bseed + 2).shuffle(train); types = [it['type'] for it in train]
    tok, model = H.load(); lets = H.letters(tok); enc = H.Enc(tok, [BW.prompt(it) for it in train]); bT = H.batches_of(len(train), H.V1['batch'])
    lp0 = H.lp_table(model, enc, bT, lets); del model; torch.cuda.empty_cache(); model = H.load()[1]; H.merge_adapter(model, HR.adapter_path(a.adapter)); lpA = H.lp_table(model, enc, bT, lets)
    res = {'rule': a.rule, 'bseed': a.bseed, 'adapter': a.adapter, 'arms': {}}; phi_seed = 1000 * a.bseed + 5
    for arm in a.arms.split(','):
        del model; torch.cuda.empty_cache(); model = H.load()[1]
        if arm in ('A_raw', 'A_matched'): H.merge_adapter(model, HR.adapter_path(a.adapter))
        off = {'base': None, 'A_raw': None, 'prior': lpA - lp0, 'A_matched': -(lpA - lp0)}[arm]
        H.add_phi(model, phi_seed); G, d = grads(model, enc, train, lets, off); res['arms'][arm] = stats(G, d, types)
        print('%s %s b%d %s: %s' % (a.adapter, a.rule, a.bseed, arm, {k: round(v, 4) for k, v in res['arms'][arm].items()}), flush=True)
    res['seconds'] = time.time() - T0; H.save_json(res, os.path.join(out, '%s_b%d_%s.json' % (a.rule, a.bseed, os.path.splitext(os.path.basename(a.adapter))[0])))


if __name__ == '__main__':
    main()
