"""Stage C1 (PROTOCOL.md): consolidate significance, not content. LoRA on query/key projections whose input side is restricted to the recall
subspace (R basis) of each block's input residual; arms base, constrained, constrained_shuffled, qk_unconstrained, random_subspace, standard.
Behavioral evaluation on the five-question items; instrument evaluation (R/P/J-proxy reorganization) for constrained and its shuffled control.
  python exp_consolidate.py [--smoke]"""
import argparse, hashlib, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'retrogenesis'))
import torch, torch.nn as nn, torch.nn.functional as Fnn
import local_models as LM, world as WD, questions as Q, rlens as RL, exp_abstraction as EA, exp_rj as RJ

CFG = {'seed': 20260914, 'n_train': 192, 'n_items': 96, 'epochs': 3, 'lr': 2e-4, 'micro': 4, 'rank': 8, 'alpha': 16, 'k': 64, 'n_inst_prompts': 60, 'n_heldout': 32, 'cap_minutes': 45.0}
R_BASES = 'results/R1/bases.pt'; STD_LORA = 'results/E1/lora_episodes_only.pt'


class ProjLoRA(nn.Module):
    """y = W x + (alpha/r) B A P^T x with P (d x k) fixed; P = None gives an unconstrained LoRA on the same projection."""
    def __init__(self, base, r, alpha, P, seed):
        super().__init__(); self.base, self.scale = base, alpha / r; g = torch.Generator().manual_seed(seed); kin = base.in_features if P is None else P.shape[1]
        self.register_buffer('P', None if P is None else P.detach().float().clone()); self.A = nn.Parameter((torch.rand(r, kin, generator=g) * 2 - 1) / kin ** 0.5); self.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(self, x):
        z = x.float() if self.P is None else x.float() @ self.P; return self.base(x) + (Fnn.linear(Fnn.linear(z, self.A), self.B) * self.scale).to(x.dtype)


def install(model, bases, mode, r, alpha, seed, k):
    """mode: 'constrained' (P = R basis of the block input), 'random' (random orthonormal P), 'qk' (unconstrained). Blocks 1..n-1 (block 0 input is the embedding)."""
    mods = []; d = model.config.hidden_size
    for li in range(1, len(model.model.layers)):
        if mode == 'constrained': P = bases['R_basis'][li - 1].float()[:, :k]
        elif mode == 'random': Pr = torch.randn(d, k, generator=torch.Generator().manual_seed(seed + 1000 + li)); P, _ = torch.linalg.qr(Pr)
        else: P = None
        att = model.model.layers[li].self_attn
        for name in ('q_proj', 'k_proj'):
            m = ProjLoRA(getattr(att, name), r, alpha, P, seed + 7 * li + (0 if name == 'q_proj' else 3)).to(model.device); setattr(att, name, m); mods.append((li, name, m))
    return mods


def uninstall(model, mods):
    for li, name, m in mods: setattr(model.model.layers[li].self_attn, name, m.base)


def train_texts(S, tok, model, params, texts, seed, tag):
    cfg = S.cfg
    for p in params: p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg['lr']); rng = random.Random(seed); ids = [tok(t, add_special_tokens=False)['input_ids'] + [LM.EOS_IDS[1]] for t in texts]; trace = []
    for ep in range(cfg['epochs']):
        order = list(range(len(ids))); rng.shuffle(order); tot = 0.0; n = 0
        for i in range(0, len(order), cfg['micro']):
            batch = [ids[j] for j in order[i:i + cfg['micro']]]; L = max(len(b) for b in batch)
            x = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long); y = torch.full((len(batch), L), -100, dtype=torch.long); m = torch.zeros((len(batch), L), dtype=torch.long)
            for kk, b in enumerate(batch): x[kk, :len(b)] = torch.tensor(b); y[kk, :len(b)] = torch.tensor(b); m[kk, :len(b)] = 1
            x, y, m = x.cuda(), y.cuda(), m.cuda(); hidden = model.model(input_ids=x, attention_mask=m).last_hidden_state[:, :-1]; tgt = y[:, 1:]; sel = tgt != -100
            loss = Fnn.cross_entropy(model.lm_head(hidden[sel]).float(), tgt[sel]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True); opt.step(); tot += float(loss); n += 1
        trace.append(tot / n); S.log('%s epoch %d: loss %.4f' % (tag, ep, tot / n))
    for p in params: p.requires_grad_(False)
    return trace


@torch.no_grad()
def heldout_kl(tok, model, held, base_lps):
    kls = []
    for text, blp in zip(held, base_lps):
        ids = tok(text, add_special_tokens=False)['input_ids'][:RL.T]; lp = torch.log_softmax(model(input_ids=torch.tensor([ids], device='cuda')).logits[0].float(), -1).cpu(); kls.append(float((torch.exp(blp) * (blp - lp)).sum(-1).mean()))
    return sum(kls) / len(kls)


def behavioral(tok, model, items):
    res = {}
    for dom, its in items.items():
        res[dom] = {}
        for q in EA.ALLQ:
            rows = EA.score(tok, model, its, q); res[dom][q] = {'rate': sum(r['correct'] for r in rows) / len(rows), 'rows': rows}
    far3 = [res[d][q]['rate'] for d in ('clinic', 'software') for q in EA.REL3]; res['far3_mean'] = sum(far3) / len(far3); res['plant3_mean'] = sum(res['plant'][q]['rate'] for q in EA.REL3) / 3; res['surface_min'] = min(res[d]['surface']['rate'] for d in items)
    res['recall_far_mean'] = (res['clinic']['recall']['rate'] + res['software']['recall']['rate']) / 2; return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--only', default=''); ap.add_argument('--tag', default='C1'); ap.add_argument('--from_results', default=''); ap.add_argument('--n_inst_prompts', type=int, default=None); ap.add_argument('--cap_minutes', type=float, default=None); ap.add_argument('--inst_from_lora', default='', help='comma list of constrained arms whose saved lora_<tag>.pt is reloaded for the instrument fit'); a = ap.parse_args(); cfg = dict(CFG)
    if a.n_inst_prompts: cfg['n_inst_prompts'] = a.n_inst_prompts
    if a.cap_minutes: cfg['cap_minutes'] = a.cap_minutes
    if a.smoke: cfg.update(n_train=8, n_items=8, epochs=1, n_inst_prompts=4, n_heldout=4)
    S = EA.Stage(cfg, os.path.join('results', a.tag + ('_smoke' if a.smoke else ''))); S.res['code']['exp_consolidate.py'] = EA.file_sha('exp_consolidate.py'); torch.manual_seed(cfg['seed']); T0 = time.time()
    train = WD.balanced_situations('plant', cfg['n_train'], cfg['seed'] + 1); items = Q.build_items(seed=cfg['seed'], n=cfg['n_items'], k_pool=64 if not a.smoke else 16); texts = [WD.episode_text(e) for e in train]
    rng = random.Random(cfg['seed'] + 99); outs = [e['outcome'] for e in train]; rng.shuffle(outs); shuf_texts = [WD.episode_text(e, o) for e, o in zip(train, outs)]
    tok, model = RL.load_eager(LM.TARGET_DIR); bases = torch.load(R_BASES); S.log('model loaded (eager); R bases for %d layers' % len(bases['R_basis']))
    prompts, _ = RL.corpus(tok, 120 + 32); inst_prompts, held = prompts[:cfg['n_inst_prompts']], prompts[120:120 + cfg['n_heldout']]; d = model.config.hidden_size
    U = torch.randn(120, RJ.N_POS, RJ.N_U, d, generator=torch.Generator().manual_seed(cfg['seed'] + 5)); U = U / U.norm(dim=-1, keepdim=True)
    base_prof, base_lps = RJ.profile_and_dist(tok, model, held)
    if a.from_results:
        prev = json.load(open(a.from_results)); S.res['arms']['base'] = prev['arms']['base']; S.res['arms']['standard'] = prev['arms'].get('standard'); S.res['reused_from'] = a.from_results
    else: S.res['arms']['base'] = dict(behavioral(tok, model, items), heldout_kl=0.0); S.log('base far3 %.3f recall_far %.3f surface_min %.2f' % (S.res['arms']['base']['far3_mean'], S.res['arms']['base']['recall_far_mean'], S.res['arms']['base']['surface_min']))
    # standard q/v LoRA reference (E1 episodes-only)
    if not a.smoke and not a.from_results:
        n_lora = LM.add_lora(model, cfg['rank'], cfg['alpha'], ('q_proj', 'v_proj'), seed=cfg['seed']); LM.load_lora_state(model, torch.load(STD_LORA)); S.res['arms']['standard'] = dict(behavioral(tok, model, items), heldout_kl=heldout_kl(tok, model, held, base_lps)); S.log('standard far3 %.3f kl %.4f' % (S.res['arms']['standard']['far3_mean'], S.res['arms']['standard']['heldout_kl']))
        for _, m in LM.lora_modules(model): pass
        # remove the standard LoRA by restoring base linears
        for name, mod in list(model.named_modules()):
            for cn, child in list(mod.named_children()):
                if isinstance(child, LM.LoRALinear): setattr(mod, cn, child.base)
    arms = [('constrained', 'constrained', texts), ('constrained_shuffled', 'constrained', shuf_texts), ('qk_unconstrained', 'qk', texts), ('random_subspace', 'random', texts)]; inst_rows = {}
    if a.only: arms = [x for x in arms if x[0] in a.only.split(',')]
    if a.from_results and os.path.exists(os.path.join(os.path.dirname(a.from_results), 'inst_rows.pt')): inst_rows.update(torch.load(os.path.join(os.path.dirname(a.from_results), 'inst_rows.pt')))
    for tag, mode, tr_texts in arms:
        if time.time() - T0 > 60 * cfg['cap_minutes']: S.res['status'] = 'incomplete: cap'; S.dump(); return
        mods = install(model, bases, mode, cfg['rank'], cfg['alpha'], cfg['seed'], cfg['k']); params = [p for _, _, m in mods for p in (m.A, m.B)]
        t = time.time(); trace = train_texts(S, tok, model, params, tr_texts, cfg['seed'] + 31, tag); S.res['timings']['train_' + tag] = time.time() - t
        torch.save({'%d.%s' % (li, n): {'A': m.A.detach().cpu(), 'B': m.B.detach().cpu()} for li, n, m in mods}, os.path.join(S.out, 'lora_%s.pt' % tag))
        S.res['arms'][tag] = dict(behavioral(tok, model, items), heldout_kl=heldout_kl(tok, model, held, base_lps), train_trace=trace, B_norm=float(sum(m.B.norm() ** 2 for _, _, m in mods) ** 0.5)); S.dump()
        S.log('%-20s far3 %.3f plant3 %.3f recall_far %.3f surface_min %.2f kl %.4f' % (tag, S.res['arms'][tag]['far3_mean'], S.res['arms'][tag]['plant3_mean'], S.res['arms'][tag]['recall_far_mean'], S.res['arms'][tag]['surface_min'], S.res['arms'][tag]['heldout_kl']))
        if tag in ('constrained', 'constrained_shuffled') and not (time.time() - T0 > 60 * cfg['cap_minutes']):
            t = time.time(); inst_rows[tag] = RJ.fit_readouts(tok, model, inst_prompts, U, S.log, tag); S.res['timings']['inst_' + tag] = time.time() - t; torch.save(inst_rows, os.path.join(S.out, 'inst_rows.pt'))
        uninstall(model, mods)
    for tag in [x for x in a.inst_from_lora.split(',') if x]:
        if tag in inst_rows: continue
        src = os.path.dirname(a.from_results) if a.from_results else S.out; st = torch.load(os.path.join(src, 'lora_%s.pt' % tag)); mods = install(model, bases, 'constrained', cfg['rank'], cfg['alpha'], cfg['seed'], cfg['k'])
        for li, n, m in mods: m.A.data.copy_(st['%d.%s' % (li, n)]['A'].to(m.A.device)); m.B.data.copy_(st['%d.%s' % (li, n)]['B'].to(m.B.device))
        S.res['arms'].setdefault(tag, {})['reloaded_lora'] = os.path.join(src, 'lora_%s.pt' % tag); S.log('%s: reloaded LoRA weights from %s for the instrument fit' % (tag, src))
        t = time.time(); inst_rows[tag] = RJ.fit_readouts(tok, model, inst_prompts, U, S.log, tag); S.res['timings']['inst_' + tag] = time.time() - t; uninstall(model, mods); torch.save(inst_rows, os.path.join(S.out, 'inst_rows.pt')); S.dump()
    # instrument analysis against the RJ1 base rows (same prompts, positions and random vectors)
    if inst_rows and not a.smoke:
        base_rows = torch.load('results/RJ1/rows.pt')['base']; nR = cfg['n_inst_prompts'] * RJ.N_POS; nJ = nR * RJ.N_U; nl = len(model.model.layers); table = {}
        def sub(M): return RL.span_of_rows(M, cfg['k'])[0]
        for l in range(nl - 1):
            Rb, Pb, Jb = sub(base_rows[0][l][:nR]), sub(base_rows[1][l][:nR]), sub(base_rows[2][l][:nJ]); e = {}
            for tag, (R, P, J) in inst_rows.items(): e['R_reorg_' + tag] = 1 - RL.overlap(Rb, sub(R[l])); e['J_reorg_' + tag] = 1 - RL.overlap(Jb, sub(J[l])); e['P_reorg_' + tag] = 1 - RL.overlap(Pb, sub(P[l]))
            table[l] = e
        upper = [l for l in table if 14 <= l <= 26]; S.res['instrument'] = {'table': table, 'R_over_J_layers': sum(table[l]['R_reorg_constrained'] > table[l]['J_reorg_constrained'] for l in upper), 'R_over_shuffled_layers': sum(table[l]['R_reorg_constrained'] > table[l]['R_reorg_constrained_shuffled'] for l in upper), 'n_upper': len(upper)}
        S.log('instrument: R>J at %d/%d upper layers; R(constrained)>R(shuffled) at %d/%d' % (S.res['instrument']['R_over_J_layers'], len(upper), S.res['instrument']['R_over_shuffled_layers'], len(upper)))
        torch.save(inst_rows, os.path.join(S.out, 'inst_rows.pt'))
    if a.from_results:
        for k_, v_ in prev['arms'].items():
            if v_ is None: continue
            if k_ not in S.res['arms'] or S.res['arms'][k_] is None: S.res['arms'][k_] = v_
            elif 'far3_mean' not in S.res['arms'][k_]: S.res['arms'][k_] = dict(v_, **S.res['arms'][k_])
    A = S.res['arms']; f = lambda t: A.get(t, {}).get('far3_mean') if A.get(t) else None; b, c = f('base'), f('constrained'); sub_ = lambda x, y: (x - y) if (x is not None and y is not None) else None
    ok = lambda *xs: all(x is not None for x in xs); I = S.res.get('instrument', {})
    S.res['criteria'] = {'constrained_gain': sub_(c, b), 'vs_shuffled': sub_(c, f('constrained_shuffled')), 'vs_qk': sub_(c, f('qk_unconstrained')), 'vs_random': sub_(c, f('random_subspace')), 'surface_min': A.get('constrained', {}).get('surface_min'), 'kl_constrained': A.get('constrained', {}).get('heldout_kl'), 'kl_standard': (A.get('standard') or {}).get('heldout_kl'),
                        'behavioral_pass': (c - b >= 0.10 and c - f('constrained_shuffled') >= 0.10 and c > f('qk_unconstrained') and c > f('random_subspace')) if ok(b, c, f('constrained_shuffled'), f('qk_unconstrained'), f('random_subspace')) else None,
                        'instrument_pass': (I.get('R_over_J_layers', 0) > 6.5 and I.get('R_over_shuffled_layers', 0) > 6.5) if I else None,
                        'content_intact': (A['constrained']['surface_min'] >= 0.95 and (A.get('standard') is None or A['constrained']['heldout_kl'] <= A['standard']['heldout_kl'])) if A.get('constrained', {}).get('surface_min') is not None else None}
    S.res['status'] = 'complete'; S.res['timings']['total'] = time.time() - T0; S.dump(); S.log('done: %s' % S.res['criteria'])


if __name__ == '__main__':
    main()
