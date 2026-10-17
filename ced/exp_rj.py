"""Stage RJ1 (PROTOCOL.md): does experience reorganize the recall channel (R) more than the verbalization channel (J)?
States: base, experience (E1 episodes-only LoRA), control (shuffled-outcome LoRA trained here). Readouts R, P and a same-corpus J-proxy are
refitted per state on the same generic prompts; reorganization = 1 - overlap(base, state); steering along the experience change in R vs J.
  python exp_rj.py [--smoke]"""
import argparse, hashlib, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'retrogenesis'))
import torch, torch.nn.functional as Fnn
import local_models as LM, world as WD, rlens as RL

K = 64; T = 128; SKIP = 16; N_POS = 8; N_U = 2; STEER_LAYERS = [12, 18, 22]; EXP_LORA = 'results/E1/lora_episodes_only.pt'
CFG = {'seed': 20260914, 'n_prompts': 120, 'n_heldout': 32, 'n_train': 192, 'epochs': 3, 'lr': 2e-4, 'micro': 4, 'lora_rank': 8, 'lora_alpha': 16, 'lora_targets': ['q_proj', 'v_proj'], 'cap_minutes': 60.0, 'steer_frac': 0.1}


def file_sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def train_shuffled(S, tok, model, train, seed):
    """Stage-1 recipe on outcome-shuffled episodes (whole-text loss, Adam 2e-4, 3 epochs, micro 4); returns the LoRA state and agreement."""
    cfg = S.cfg; rng = random.Random(seed); outs = [e['outcome'] for e in train]; rng.shuffle(outs); texts = [WD.episode_text(e, o) for e, o in zip(train, outs)]; agree = sum(o == e['outcome'] for o, e in zip(outs, train)) / len(train)
    params = [p for _, m in LM.lora_modules(model) for p in (m.A, m.B)]
    for p in params: p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg['lr']); ids = [tok(t, add_special_tokens=False)['input_ids'] + [LM.EOS_IDS[1]] for t in texts]; trace = []
    for ep in range(cfg['epochs']):
        order = list(range(len(ids))); rng.shuffle(order); tot = 0.0; n = 0
        for i in range(0, len(order), cfg['micro']):
            batch = [ids[j] for j in order[i:i + cfg['micro']]]; L = max(len(b) for b in batch)
            x = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long); y = torch.full((len(batch), L), -100, dtype=torch.long); m = torch.zeros((len(batch), L), dtype=torch.long)
            for k, b in enumerate(batch): x[k, :len(b)] = torch.tensor(b); y[k, :len(b)] = torch.tensor(b); m[k, :len(b)] = 1
            x, y, m = x.cuda(), y.cuda(), m.cuda(); hidden = model.model(input_ids=x, attention_mask=m).last_hidden_state[:, :-1]; tgt = y[:, 1:]; sel = tgt != -100
            loss = Fnn.cross_entropy(model.lm_head(hidden[sel]).float(), tgt[sel]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True); opt.step(); tot += float(loss); n += 1
        trace.append(tot / n); S.log('control epoch %d: loss %.4f' % (ep, tot / n))
    for p in params: p.requires_grad_(False)
    return LM.lora_state(model), trace, agree


def fit_readouts(tok, model, prompts, U, log, tag):
    """Per source layer: R rows, P rows and J-proxy rows (2 random projections of the final residual per position). U: fixed unit vectors."""
    nl = len(model.model.layers); R = {l: [] for l in range(nl - 1)}; P = {l: [] for l in range(nl - 1)}; J = {l: [] for l in range(nl - 1)}
    for i, text in enumerate(prompts):
        ids = tok(text, add_special_tokens=False)['input_ids'][:T]; x = torch.tensor([ids], device='cuda'); emb = model.get_input_embeddings()(x).detach().requires_grad_(True)
        out = model(inputs_embeds=emb, output_attentions=True, output_hidden_states=True); hs = [out.hidden_states[l + 1] for l in range(nl - 1)]; hf = out.hidden_states[-1][0]
        n = len(ids); mask = torch.triu(torch.ones(n, n, device='cuda'), diagonal=1); S = sum((A[0].float() * mask.T.unsqueeze(0)).sum(dim=(0, 1)) for A in out.attentions)
        lp = torch.log_softmax(out.logits[0].float(), -1); nxt = lp[torch.arange(n - 1), x[0, 1:]]; positions = torch.linspace(SKIP, n - 2, N_POS).round().long().tolist()
        for j, p in enumerate(positions):
            targets = [S[p], nxt[p]] + [(hf[p].float() * U[i % U.shape[0], j, u_].cuda()).sum() for u_ in range(N_U)]
            for t_i, tgt in enumerate(targets):
                g = torch.autograd.grad(tgt, hs, retain_graph=not (j == len(positions) - 1 and t_i == len(targets) - 1))
                for l in range(nl - 1):
                    row = g[l][0, p].detach().half().cpu(); (R if t_i == 0 else P if t_i == 1 else J)[l].append(row)
        if i % 20 == 0: log('  %s readouts prompt %d/%d' % (tag, i, len(prompts)))
    return {l: torch.stack(v) for l, v in R.items()}, {l: torch.stack(v) for l, v in P.items()}, {l: torch.stack(v) for l, v in J.items()}


@torch.no_grad()
def profile_and_dist(tok, model, prompts, layer=None, vec=None):
    """Normalized attention-mass profile S/sum and next-token log-probs on prompts, optionally with vec added at all positions of `layer`."""
    hook = None
    if vec is not None:
        def h(module, inputs, output):
            o = (output[0] if isinstance(output, tuple) else output).clone(); o += vec.to(o.dtype); return (o,) + tuple(output[1:]) if isinstance(output, tuple) else o
        hook = model.model.layers[layer].register_forward_hook(h)
    profs, lps = [], []
    try:
        for text in prompts:
            ids = tok(text, add_special_tokens=False)['input_ids'][:T]; x = torch.tensor([ids], device='cuda'); out = model(input_ids=x, output_attentions=True); n = len(ids)
            mask = torch.triu(torch.ones(n, n, device='cuda'), diagonal=1); S = sum((A[0].float() * mask.T.unsqueeze(0)).sum(dim=(0, 1)) for A in out.attentions); profs.append((S / S.sum()).cpu()); lps.append(torch.log_softmax(out.logits[0].float(), -1).cpu())
    finally:
        if hook: hook.remove()
    return profs, lps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args(); cfg = dict(CFG)
    if a.smoke: cfg.update(n_prompts=6, n_heldout=4, n_train=8, epochs=1)
    out = os.path.join('results', 'RJ1_smoke' if a.smoke else 'RJ1'); os.makedirs(out, exist_ok=True); T0 = time.time(); logf = open(os.path.join(out, 'log.txt'), 'a')
    def log(m): s = '[%6.0fs] %s' % (time.time() - T0, m); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    S_ = type('S', (), {})(); S_.cfg = cfg; S_.log = log
    res = {'config': cfg, 'code': {f: file_sha(f) for f in ('exp_rj.py', 'rlens.py', 'world.py', 'PROTOCOL.md')}, 'exp_lora': EXP_LORA, 'status': 'running', 'timings': {}}
    torch.manual_seed(cfg['seed']); tok, model = RL.load_eager(LM.TARGET_DIR); n_lora = LM.add_lora(model, cfg['lora_rank'], cfg['lora_alpha'], tuple(cfg['lora_targets']), seed=cfg['seed']); init = LM.lora_state(model)
    prompts, cstat = RL.corpus(tok, cfg['n_prompts'] + cfg['n_heldout']); fit_prompts, held = prompts[:cfg['n_prompts']], prompts[cfg['n_prompts']:]; res['corpus'] = dict(cstat, fit=len(fit_prompts), heldout=len(held)); log('model loaded (eager, %d LoRA modules); corpus %s' % (n_lora, res['corpus']))
    d = model.config.hidden_size; U = torch.randn(len(fit_prompts), N_POS, N_U, d, generator=torch.Generator().manual_seed(cfg['seed'] + 5)); U = U / U.norm(dim=-1, keepdim=True)
    # control LoRA
    train = WD.balanced_situations('plant', cfg['n_train'], cfg['seed'] + 1); LM.load_lora_state(model, init); t = time.time(); ctrl_state, ctrace, agree = train_shuffled(S_, tok, model, train, cfg['seed'] + 99); torch.save(ctrl_state, os.path.join(out, 'lora_control.pt'))
    res['control'] = {'train_trace': ctrace, 'shuffled_outcome_agreement': agree, 'seconds': time.time() - t}; log('control LoRA trained (%.0f s)' % (time.time() - t))
    states = {'base': init, 'experience': torch.load(EXP_LORA) if not a.smoke else init, 'control': ctrl_state}; rows = {}
    for name, st in states.items():
        LM.load_lora_state(model, st); t = time.time(); rows[name] = fit_readouts(tok, model, fit_prompts, U, log, name); res['timings']['fit_' + name] = time.time() - t; log('%s readouts done (%.0f s)' % (name, time.time() - t))
    torch.save({k: v for k, v in rows.items()}, os.path.join(out, 'rows.pt'))
    # analysis
    Jw = torch.load(RL.JLENS, map_location='cpu', weights_only=False)['J']; nl = len(model.model.layers); nrows = rows['base'][0][0].shape[0]; half = torch.arange(nrows) % 2 == 0; table = {}
    def sub(M): return RL.span_of_rows(M, K)[0]
    def sh(M):
        h = torch.arange(M.shape[0]) % 2 == 0; return RL.overlap(sub(M[h]), sub(M[~h]))
    for l in range(nl - 1):
        Rb, Pb, Jb = [sub(rows['base'][i][l]) for i in range(3)]; Jwl = RL.top_right(Jw[l], K)[0]
        e = {'R_split_half': sh(rows['base'][0][l]), 'J_split_half': sh(rows['base'][2][l]), 'P_split_half': sh(rows['base'][1][l]),
             'Jproxy_vs_Jwiki': RL.overlap(Jb, Jwl), 'R_vs_Jwiki': RL.overlap(Rb, Jwl), 'R_vs_Jproxy': RL.overlap(Rb, Jb)}
        for name in ('experience', 'control'):
            Rs, Ps, Js = [sub(rows[name][i][l]) for i in range(3)]; e['R_reorg_' + name] = 1 - RL.overlap(Rb, Rs); e['J_reorg_' + name] = 1 - RL.overlap(Jb, Js); e['P_reorg_' + name] = 1 - RL.overlap(Pb, Ps)
        table[l] = e
    upper = [l for l in table if 14 <= l <= 26]
    a_ok = [table[l]['R_reorg_experience'] > table[l]['J_reorg_experience'] for l in upper]; b_ok = [table[l]['R_reorg_experience'] > table[l]['R_reorg_control'] for l in upper]
    meaningful = [ (1 - table[l]['R_reorg_experience']) < table[l]['R_split_half'] for l in upper]
    res['table'] = table; res['proxy_validation'] = {'Jproxy_vs_Jwiki_mean_upper': sum(table[l]['Jproxy_vs_Jwiki'] for l in upper) / len(upper), 'R_vs_Jwiki_mean_upper': sum(table[l]['R_vs_Jwiki'] for l in upper) / len(upper), 'pass': sum(table[l]['Jproxy_vs_Jwiki'] for l in upper) / len(upper) >= 0.2}
    res['criteria'] = {'a_R_over_J_layers': sum(a_ok), 'b_R_over_control_layers': sum(b_ok), 'n_upper_layers': len(upper), 'meaningful_change_layers': sum(meaningful), 'pass': sum(a_ok) > len(upper) / 2 and sum(b_ok) > len(upper) / 2}
    log('proxy validation %s' % res['proxy_validation']); log('criteria %s' % res['criteria'])
    log('layer  Rrel  Jrel  Jp-Jw  R-Jw   R_exp  J_exp  P_exp  R_ctl  J_ctl  P_ctl')
    for l in table: e = table[l]; log('%5d  %.2f  %.2f  %.2f   %.2f   %.3f  %.3f  %.3f  %.3f  %.3f  %.3f' % (l, e['R_split_half'], e['J_split_half'], e['Jproxy_vs_Jwiki'], e['R_vs_Jwiki'], e['R_reorg_experience'], e['J_reorg_experience'], e['P_reorg_experience'], e['R_reorg_control'], e['J_reorg_control'], e['P_reorg_control']))
    json.dump(res, open(os.path.join(out, 'results.json'), 'w'), indent=1, default=str)
    # steering
    LM.load_lora_state(model, init); base_prof, base_lp = profile_and_dist(tok, model, held); steer = {}
    for l in STEER_LAYERS:
        if l >= nl - 1: continue
        dR = RL.top_right((rows['experience'][0][l].float() - rows['base'][0][l].float()), 1)[0][:, 0]; dJ = RL.top_right((rows['experience'][2][l].float() - rows['base'][2][l].float()), 1)[0][:, 0]; rnd = torch.randn(d, generator=torch.Generator().manual_seed(l)); rnd = rnd / rnd.norm()
        with torch.no_grad():
            x = torch.tensor([tok(held[0], add_special_tokens=False)['input_ids'][:T]], device='cuda'); hn = float(model(input_ids=x, output_hidden_states=True).hidden_states[l + 1][0].float().norm(dim=1).mean())
        c = cfg['steer_frac'] * hn; steer[l] = {'residual_norm': hn, 'c': c}
        for name, vec in (('d_R', dR), ('d_J', dJ), ('random', rnd)):
            profs, lps = profile_and_dist(tok, model, held, layer=l, vec=c * vec.cuda())
            dS = sum(float((p - q).abs().sum()) for p, q in zip(profs, base_prof)) / len(held); kl = sum(float((torch.exp(q) * (q - p)).sum(-1).mean()) for p, q in zip(lps, base_lp)) / len(held)
            steer[l][name] = {'attention_profile_L1': dS, 'next_token_KL': kl, 'ratio': dS / (kl + 1e-9)}
        log('steer layer %d: %s' % (l, {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in steer[l].items() if isinstance(v, dict)}))
    res['steering'] = steer; res['steering_criteria'] = {l: (steer[l]['d_R']['ratio'] > steer[l]['d_J']['ratio'] and steer[l]['d_R']['ratio'] > steer[l]['random']['ratio']) for l in steer}
    res['status'] = 'complete'; res['seconds'] = time.time() - T0; json.dump(res, open(os.path.join(out, 'results.json'), 'w'), indent=1, default=str); log('done in %.0f s: %s | steering %s' % (res['seconds'], res['criteria'], res['steering_criteria']))


if __name__ == '__main__':
    main()


def fit_readouts_batched(tok, model, prompts, U, log, tag, chunk=16):
    """Same rows as fit_readouts (same order), with the per-prompt cotangents batched: the N_POS R targets, the N_POS P targets and the N_POS*N_U
    J-proxy targets are each differentiated in one vmapped backward (torch.autograd.grad with is_grads_batched) instead of one backward per target."""
    nl = len(model.model.layers); R = {l: [] for l in range(nl - 1)}; P = {l: [] for l in range(nl - 1)}; J = {l: [] for l in range(nl - 1)}
    for i, text in enumerate(prompts):
        ids = tok(text, add_special_tokens=False)['input_ids'][:T]; x = torch.tensor([ids], device='cuda'); emb = model.get_input_embeddings()(x).detach().requires_grad_(True)
        out = model(inputs_embeds=emb, output_attentions=True, output_hidden_states=True); hs = [out.hidden_states[l + 1] for l in range(nl - 1)]; hf = out.hidden_states[-1][0]
        n = len(ids); mask = torch.triu(torch.ones(n, n, device='cuda'), diagonal=1); S = sum((A[0].float() * mask.T.unsqueeze(0)).sum(dim=(0, 1)) for A in out.attentions)
        lp = torch.log_softmax(out.logits[0].float(), -1); nxt = lp[torch.arange(n - 1), x[0, 1:]]; positions = torch.linspace(SKIP, n - 2, N_POS).round().long().tolist(); pos = torch.tensor(positions, device='cuda')
        Uv = U[i % U.shape[0]].cuda().float(); Jv = (hf[pos].float().unsqueeze(1) * Uv).sum(-1).reshape(-1)          # (N_POS * N_U,), position-major like fit_readouts
        targets = [('R', S[pos], pos), ('P', nxt[pos], pos), ('J', Jv, pos.repeat_interleave(N_U))]
        for t_i, (kind, v, rows_pos) in enumerate(targets):
            m = v.numel(); store = R if kind == 'R' else P if kind == 'P' else J
            for c0 in range(0, m, chunk):
                sel = torch.arange(c0, min(c0 + chunk, m), device='cuda'); eye = torch.zeros(len(sel), m, device='cuda'); eye[torch.arange(len(sel)), sel] = 1.0
                last = t_i == len(targets) - 1 and c0 + chunk >= m
                g = torch.autograd.grad(v, hs, grad_outputs=eye, is_grads_batched=True, retain_graph=not last)
                for l in range(nl - 1):
                    rows = g[l][torch.arange(len(sel)), 0, rows_pos[sel]].detach().half().cpu(); store[l].extend(list(rows))
        if i % 20 == 0: log('  %s readouts (batched) prompt %d/%d' % (tag, i, len(prompts)))
    return {l: torch.stack(v) for l, v in R.items()}, {l: torch.stack(v) for l, v in P.items()}, {l: torch.stack(v) for l, v in J.items()}
