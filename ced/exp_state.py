"""Stage ST1 (PROTOCOL.md): state-gated internalization screen on Qwen3-1.7B. The CR3g consequence recipe (balanced experience set, KL anchor 0.03,
3 passes, interpret+pursue trained by REINFORCE on the model's own letter choices, lr 5e-4, batch 4) is run with the model held in a different
internal state during the encounters only: a fixed vector (results/ST1/dirs.pt, fitted by state_dirs.py) is added at the final prompt position at
one layer during every training forward pass (the KL-anchor cache, choice sampling and log-probs). Evaluation is always clean: adapter on, no
injection. Arms share seed, episodes, outcomes, update rule and anchor; only the state at the encounter differs. No R/J screen, no text arm; the
shuffled control runs only for a winning state in a follow-up.   python exp_state.py --tag ST1 --seed 20260920 [--arms none,think+,think-,cr+,cr-,rand+]"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'retrogenesis'))
import torch
import local_models as LM, questions as Q, rlens as RL, exp_abstraction as EA, exp_rj as RJ, exp_consolidate as EC, exp_verbal as V, exp_consequence as XC
from state_dirs import hook_last


def consequence_pass_exact(S, tok, model, params, items, outcomes, seed, tag):
    """ST1 amendment: the exact policy gradient over the four letters. The outcome does not depend on the choice, so once it is revealed every
    letter's vindication r(l) is known; loss = -sum_l p(l) (r(l) - E_p r) log p(l), with the same KL anchor as consequence_pass. No sampling, so no
    variance-driven collapse; the update is still weighted by the model's own choice distribution. The trace logs the expected vindicated rate."""
    cfg = S.cfg; QS = XC.QS; letters = [tok(l, add_special_tokens=False)['input_ids'][0] for l in 'ABCD']; opt = torch.optim.Adam(params, lr=cfg['lr']); trace = []; hist = {q: [] for q in QS}; ent = {q: [] for q in QS}; mass = {q: [] for q in QS}; base = {q: 0.0 for q in QS}
    beta = cfg.get('kl_beta', 0.0); base_lp = {}
    if beta > 0:
        with torch.no_grad():
            for q in QS:
                for i in range(0, len(items), cfg['batch']):
                    chunk = items[i:i + cfg['batch']]; lp0 = XC.letter_logprobs(tok, model, ['Current situation. ' + it['situation'] + '\n\n' + it['questions'][q] for it in chunk], letters, cfg['batch']).detach()
                    for k_, it in enumerate(chunk): base_lp[(id(it), q)] = lp0[k_]
        S.log('%s: base letter distributions cached for the KL anchor (beta %.3f)' % (tag, beta))
    for p in params: p.requires_grad_(True)
    for ps in range(cfg['passes']):
        for i in range(0, len(items), cfg['batch']):
            chunk = items[i:i + cfg['batch']]; outs = outcomes[i:i + cfg['batch']]; opt.zero_grad()
            for q in QS:
                lp_raw = XC.letter_logprobs(tok, model, ['Current situation. ' + it['situation'] + '\n\n' + it['questions'][q] for it in chunk], letters, cfg['batch'])
                tau = cfg.get('tau', 1.0); lp = lp_raw / tau - torch.logsumexp(lp_raw / tau, 1, keepdim=True)      # the policy is the distribution over the four letters (amendment 2); at temperature tau > 1 (amendment 5, ST1b) it cannot saturate to a constant letter, which froze think+ and rand+ in ST1
                R = torch.tensor([[XC.vindicated(it, q, o, L) for L in 'ABCD'] for it, o in zip(chunk, outs)], device=lp.device); p = lp.exp().detach(); p = torch.where(torch.isfinite(p), p, torch.full_like(p, 0.25))
                eps = cfg.get('eps', 0.0); pm = (1 - eps) * p + eps / 4                                                                                   # ST2 amendment: epsilon mixture for the gradient weights, so a wrong constant letter can still be escaped (the saturation trap froze arms in every run, including a plain arm on seed 20260919)
                er = (pm * R).sum(1); adv = R - base[q]; base[q] = 0.9 * base[q] + 0.1 * float(er.mean()); loss_q = -((pm * adv) * lp).sum(1).mean()    # moving baseline as in CR3g (amendment 3): the per-item expectation gave a zero advantage on a near-deterministic policy
                if beta > 0: lp0 = torch.stack([base_lp[(id(it), q)] for it in chunk]); loss_q = loss_q + beta * (lp_raw.exp() * (lp_raw - lp0)).sum(1).mean()      # anchor on the raw log-probs, as in CR3g: it also holds the letters' total mass
                if cfg.get('mass_gamma', 0.0) > 0: loss_q = loss_q + cfg['mass_gamma'] * (-torch.logsumexp(lp_raw, 1)).mean()                                              # amendment 4 (ST1b): keep the answer on the letters; the think- arm at dose 0.2 drove the letters' mass to zero
                (loss_q / len(QS)).backward(); hist[q].extend((p * (R > 0).float()).sum(1).tolist()); ent[q].extend((-(p * p.clamp_min(1e-9).log()).sum(1)).tolist()); mass[q].extend(lp_raw.exp().sum(1).detach().tolist()); del lp, lp_raw
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True); opt.step()
            seen = ps * len(items) + i + len(chunk); ck = cfg.get('ckpt_every', 0)
            if ck and seen % ck == 0: torch.save(LM.lora_state(model), os.path.join(S.out, 'ckpt_%s_%04d.pt' % (tag.replace('+', 'p').replace('-', 'm'), seen)))      # P1: dense checkpoints along the run
            every, recent = cfg.get('log_every', 12), cfg.get('recent', 96)
            if (i // cfg['batch']) % every == every - 1 or i + cfg['batch'] >= len(items):
                rates = {q: sum(hist[q][-recent:]) / max(1, len(hist[q][-recent:])) for q in QS}; ents = {q: sum(ent[q][-recent:]) / max(1, len(ent[q][-recent:])) for q in QS}; ms = {q: sum(mass[q][-recent:]) / max(1, len(mass[q][-recent:])) for q in QS}
                trace.append({'situations': ps * len(items) + i + len(chunk), 'vindicated_recent': rates, 'entropy_recent': ents, 'letter_mass_recent': ms}); S.log('%s after %d situations: expected vindicated (last %d) %s entropy %s letter mass %s' % (tag, ps * len(items) + i + len(chunk), recent, {q: round(v, 2) for q, v in rates.items()}, {q: round(v, 2) for q, v in ents.items()}, {q: round(v, 2) for q, v in ms.items()}))
    for p in params: p.requires_grad_(False)
    return trace


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='ST1'); ap.add_argument('--seed', type=int, default=20260920); ap.add_argument('--arms', default='none,think+,think-,cr+,cr-,rand+'); ap.add_argument('--dose', type=float, default=None); ap.add_argument('--doses', default=None, help='per-direction doses, e.g. think=0.08,cr=0.1,rand=0.1 (ST1b); overrides --dose'); ap.add_argument('--cap_minutes', type=float, default=55.0); ap.add_argument('--mass_gamma', type=float, default=0.0); ap.add_argument('--tau', type=float, default=1.0); ap.add_argument('--eps', type=float, default=0.0); ap.add_argument('--ckpt_every', type=int, default=0); ap.add_argument('--dirs_path', default=None, help='direction file (default results/ST1/dirs.pt; P2 uses results/P2/dirs.pt)'); ap.add_argument('--questions', default='interpret,pursue', help='which choices receive consequences (D2: the structure of the experience)'); ap.add_argument('--kl_beta', type=float, default=0.03); ap.add_argument('--proxy', action='store_true', help='AL1: the desk cue is a perfect proxy for the rule in the experience items only (needs CED_THIRD_CUE=1)'); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--update', default='exact', help="'exact' (ST1 amendment: exact policy gradient over the letters) or 'sampled' (CR3g REINFORCE)"); a = ap.parse_args()
    QS = a.questions.split(','); cfg = dict(XC.CFG); cfg.update(seed=a.seed, n_per_domain=192, batch=4, eval_batch=8, lr=5e-4, passes=3, kl_beta=a.kl_beta, balanced=True, control_rule='time', trained_questions=QS, cap_minutes=a.cap_minutes, arms=a.arms.split(','), update=a.update, mass_gamma=a.mass_gamma, tau=a.tau, eps=a.eps, ckpt_every=a.ckpt_every)
    if a.smoke: cfg.update(n_per_domain=8, n_items=8, n_verbal=2, n_heldout=4, passes=1)
    XC.QS = QS; dp = a.dirs_path or os.path.join('results', 'ST1', 'dirs.pt'); D = torch.load(dp); L = D['layer']; dose = a.dose if a.dose is not None else json.load(open(dp[:-3] + '.json'))['dose']; cfg.update(layer=L, dose=dose, dirs_path=dp)
    S = EA.Stage(cfg, os.path.join('results', a.tag + ('_smoke' if a.smoke else ''))); S.res['code'] = {f: EA.file_sha(f) for f in ('exp_state.py', 'state_dirs.py', 'exp_consequence.py')}; torch.manual_seed(cfg['seed']); T0 = time.time()
    tok, model = RL.load_eager(LM.TARGET_DIR); tok.padding_side = 'left'; model.set_attn_implementation('sdpa'); S.log('model loaded; layer %d dose %.2f arms %s' % (L, dose, cfg['arms']))
    items = Q.build_items(seed=cfg['seed'], n=cfg['n_items'], k_pool=64 if not a.smoke else 16); vitems = {d: V.verbal_items(d, cfg['n_verbal'], cfg['seed'] + 40 + i) for i, d in enumerate(('plant', 'clinic', 'software'))}
    prompts, _ = RL.corpus(tok, 120 + 32); held = prompts[120:120 + cfg['n_heldout']]
    model.set_attn_implementation('eager'); base_prof, base_lps = RJ.profile_and_dist(tok, model, held); model.set_attn_implementation('sdpa')
    n = cfg['n_per_domain']; cfg['proxy'] = bool(a.proxy)
    import world as WD
    if a.proxy: assert WD.THIRD_CUE, '--proxy needs CED_THIRD_CUE=1'
    WD.PROXY = bool(a.proxy); pl = XC.experience_items('plant', n, cfg['seed'] + 1, cfg['seed'] + 2, True); cl = XC.experience_items('clinic', n, cfg['seed'] + 2, cfg['seed'] + 12, True); WD.PROXY = False      # AL1: the proxy couples the desk to the rule in the experience only; evaluation items (built above) keep the desk random
    exp = [x for pair in zip(pl, cl) for x in pair]; outs_true = [it['outcome'] for it in exp]; S.res['label_stats'] = {'n_situations': len(exp), 'common_rate': sum(XC.is_common(o) for o in outs_true) / len(outs_true)}
    def evaluate(tag, with_kl):
        r = XC.behavioral(tok, model, items, cfg['eval_batch']); r['probes'] = XC.probes(tok, model, vitems, cfg['eval_batch']); r['heldout_kl'] = EC.heldout_kl(tok, model, held, base_lps) if with_kl else 0.0
        S.log('%-7s software c/i/p %.3f recall %.3f | trained c/i/p %.3f | forced %.3f free_both %.3f | surface %.2f kl %.4f' % (tag, r['far3_mean'], r['software_recall'], r['trained3_mean'], r['probes']['forced_acc'], r['probes']['free_both'], r['surface_min'], r['heldout_kl'])); return r
    S.res['arms']['base'] = evaluate('base', False); S.dump()
    per_dose = {k: float(v) for k, v in (x.split('=') for x in a.doses.split(','))} if a.doses else {}; cfg['per_direction_dose'] = per_dose
    def vec_for(arm):
        if arm == 'none': return None
        name, sign = arm[:-1], (1.0 if arm[-1] == '+' else -1.0); dname = name if name in D['dirs'] else name.rstrip('h')      # 'crh' = the cr direction at its own (half) dose
        return (sign * per_dose.get(name, dose) * D['mean_norm'] * D['dirs'][dname]).cuda()
    over = lambda: time.time() - T0 > 60 * cfg['cap_minutes']
    for arm in cfg['arms']:
        if over(): S.res['status'] = 'incomplete: cap before %s' % arm; S.dump(); break
        torch.manual_seed(cfg['seed']); LM.add_lora(model, cfg['rank'], cfg['alpha'], ('q_proj', 'v_proj'), seed=cfg['seed']); params = [p for _, m in LM.lora_modules(model) for n_, p in m.named_parameters() if not n_.startswith('base')]
        v = vec_for(arm); hk = hook_last(model, L, v) if v is not None else None; t = time.time()
        try: trace = (consequence_pass_exact if cfg['update'] == 'exact' else XC.consequence_pass)(S, tok, model, params, exp, outs_true, cfg['seed'] + 31, arm)
        finally:
            if hk: hk.remove()
        S.res['timings']['train_' + arm] = time.time() - t; torch.save(LM.lora_state(model), os.path.join(S.out, 'lora_%s.pt' % arm.replace('+', 'p').replace('-', 'm')))
        S.res['arms'][arm] = dict(evaluate(arm, True), train_trace=trace, final_vindicated=trace[-1]['vindicated_recent'] if trace else None); S.dump()
        XC.strip_lora(model); del params; torch.cuda.empty_cache()
    A = S.res['arms']; g = lambda t, k: A[t][k] if t in A else None; c = {}
    if 'none' in A:
        for arm in cfg['arms']:
            if arm in A and arm != 'none': c[arm] = {'far_delta': g(arm, 'far3_mean') - g('none', 'far3_mean'), 'trained_delta': g(arm, 'trained3_mean') - g('none', 'trained3_mean'), 'kl': g(arm, 'heldout_kl'), 'surface': g(arm, 'surface_min'), 'verbal_gain': max(A[arm]['probes']['forced_acc'] - A['base']['probes']['forced_acc'], A[arm]['probes']['free_both'] - A['base']['probes']['free_both'])}
        state_arms = [x for x in c if not x.startswith('rand')]; rand_arms = [x for x in c if x.startswith('rand')]
        c['C1_state_changes_write'] = any(abs(c[x]['far_delta']) >= 0.10 and c[x]['surface'] >= 0.95 and c[x]['kl'] <= cfg['kl_bar'] for x in state_arms) and all(abs(c[x]['far_delta']) < 0.05 for x in rand_arms)
        c['C2_sign'] = {'think': (g('think+', 'far3_mean') is not None and g('think-', 'far3_mean') is not None and g('think+', 'far3_mean') >= g('none', 'far3_mean') >= g('think-', 'far3_mean')), 'cr': (g('cr+', 'far3_mean') is not None and g('cr-', 'far3_mean') is not None and g('cr+', 'far3_mean') >= g('none', 'far3_mean') >= g('cr-', 'far3_mean'))}
        c['C3_silent_all'] = all(c[x]['verbal_gain'] < 0.10 for x in state_arms + rand_arms)
        c['kill_state_independent'] = all(abs(c[x]['far_delta']) < 0.05 and abs(c[x]['trained_delta']) < 0.05 for x in state_arms)
        c['none_replicates_CR3g'] = g('none', 'far3_mean') >= g('base', 'far3_mean') + 0.10
    S.res['criteria'] = c
    if not S.res['status'].startswith('incomplete'): S.res['status'] = 'complete'
    S.res['timings']['total'] = time.time() - T0; S.dump(); S.log('done (%s): %s' % (S.res['status'], json.dumps(c)))


if __name__ == '__main__':
    main()
