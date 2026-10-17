"""Stage CR1 (PROTOCOL.md): consequence learning by the model's own choices, letter-only, online. The model meets 768 situations (plant and
clinic interleaved) untold, chooses a prior case, an explanation and a first investigation by letter; the world reveals the outcome; a
reward-weighted (REINFORCE with moving baseline) update on the chosen letters' log-probs follows every batch. Arms: base; consequence (true
outcomes); shuffled (outcomes shuffled across situations, rewards unrelated to the cues); text (the same 768 episodes read as text, no
choices; one epoch, matched size). Evaluation letter-only: five questions in plant, clinic and software (held out); forced-choice and short
free-text rule probes; surface control; held-out prose KL; 12-prompt R/J screen.   python exp_consequence.py [--smoke] [--tag CR1]"""
import argparse, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'retrogenesis'))
import torch
import local_models as LM, world as WD, questions as Q, rlens as RL, exp_abstraction as EA, exp_rj as RJ, exp_consolidate as EC, exp_verbal as V

CFG = {'seed': 20260914, 'n_per_domain': 96, 'batch': 8, 'lr': 1e-4, 'rank': 8, 'alpha': 16, 'passes': 1, 'n_items': 96, 'eval_batch': 32, 'n_verbal': 8, 'screen_prompts': 12, 'n_heldout': 32,
       'text_epochs': 1, 'text_lr': 2e-4, 'text_micro': 4, 'cap_minutes': 12.0, 'kl_bar': 0.30}
QS = ['recall', 'interpret', 'pursue']; UPPER = list(range(14, 27))


def is_common(outcome): return 'single shared cause' in outcome


def experience_items(dom, n, seed, pool_seed, balanced=False):
    """Situations with both answer keys per question (built from the same rng state so the option order is identical) and the outcome text.
    balanced=False: the four cue combinations equally (outcomes 1:3 common:separate, so a constant policy earns 0.75); balanced=True: half common,
    half near-miss (one cue only), so a constant policy earns 0.5 and reward favours cue detection (the test-set design, different seeds)."""
    sits = (WD.test_situations if balanced else WD.balanced_situations)(dom, n, seed); pool = WD.balanced_situations(dom, 64, pool_seed); by_id = {p['id']: p for p in pool}; rng = random.Random(seed + 7); items = []
    for s in sits:
        st0 = rng.getstate(); keys = {}
        for flag in (True, False):
            rng.setstate(st0); s2 = dict(s, common=flag); keys[flag] = {'recall': Q.q_recall(s2, pool, rng), 'interpret': Q.q_interpret(s2, rng), 'pursue': Q.q_pursue(s2, rng)}
        for q in ('interpret', 'pursue'): assert keys[True][q]['question'] == keys[False][q]['question']
        rc = keys[True]['recall']; case_common = {'ABCD'[k]: by_id[cid]['common'] for k, cid in enumerate(rc['options'])}
        items.append({'domain': dom, 'situation': s['text'], 'common': s['common'], 'outcome': s['outcome'], 'questions': {q: keys[True][q]['question'] for q in QS},
                      'key': {'interpret': {True: keys[True]['interpret']['answer'], False: keys[False]['interpret']['answer']}, 'pursue': {True: keys[True]['pursue']['answer'], False: keys[False]['pursue']['answer']}, 'recall_case_common': case_common}})
    return items


def vindicated(item, q, outcome, letter):
    """+1 if the chosen letter is what the revealed outcome vindicates, else -1."""
    c = is_common(outcome)
    if q == 'recall': return 1.0 if item['key']['recall_case_common'][letter] == c else -1.0
    return 1.0 if letter == item['key'][q][c] else -1.0


def letter_logprobs(tok, model, prompts, letters, batch):
    out = []
    for i in range(0, len(prompts), batch):
        msgs = [[{'role': 'system', 'content': EA.SYSTEM}, {'role': 'user', 'content': p}] for p in prompts[i:i + batch]]; seqs = [LM.chat_ids(tok, m) for m in msgs]; ids, mask = LM._left_pad(seqs, tok.pad_token_id)
        out.append(torch.log_softmax(model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits[:, -1, :].float(), -1)[:, letters])
    return torch.cat(out)


def consequence_pass(S, tok, model, params, items, outcomes, seed, tag):
    cfg = S.cfg; letters = [tok(l, add_special_tokens=False)['input_ids'][0] for l in 'ABCD']; opt = torch.optim.Adam(params, lr=cfg['lr']); base = {q: 0.0 for q in QS}; trace = []; hist = {q: [] for q in QS}
    beta = cfg.get('kl_beta', 0.0); base_lp = {}
    if beta > 0:                                                                    # KL anchor to the base letter distribution (B is zero before the first update, so this is the base model)
        with torch.no_grad():
            for q in QS:
                for i in range(0, len(items), cfg['batch']):
                    chunk = items[i:i + cfg['batch']]; lp0 = letter_logprobs(tok, model, ['Current situation. ' + it['situation'] + '\n\n' + it['questions'][q] for it in chunk], letters, cfg['batch']).detach()
                    for k_, it in enumerate(chunk): base_lp[(id(it), q)] = lp0[k_]
        S.log('%s: base letter distributions cached for the KL anchor (beta %.3f)' % (tag, beta))
    for p in params: p.requires_grad_(True)
    for ps in range(cfg['passes']):
        for i in range(0, len(items), cfg['batch']):
            chunk = items[i:i + cfg['batch']]; outs = outcomes[i:i + cfg['batch']]; opt.zero_grad()
            for q in QS:                                                        # one backward per question: the graph is freed each time (peak memory a third of the joint version)
                lp = letter_logprobs(tok, model, ['Current situation. ' + it['situation'] + '\n\n' + it['questions'][q] for it in chunk], letters, cfg['batch'])
                probs = lp.exp().detach().clamp_min(1e-6); probs = torch.where(torch.isfinite(probs), probs, torch.full_like(probs, 0.25))     # a collapsing policy produced a device-side assert in multinomial (CR3f control arm)
                choice = torch.multinomial(probs, 1).squeeze(1); r = torch.tensor([vindicated(it, q, o, 'ABCD'[int(c)]) for it, o, c in zip(chunk, outs, choice)], device=lp.device)
                adv = r - base[q]; base[q] = 0.9 * base[q] + 0.1 * float(r.mean()); loss_q = -(adv * lp[torch.arange(len(chunk)), choice]).mean()
                if beta > 0: lp0 = torch.stack([base_lp[(id(it), q)] for it in chunk]); loss_q = loss_q + beta * (lp.exp() * (lp - lp0)).sum(1).mean()
                (loss_q / len(QS)).backward(); hist[q].extend((r > 0).tolist()); del lp
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True); opt.step()
            every, recent = cfg.get('log_every', 12), cfg.get('recent', 96)
            if (i // cfg['batch']) % every == every - 1 or i + cfg['batch'] >= len(items):
                rates = {q: sum(hist[q][-recent:]) / max(1, len(hist[q][-recent:])) for q in QS}; trace.append({'situations': ps * len(items) + i + len(chunk), 'vindicated_recent': rates}); S.log('%s after %d situations: vindicated (last %d) %s' % (tag, ps * len(items) + i + len(chunk), recent, {q: round(v, 2) for q, v in rates.items()}))
    for p in params: p.requires_grad_(False)
    return trace


@torch.no_grad()
def behavioral(tok, model, items, batch):
    res = {}
    for dom, its in items.items():
        res[dom] = {}
        for q in EA.ALLQ: rows = EA.score(tok, model, its, q, batch=batch); res[dom][q] = {'rate': sum(r['correct'] for r in rows) / len(rows), 'chosen': [r['chosen'] for r in rows]}
    res['far3_mean'] = sum(res['software'][q]['rate'] for q in EA.REL3) / 3; res['trained3_mean'] = sum(res[d][q]['rate'] for d in ('plant', 'clinic') for q in EA.REL3) / 6
    res['surface_min'] = min(res[d]['surface']['rate'] for d in items); res['software_recall'] = res['software']['recall']['rate']; return res


@torch.no_grad()
def probes(tok, model, vitems, batch):
    letters = [tok(l, add_special_tokens=False)['input_ids'][0] for l in 'ABCD']; out = {}
    for dom, vi in vitems.items():
        lp = letter_logprobs(tok, model, [v['forced'] for v in vi], letters, batch); forced = sum('ABCD'[int(k)] == v['forced_answer'] for k, v in zip(lp.argmax(1), vi)) / len(vi); tails = []
        for i in range(0, len(vi), batch):
            msgs = [[{'role': 'system', 'content': EA.SYSTEM}, {'role': 'user', 'content': v['free']}] for v in vi[i:i + batch]]; seqs = [LM.chat_ids(tok, m) for m in msgs]; ids, mask = LM._left_pad(seqs, tok.pad_token_id)
            gen = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=60, do_sample=False, pad_token_id=tok.pad_token_id)
            tails.extend(tok.decode([t for t in gen[k, ids.shape[1]:].tolist() if t != tok.pad_token_id], skip_special_tokens=True) for k in range(len(msgs)))
        rub = [V.rubric(t, dom) for t in tails]; out[dom] = {'forced_acc': forced, 'free_both': sum(x['both'] for x in rub) / len(rub), 'free_time': sum(x['time'] for x in rub) / len(rub), 'free_infra': sum(x['infra'] for x in rub) / len(rub), 'free_tails': [t[:300] for t in tails]}
    out['forced_acc'] = sum(out[d]['forced_acc'] for d in vitems) / 3; out['free_both'] = sum(out[d]['free_both'] for d in vitems) / 3; return out


def strip_lora(model):
    for name, mod in list(model.named_modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, LM.LoRALinear): setattr(mod, cn, child.base)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--tag', default='CR1')
    for k_ in ('n_per_domain', 'batch', 'passes', 'screen_prompts', 'eval_batch'): ap.add_argument('--' + k_, type=int, default=None)
    for k_ in ('lr', 'cap_minutes', 'text_lr', 'kl_beta'): ap.add_argument('--' + k_, type=float, default=None)
    ap.add_argument('--questions', default='recall,interpret,pursue', help='which choices are trained from consequences; the others stay evaluation-only'); ap.add_argument('--balanced', action='store_true', help='experience set half common, half near-miss'); ap.add_argument('--control_rule', default='shuffled', help="'shuffled' or a world rule (e.g. 'time'): the control arm's consequences follow that rule on the same situations"); ap.add_argument('--seed', type=int, default=None); a = ap.parse_args(); cfg = dict(CFG); cfg['balanced'] = a.balanced; cfg['control_rule'] = a.control_rule
    if a.seed is not None: cfg['seed'] = a.seed
    for k_ in ('n_per_domain', 'batch', 'passes', 'screen_prompts', 'eval_batch', 'lr', 'cap_minutes', 'text_lr', 'kl_beta'):
        if getattr(a, k_) is not None: cfg[k_] = getattr(a, k_)
    global QS; QS = a.questions.split(','); cfg['trained_questions'] = QS
    if a.smoke: cfg.update(n_per_domain=16, n_items=8, n_verbal=2, screen_prompts=2, n_heldout=4)
    S = EA.Stage(cfg, os.path.join('results', a.tag + ('_smoke' if a.smoke else ''))); S.res['code']['exp_consequence.py'] = EA.file_sha('exp_consequence.py'); torch.manual_seed(cfg['seed']); T0 = time.time()
    tok, model = RL.load_eager(LM.TARGET_DIR); tok.padding_side = 'left'; model.set_attn_implementation('sdpa'); S.log('model loaded')
    items = Q.build_items(seed=cfg['seed'], n=cfg['n_items'], k_pool=64 if not a.smoke else 16); vitems = {d: V.verbal_items(d, cfg['n_verbal'], cfg['seed'] + 40 + i) for i, d in enumerate(('plant', 'clinic', 'software'))}
    prompts, _ = RL.corpus(tok, 120 + 32); screen_prompts, held = prompts[:cfg['screen_prompts']], prompts[120:120 + cfg['n_heldout']]; d = model.config.hidden_size
    U = torch.randn(120, RJ.N_POS, RJ.N_U, d, generator=torch.Generator().manual_seed(cfg['seed'] + 5)); U = U / U.norm(dim=-1, keepdim=True)
    model.set_attn_implementation('eager'); base_prof, base_lps = RJ.profile_and_dist(tok, model, held); model.set_attn_implementation('sdpa')
    def fit(tag): model.set_attn_implementation('eager'); rows = RJ.fit_readouts_batched(tok, model, screen_prompts, U, S.log, tag); model.set_attn_implementation('sdpa'); return rows
    n = cfg['n_per_domain']; pl = experience_items('plant', n, cfg['seed'] + 1, cfg['seed'] + 2, a.balanced); cl = experience_items('clinic', n, cfg['seed'] + 2, cfg['seed'] + 12, a.balanced)
    exp = [x for pair in zip(pl, cl) for x in pair]; outs_true = [it['outcome'] for it in exp]; outs_shuf = list(outs_true); random.Random(cfg['seed'] + 99).shuffle(outs_shuf)
    if a.control_rule != 'shuffled':                                                 # counterfactual control: consequences follow another rule on the same situations
        rng_c = random.Random(cfg['seed'] + 98); gen = WD.test_situations if a.balanced else WD.balanced_situations
        sits_c = [x for pair in zip(gen('plant', n, cfg['seed'] + 1), gen('clinic', n, cfg['seed'] + 2)) for x in pair]           # the same situations, in the same interleaved order as exp
        assert all(s['text'] == it['situation'] for s, it in zip(sits_c, exp))
        rule_c = WD.RULES[a.control_rule]; outs_shuf = [WD.outcome(dict(s, common=bool(rule_c(s['same_time'], s['same_infra'], s['same_desk']) if a.control_rule == 'td' else rule_c(s['same_time'], s['same_infra']))), rng_c) for s in sits_c]
        cfg['control_common_rate'] = sum('single shared cause' in o for o in outs_shuf) / len(outs_shuf)
    S.res['label_stats'] = {'n_situations': len(exp), 'shuffled_type_agreement': sum(is_common(a_) == is_common(b_) for a_, b_ in zip(outs_true, outs_shuf)) / len(exp)}
    texts = [WD.episode_text(dict(it, id='%s-%03d' % (it['domain'][0].upper(), i), text=it['situation']), it['outcome']) for i, it in enumerate(exp)]
    def evaluate(tag, with_kl):
        r = behavioral(tok, model, items, cfg['eval_batch']); r['probes'] = probes(tok, model, vitems, cfg['eval_batch']); r['heldout_kl'] = EC.heldout_kl(tok, model, held, base_lps) if with_kl else 0.0
        S.log('%-11s software c/i/p %.3f recall %.3f | trained c/i/p %.3f | forced %.3f free_both %.3f (infra %.2f time %.2f) | surface %.2f kl %.4f' % (tag, r['far3_mean'], r['software_recall'], r['trained3_mean'], r['probes']['forced_acc'], r['probes']['free_both'],
              sum(r['probes'][d_]['free_infra'] for d_ in ('plant', 'clinic', 'software')) / 3, sum(r['probes'][d_]['free_time'] for d_ in ('plant', 'clinic', 'software')) / 3, r['surface_min'], r['heldout_kl'])); return r
    S.res['arms']['base'] = evaluate('base', False); S.dump(); rows = {}
    t = time.time(); rows['base'] = fit('base'); S.res['timings']['screen_base'] = time.time() - t
    over = lambda: time.time() - T0 > 60 * cfg['cap_minutes']
    for tag in ('consequence', 'shuffled', 'text'):
        if over(): S.res['status'] = 'incomplete: cap before %s' % tag; S.dump(); break
        LM.add_lora(model, cfg['rank'], cfg['alpha'], ('q_proj', 'v_proj'), seed=cfg['seed']); params = [p for _, m in LM.lora_modules(model) for n_, p in m.named_parameters() if not n_.startswith('base')]; t = time.time()
        if tag == 'text': S.cfg = dict(cfg, epochs=cfg['text_epochs'], lr=cfg['text_lr'], micro=cfg['text_micro']); trace = EC.train_texts(S, tok, model, params, texts, cfg['seed'] + 31, tag); S.cfg = cfg
        else: trace = consequence_pass(S, tok, model, params, exp, outs_true if tag == 'consequence' else outs_shuf, cfg['seed'] + 31, tag)
        S.res['timings']['train_' + tag] = time.time() - t; torch.save(LM.lora_state(model), os.path.join(S.out, 'lora_%s.pt' % tag))
        S.res['arms'][tag] = dict(evaluate(tag, True), train_trace=trace); S.dump()
        if tag in ('consequence', 'shuffled') and not over(): t = time.time(); rows[tag] = fit(tag); S.res['timings']['screen_' + tag] = time.time() - t
        strip_lora(model); del params; torch.cuda.empty_cache()                      # the second trained arm ran three times slower than the first in CR3/CR3c/CR3f: allocator pressure
    torch.save(rows, os.path.join(S.out, 'screen_rows.pt')); A = S.res['arms']; g = lambda t, k: A[t][k] if t in A else None
    if 'consequence' in rows and 'shuffled' in rows and not a.smoke:
        sub = lambda M: RL.span_of_rows(M, 64)[0]; table = {}
        for l in range(len(model.model.layers) - 1):
            Rb, Jb = sub(rows['base'][0][l]), sub(rows['base'][2][l]); e = {}
            for tag in ('consequence', 'shuffled'): e['R_' + tag] = 1 - RL.overlap(Rb, sub(rows[tag][0][l])); e['J_' + tag] = 1 - RL.overlap(Jb, sub(rows[tag][2][l]))
            e['R_specific'] = e['R_consequence'] - e['R_shuffled']; e['J_specific'] = e['J_consequence'] - e['J_shuffled']; table[l] = e
        S.res['screen'] = {'table': table, 'R_specific_ge_005_layers': sum(table[l]['R_specific'] >= 0.05 for l in UPPER), 'R_specific_over_J_layers': sum(table[l]['R_specific'] > table[l]['J_specific'] for l in UPPER), 'upper_means': {k: sum(table[l][k] for l in UPPER) / len(UPPER) for k in table[14]}}
        S.log('screen: %s' % {k: v for k, v in S.res['screen'].items() if k != 'table'})
    sc = S.res.get('screen', {}); have = all(t in A for t in ('base', 'consequence', 'shuffled'))
    c = {'application_pass': (g('consequence', 'far3_mean') >= g('base', 'far3_mean') + 0.10 and g('consequence', 'far3_mean') >= g('shuffled', 'far3_mean') + 0.10 and ('text' not in A or g('consequence', 'far3_mean') >= g('text', 'far3_mean') + 0.05)) if have else None,
         'in_domain_pass': (g('consequence', 'trained3_mean') >= g('base', 'trained3_mean') + 0.15 and g('consequence', 'trained3_mean') >= g('shuffled', 'trained3_mean') + 0.15) if have else None,
         'recall_pass': (g('consequence', 'software_recall') >= g('base', 'software_recall') + 0.10 and g('consequence', 'software_recall') >= g('shuffled', 'software_recall') + 0.10) if have else None,
         'verbal_gain_forced': (A['consequence']['probes']['forced_acc'] - A['base']['probes']['forced_acc']) if 'consequence' in A else None, 'verbal_gain_free': (A['consequence']['probes']['free_both'] - A['base']['probes']['free_both']) if 'consequence' in A else None,
         'consequence_minus_text_software': (g('consequence', 'far3_mean') - g('text', 'far3_mean')) if 'text' in A and 'consequence' in A else None,
         'screen': ('candidate' if (sc.get('R_specific_ge_005_layers', 0) > 6.5 and sc.get('R_specific_over_J_layers', 0) > 6.5) else 'none') if sc else None,
         'content_intact': (A['consequence']['surface_min'] >= 0.95 and A['consequence']['heldout_kl'] <= cfg['kl_bar']) if 'consequence' in A else None}
    if have: c['dissociation'] = 'silent' if (c['application_pass'] and max(c['verbal_gain_forced'], c['verbal_gain_free']) < 0.10) else ('reportable' if max(c['verbal_gain_forced'], c['verbal_gain_free']) >= 0.10 else 'nothing learned')
    S.res['criteria'] = c
    if not S.res['status'].startswith('incomplete'): S.res['status'] = 'complete'
    S.res['timings']['total'] = time.time() - T0; S.dump(); S.log('done (%s): %s' % (S.res['status'], c))


if __name__ == '__main__':
    main()
