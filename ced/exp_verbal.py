"""Stage V1 (PROTOCOL.md): experience by consequence, evaluated where the model computes the relation (thinking mode), with the
verbalization-versus-application test. Arms: base; exp2 (episodes with true outcomes from two domains, plant and clinic); exp1 (plant only,
matched count; diversity control); shuf2 (exp2 episodes with outcomes shuffled within domain). Evaluations, all untold: interpret (thinking) in
all domains with software held out; structure-matched recall (thinking) in software; free-text and forced-choice rule verbalization (thinking);
surface control and held-out prose KL (non-thinking); R/P/J-proxy instrument with the matched-change criterion.
  python exp_verbal.py [--smoke] [--tag V1]"""
import argparse, json, os, random, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'retrogenesis'))
import torch
import local_models as LM, world as WD, questions as Q, rlens as RL, exp_abstraction as EA, exp_rj as RJ, exp_consolidate as EC

CFG = {'seed': 20260914, 'n_train_per_domain': 384, 'n_items': 96, 'n_eval': {'plant': 32, 'clinic': 32, 'software': 64}, 'n_verbal': 8, 'max_new': 768, 'gen_batch': 16,
       'epochs': 3, 'lr': 2e-4, 'micro': 4, 'rank': 8, 'alpha': 16, 'k': 64, 'n_inst_prompts': 60, 'n_heldout': 32, 'cap_minutes': 100.0, 'kl_bar': 0.30}
UPPER = list(range(14, 27)); TIME_TERMS = ['same time', 'same hour', 'ten minutes', 'time window', 'close in time', 'same window', 'simultaneous', 'within the same', 'same period', 'timing', 'same day', 'concurrent']
INFRA_TERMS = {'plant': ['loop', 'bus', 'circuit', 'utility'], 'clinic': ['ward', 'bay', 'wing', 'hall', 'floor', 'location'], 'software': ['dependency', 'store', 'authority', 'broker', 'resolver', 'pipeline', 'provider', 'component']}
GENERIC_INFRA = ['infrastructure', 'shared system', 'same system', 'same line', 'same source', 'same equipment', 'same environment']


def episodes(dom, n, seed, shuffle_outcomes=False):
    eps = WD.balanced_situations(dom, n, seed); outs = [e['outcome'] for e in eps]
    if shuffle_outcomes: random.Random(seed + 99).shuffle(outs)
    return [WD.episode_text(e, o) for e, o in zip(eps, outs)]


def verbal_items(dom, n, seed):
    D = WD.DOMAINS[dom]; rng = random.Random(seed); items = []; infra = D['infra']; carriers = D['carrier'] + 's'
    rules = {'time': 'They share a single cause only when all three are logged in the same time window, whatever the %s' % infra, 'infra': 'They share a single cause only when all three %s sit on the same %s, whatever the timing' % (carriers, infra),
             'and': 'They share a single cause only when both hold: the same time window and the same %s' % infra, 'or': 'They share a single cause when either holds: the same time window or the same %s' % infra}
    for i in range(n):
        site = rng.choice(D['site']); keys = list(rules); rng.shuffle(keys); letters = 'ABCD'; ans = letters[keys.index('and')]
        fc = 'You have reviewed many %s from %s. Which rule best describes when three anomalies logged together turn out to share a single cause?\n' % (D['record'], site) + '\n'.join('%s. %s' % (letters[k], rules[key]) for k, key in enumerate(keys)) + '\n\nAnswer with the letter only.'
        ft = 'You have reviewed many %s from %s. In these records, when do three anomalies logged together turn out to share a single cause, and when do they turn out to be separate? State the rule in one sentence.' % (D['record'], site)
        items.append({'domain': dom, 'forced': fc, 'forced_answer': ans, 'free': ft})
    return items


def rubric(text, dom):
    t = text.lower(); time_hit = any(w in t for w in TIME_TERMS); infra_hit = any(w in t for w in INFRA_TERMS[dom] + GENERIC_INFRA)
    return {'time': time_hit, 'infra': infra_hit, 'both': time_hit and infra_hit}


CLOSE = '\n</think>\n\n'


@torch.no_grad()
def generate(tok, model, prompts, system, max_new, batch, answer_tokens=16):
    """Thinking-mode generation with budget forcing: sequences that do not close their thinking block within max_new tokens are closed with
    CLOSE and continued for answer_tokens, so every prompt yields an answer from whatever reasoning was done; 'finished' records which did."""
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]; msgs = [[{'role': 'system', 'content': system}, {'role': 'user', 'content': p}] for p in chunk]
        texts = [tok.apply_chat_template(m, add_generation_prompt=True, enable_thinking=True, tokenize=False) for m in msgs]; seqs = [tok(t, add_special_tokens=False)['input_ids'] for t in texts]
        ids, mask = LM._left_pad(seqs, tok.pad_token_id); gen = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        rows = []; redo = []
        for k in range(len(chunk)):
            g = [t for t in gen[k, ids.shape[1]:].tolist() if t != tok.pad_token_id]; text = tok.decode(g, skip_special_tokens=False); fin = '</think>' in text
            rows.append({'finished': fin, 'tail': text.split('</think>')[-1] if fin else '', 'len': len(g)})
            if not fin: redo.append((k, seqs[k] + [t for t in g if t not in LM.EOS_IDS] + tok(CLOSE, add_special_tokens=False)['input_ids']))
        if redo:
            ids2, mask2 = LM._left_pad([s for _, s in redo], tok.pad_token_id); gen2 = model.generate(input_ids=ids2.cuda(), attention_mask=mask2.cuda(), max_new_tokens=answer_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, (k, _) in enumerate(redo): rows[k]['tail'] = tok.decode([t for t in gen2[j, ids2.shape[1]:].tolist() if t != tok.pad_token_id], skip_special_tokens=False)
        for r in rows:
            tail = r['tail'].replace('<|im_end|>', '').replace('<|endoftext|>', ''); m = re.search(r'\b([ABCD])\b', tail.replace('*', '')); r['letter'] = m.group(1) if m else None; r['tail'] = tail.strip()[:600]
        out.extend(rows)
    return out


def bal(rows):
    pos = [ok for cm, ok in rows if cm]; neg = [ok for cm, ok in rows if not cm]
    return ((sum(pos) / len(pos) if pos else 0.0) + (sum(neg) / len(neg) if neg else 0.0)) / 2


def evaluate(S, tok, model, items, vitems, held, base_lps, tag):
    cfg = S.cfg; res = {}
    for dom, its in items.items():
        its = its[:cfg['n_eval'][dom]]; prompts = ['Current situation. ' + it['situation'] + '\n\n' + it['questions']['interpret']['question'] for it in its]
        g = generate(tok, model, prompts, EA.SYSTEM, cfg['max_new'], cfg['gen_batch']); rows = [(it['common'], (r['letter'] or 'Z') == it['questions']['interpret']['answer']) for it, r in zip(its, g)]
        fin = [(it['common'], (r['letter'] or 'Z') == it['questions']['interpret']['answer']) for it, r in zip(its, g) if r['finished']]
        res[dom] = {'interpret': {'bal_acc': bal(rows), 'acc': sum(ok for _, ok in rows) / len(rows), 'finished': sum(r['finished'] for r in g) / len(g), 'bal_acc_finished': bal(fin) if fin else None, 'mean_len': sum(r['len'] for r in g) / len(g), 'chosen': [r['letter'] for r in g]}}
        S.log('  %s %s interpret: bal %.3f (finished %.2f, bal among finished %s)' % (tag, dom, res[dom]['interpret']['bal_acc'], res[dom]['interpret']['finished'], ('%.3f' % res[dom]['interpret']['bal_acc_finished']) if fin else 'n/a'))
        if dom == 'software':
            prompts = ['Current situation. ' + it['situation'] + '\n\n' + it['questions']['recall']['question'] for it in its]; g = generate(tok, model, prompts, EA.SYSTEM, cfg['max_new'], cfg['gen_batch'])
            hits = [(r['letter'] or 'Z') == it['questions']['recall']['answer'] for it, r in zip(its, g)]; finh = [h for h, r in zip(hits, g) if r['finished']]
            res[dom]['recall'] = {'acc': sum(hits) / len(hits), 'finished': sum(r['finished'] for r in g) / len(g), 'acc_finished': (sum(finh) / len(finh)) if finh else None, 'chosen': [r['letter'] for r in g]}
            S.log('  %s software recall: acc %.3f (finished %.2f)' % (tag, res[dom]['recall']['acc'], res[dom]['recall']['finished']))
        # verbalization: forced choice and free text
        vi = vitems[dom]; gf = generate(tok, model, [v['forced'] for v in vi], EA.SYSTEM, cfg['max_new'], cfg['gen_batch']); gt = generate(tok, model, [v['free'] for v in vi], EA.SYSTEM, cfg['max_new'], cfg['gen_batch'], answer_tokens=60)
        rub = [rubric(r['tail'], dom) for r in gt]
        res[dom]['verbal'] = {'forced_acc': sum((r['letter'] or 'Z') == v['forced_answer'] for v, r in zip(vi, gf)) / len(vi), 'forced_finished': sum(r['finished'] for r in gf) / len(gf), 'forced_chosen': [r['letter'] for r in gf],
                              'free_both': sum(x['both'] for x in rub) / len(rub), 'free_time': sum(x['time'] for x in rub) / len(rub), 'free_infra': sum(x['infra'] for x in rub) / len(rub), 'free_finished': sum(r['finished'] for r in gt) / len(gt), 'free_tails': [r['tail'] for r in gt]}
        S.log('  %s %s verbal: forced_acc %.3f free_both %.3f (time %.2f infra %.2f)' % (tag, dom, res[dom]['verbal']['forced_acc'], res[dom]['verbal']['free_both'], res[dom]['verbal']['free_time'], res[dom]['verbal']['free_infra']))
    res['surface_min'] = min(sum(r['correct'] for r in EA.score(tok, model, items[d][:cfg['n_eval'][d]], 'surface')) / cfg['n_eval'][d] for d in items)
    res['heldout_kl'] = EC.heldout_kl(tok, model, held, base_lps) if base_lps is not None else 0.0
    res['summary'] = {'software_interpret': res['software']['interpret']['bal_acc'], 'trained_interpret': (res['plant']['interpret']['bal_acc'] + res['clinic']['interpret']['bal_acc']) / 2, 'software_recall': res['software']['recall']['acc'],
                      'forced_acc': sum(res[d]['verbal']['forced_acc'] for d in items) / 3, 'free_both': sum(res[d]['verbal']['free_both'] for d in items) / 3}
    S.log('%-6s software interpret %.3f trained %.3f recall %.3f | verbal forced %.3f free_both %.3f | surface %.2f kl %.4f' % (tag, res['summary']['software_interpret'], res['summary']['trained_interpret'], res['summary']['software_recall'], res['summary']['forced_acc'], res['summary']['free_both'], res['surface_min'], res['heldout_kl']))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--tag', default='V1'); a = ap.parse_args(); cfg = dict(CFG)
    if a.smoke: cfg.update(n_train_per_domain=8, n_items=8, n_eval={'plant': 4, 'clinic': 4, 'software': 4}, n_verbal=2, max_new=64, epochs=1, n_inst_prompts=4, n_heldout=4)
    S = EA.Stage(cfg, os.path.join('results', a.tag + ('_smoke' if a.smoke else ''))); S.res['code']['exp_verbal.py'] = EA.file_sha('exp_verbal.py'); torch.manual_seed(cfg['seed']); T0 = time.time()
    tok, model = RL.load_eager(LM.TARGET_DIR); tok.padding_side = 'left'; S.log('model loaded (eager, thinking-mode generation)')
    items = Q.build_items(seed=cfg['seed'], n=cfg['n_items'], k_pool=64 if not a.smoke else 16); vitems = {d: verbal_items(d, cfg['n_verbal'], cfg['seed'] + 40 + i) for i, d in enumerate(('plant', 'clinic', 'software'))}
    prompts, _ = RL.corpus(tok, 120 + 32); inst_prompts, held = prompts[:cfg['n_inst_prompts']], prompts[120:120 + cfg['n_heldout']]; d = model.config.hidden_size
    U = torch.randn(120, RJ.N_POS, RJ.N_U, d, generator=torch.Generator().manual_seed(cfg['seed'] + 5)); U = U / U.norm(dim=-1, keepdim=True); base_prof, base_lps = RJ.profile_and_dist(tok, model, held)
    model.set_attn_implementation('sdpa')                       # sdpa for generation and scoring (1.2x faster, same answers); eager only around the instrument fits
    def fit(tag):
        model.set_attn_implementation('eager'); rows = RJ.fit_readouts(tok, model, inst_prompts, U, S.log, tag); model.set_attn_implementation('sdpa'); return rows
    n = cfg['n_train_per_domain']; sets = {'exp2': episodes('plant', n, cfg['seed'] + 1) + episodes('clinic', n, cfg['seed'] + 2), 'exp1': episodes('plant', 2 * n, cfg['seed'] + 1),
                                          'shuf2': episodes('plant', n, cfg['seed'] + 1, True) + episodes('clinic', n, cfg['seed'] + 2, True)}
    S.res['arms']['base'] = evaluate(S, tok, model, items, vitems, held, None, 'base'); S.dump()
    inst_rows = {}; over = lambda: time.time() - T0 > 60 * cfg['cap_minutes']
    t = time.time(); inst_rows['base'] = fit('base'); S.res['timings']['inst_base'] = time.time() - t; torch.save(inst_rows, os.path.join(S.out, 'inst_rows.pt'))
    for tag in ('exp2', 'shuf2', 'exp1'):
        if over(): S.res['status'] = 'incomplete: cap before %s' % tag; S.dump(); break
        n_lora = LM.add_lora(model, cfg['rank'], cfg['alpha'], ('q_proj', 'v_proj'), seed=cfg['seed']); params = [p for _, m in LM.lora_modules(model) for n_, p in m.named_parameters() if not n_.startswith('base')]
        t = time.time(); trace = EC.train_texts(S, tok, model, params, sets[tag], cfg['seed'] + 31, tag); S.res['timings']['train_' + tag] = time.time() - t; torch.save(LM.lora_state(model), os.path.join(S.out, 'lora_%s.pt' % tag))
        S.res['arms'][tag] = dict(evaluate(S, tok, model, items, vitems, held, base_lps, tag), train_trace=trace); S.dump()
        if tag in ('exp2', 'shuf2') and not over():
            t = time.time(); inst_rows[tag] = fit(tag); S.res['timings']['inst_' + tag] = time.time() - t; torch.save(inst_rows, os.path.join(S.out, 'inst_rows.pt'))
        for name, mod in list(model.named_modules()):
            for cn, child in list(mod.named_children()):
                if isinstance(child, LM.LoRALinear): setattr(mod, cn, child.base)
    A = S.res['arms']; g = lambda t, k: A[t]['summary'][k] if t in A else None
    if 'exp2' in inst_rows and 'shuf2' in inst_rows and not a.smoke:
        sub = lambda M: RL.span_of_rows(M, cfg['k'])[0]; table = {}; Rb_, Pb_, Jb_ = inst_rows['base']
        for l in range(len(model.model.layers) - 1):
            Rb, Jb = sub(Rb_[l]), sub(Jb_[l]); e = {}
            for tag in ('exp2', 'shuf2'): e['R_' + tag] = 1 - RL.overlap(Rb, sub(inst_rows[tag][0][l])); e['J_' + tag] = 1 - RL.overlap(Jb, sub(inst_rows[tag][2][l]))
            e['R_specific'] = e['R_exp2'] - e['R_shuf2']; e['J_specific'] = e['J_exp2'] - e['J_shuf2']; e['R_frac'] = e['R_specific'] / max(e['R_exp2'], 1e-6); e['J_frac'] = e['J_specific'] / max(e['J_exp2'], 1e-6); table[l] = e
        S.res['instrument'] = {'table': table, 'R_specific_ge_005_layers': sum(table[l]['R_specific'] >= 0.05 for l in UPPER), 'R_frac_over_J_frac_layers': sum(table[l]['R_frac'] > table[l]['J_frac'] for l in UPPER), 'n_upper': len(UPPER),
                               'upper_means': {k: sum(table[l][k] for l in UPPER) / len(UPPER) for k in table[14]}}
        S.log('instrument: %s' % {k: v for k, v in S.res['instrument'].items() if k != 'table'})
    I = S.res.get('instrument', {}); have = all(t in A for t in ('base', 'exp2', 'shuf2'))
    S.res['criteria'] = {'application_pass': (g('exp2', 'software_interpret') >= g('base', 'software_interpret') + 0.10 and g('exp2', 'software_interpret') >= g('shuf2', 'software_interpret') + 0.10) if have else None,
                         'recall_pass': (g('exp2', 'software_recall') >= g('base', 'software_recall') + 0.10 and g('exp2', 'software_recall') >= g('shuf2', 'software_recall') + 0.10) if have else None,
                         'verbal_gain_forced': (g('exp2', 'forced_acc') - g('base', 'forced_acc')) if have else None, 'verbal_gain_free': (g('exp2', 'free_both') - g('base', 'free_both')) if have else None,
                         'instrument_pass': (I.get('R_specific_ge_005_layers', 0) > 6.5 and I.get('R_frac_over_J_frac_layers', 0) > 6.5) if I else None,
                         'diversity_gain': (g('exp2', 'software_interpret') - g('exp1', 'software_interpret')) if 'exp1' in A and 'exp2' in A else None,
                         'content_intact': (A['exp2']['surface_min'] >= 0.95 and A['exp2']['heldout_kl'] <= cfg['kl_bar']) if 'exp2' in A else None}
    c = S.res['criteria']
    if have: c['dissociation'] = 'silent' if (c['application_pass'] and max(c['verbal_gain_forced'], c['verbal_gain_free']) < 0.10) else ('reportable' if max(c['verbal_gain_forced'], c['verbal_gain_free']) >= 0.10 else 'nothing learned')
    if not S.res['status'].startswith('incomplete'): S.res['status'] = 'complete'
    S.res['timings']['total'] = time.time() - T0; S.dump(); S.log('done (%s): %s' % (S.res['status'], c))


if __name__ == '__main__':
    main()
