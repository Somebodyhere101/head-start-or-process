"""Stage E1 (PROTOCOL.md): abstraction-augmented experience on Qwen3-1.7B. Episodes plus comparison records (oracle-templated or
self-generated) against episodes-only and base; evaluated on the lens item sets with all five questions.   python exp_abstraction.py [--smoke]"""
import argparse, hashlib, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'retrogenesis'))
import torch, torch.nn.functional as Fnn
import local_models as LM, world as WD, questions as Q

MODEL = LM.TARGET_DIR; SYSTEM = 'You are an analyst reviewing incident records. Answer multiple-choice questions with the letter only.'
CFG = {'seed': 20260914, 'n_train': 192, 'n_pool': 64, 'n_items': 96, 'epochs': 3, 'lr': 2e-4, 'micro': 4, 'lora_rank': 8, 'lora_alpha': 16, 'lora_targets': ['q_proj', 'v_proj'], 'group': 4, 'self_max_new': 400, 'cap_minutes': 120.0}
REL3 = ['connect', 'interpret', 'pursue']; ALLQ = ['recall'] + REL3 + ['surface']


def file_sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def cue_phrase(ep):
    return ('began %s' % ('inside one time window' if ep['same_time'] else 'far apart in time')) + ' and ' + (('all sat on %s' % ep['infra']) if ep['same_infra'] else 'sat on different loops, buses or circuits')


def oracle_record(group):
    lines = ['Comparing cases %s.' % ', '.join(e['id'] for e in group)]
    for e in group: lines.append('In case %s the anomalies %s; the outcome was %s.' % (e['id'], cue_phrase(e), 'one shared cause' if e['common'] else 'separate causes'))
    lines.append('What these cases have in common: a shared cause appears exactly when the anomalies begin in the same time window and their units share one loop; when either condition is missing, the causes are separate. The time window alone and the shared loop alone do not decide it.')
    return ' '.join(lines)


def self_prompt(group):
    return ('Here are four incident cases with their investigation outcomes.\n\n' + '\n\n'.join(WD.episode_text(e) for e in group) +
            '\n\nCompare the four cases. State in a short paragraph what the cases have in common and what predicts whether the anomalies share one cause or have separate causes. Refer to the cases by their ids.')


@torch.no_grad()
def generate_self_records(tok, model, groups, max_new):
    outs = []
    for g in groups:
        text = tok.apply_chat_template([{'role': 'user', 'content': self_prompt(g)}], add_generation_prompt=True, enable_thinking=True, tokenize=False); ids = torch.tensor([tok(text, add_special_tokens=False)['input_ids']], device=model.device)
        gen = model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=LM.EOS_IDS, temperature=None, top_p=None, top_k=None)[0, ids.shape[1]:]
        full = tok.decode(gen, skip_special_tokens=True); answer = full.split('</think>')[-1].strip() if '</think>' in full else full.strip()
        outs.append({'ids': [e['id'] for e in g], 'raw': full, 'answer': answer, 'record': 'Comparing cases %s. %s' % (', '.join(e['id'] for e in g), answer)})
    return outs


def rule_stated(text):
    t = text.lower(); return {'mentions_time': any(w in t for w in ('time', 'hour', 'minutes', 'window', 'same day', 'apart')), 'mentions_infra': any(w in t for w in ('loop', 'bus', 'circuit', 'same unit', 'shared', 'infrastructure', 'system')), 'mentions_both_needed': any(w in t for w in (' and ', 'both', 'together', 'only when', 'exactly when'))}


def train_lora(S, tok, model, texts, seed, tag):
    cfg = S.cfg; params = [p for _, m in LM.lora_modules(model) for p in (m.A, m.B)]
    for p in params: p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg['lr']); rng = random.Random(seed); ids = [tok(t, add_special_tokens=False)['input_ids'] + [LM.EOS_IDS[1]] for t in texts]; trace = []; ntok = sum(len(x) for x in ids)
    for ep in range(cfg['epochs']):
        order = list(range(len(ids))); rng.shuffle(order); tot = 0.0; n = 0
        for i in range(0, len(order), cfg['micro']):
            batch = [ids[j] for j in order[i:i + cfg['micro']]]; L = max(len(b) for b in batch)
            x = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long); y = torch.full((len(batch), L), -100, dtype=torch.long); m = torch.zeros((len(batch), L), dtype=torch.long)
            for k, b in enumerate(batch): x[k, :len(b)] = torch.tensor(b); y[k, :len(b)] = torch.tensor(b); m[k, :len(b)] = 1
            x, y, m = x.to(model.device), y.to(model.device), m.to(model.device); logits = model(input_ids=x, attention_mask=m).logits[:, :-1].float()
            loss = Fnn.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1), ignore_index=-100); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True); opt.step(); tot += float(loss); n += 1
        trace.append(tot / n); S.log('%s epoch %d: loss %.4f' % (tag, ep, tot / n))
    for p in params: p.requires_grad_(False)
    return trace, ntok


@torch.no_grad()
def score(tok, model, items, q, batch=8):
    letters = [tok(l, add_special_tokens=False)['input_ids'][0] for l in 'ABCD']; rows = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]; msgs = [[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': 'Current situation. ' + it['situation'] + '\n\n' + it['questions'][q]['question']}] for it in chunk]
        seqs = [LM.chat_ids(tok, m) for m in msgs]; ids, mask = LM._left_pad(seqs, tok.pad_token_id); lp = torch.log_softmax(model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device)).logits[:, -1, :].float(), -1)[:, letters].cpu()
        for k, it in enumerate(chunk): rows.append({'situation_id': it['situation_id'], 'common': it['common'], 'chosen': 'ABCD'[int(lp[k].argmax())], 'answer': it['questions'][q]['answer'], 'correct': 'ABCD'[int(lp[k].argmax())] == it['questions'][q]['answer']})
    return rows


class Stage:
    def __init__(self, cfg, out):
        self.cfg, self.out, self.T0 = cfg, out, time.time(); os.makedirs(out, exist_ok=True); self.logf = open(os.path.join(out, 'log.txt'), 'a')
        self.res = {'config': cfg, 'code': {f: file_sha(f) for f in ('exp_abstraction.py', 'world.py', 'questions.py', 'PROTOCOL.md')}, 'model': MODEL, 'status': 'running', 'arms': {}, 'timings': {}}
    def log(self, m):
        s = '[%6.0fs] %s' % (time.time() - self.T0, m); print(s, flush=True); self.logf.write(s + '\n'); self.logf.flush()
    def dump(self): json.dump(self.res, open(os.path.join(self.out, 'results.json'), 'w'), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args(); cfg = dict(CFG)
    if a.smoke: cfg.update(n_train=8, n_items=8, epochs=1, self_max_new=120)
    S = Stage(cfg, os.path.join('results', 'E1_smoke' if a.smoke else 'E1')); torch.manual_seed(cfg['seed'])
    train = WD.balanced_situations('plant', cfg['n_train'], cfg['seed'] + 1); items = Q.build_items(seed=cfg['seed'], n=cfg['n_items'], k_pool=64 if not a.smoke else 16)
    groups = [train[i:i + cfg['group']] for i in range(0, len(train), cfg['group'])]                    # consecutive episodes cover the four cue combinations
    t = time.time(); tok, model = LM.load_causal_lm(MODEL, 'cuda'); n_lora = LM.add_lora(model, cfg['lora_rank'], cfg['lora_alpha'], tuple(cfg['lora_targets']), seed=cfg['seed']); init = LM.lora_state(model); S.log('model loaded, %d LoRA modules (%.0f s)' % (n_lora, time.time() - t))
    ep_texts = [WD.episode_text(e) for e in train]; oracle = [oracle_record(g) for g in groups]
    t = time.time(); LM.load_lora_state(model, init); selfrec = generate_self_records(tok, model, groups, cfg['self_max_new']); S.res['timings']['self_generation'] = time.time() - t
    json.dump({'oracle': oracle, 'self': selfrec}, open(os.path.join(S.out, 'records.json'), 'w'), indent=1)
    rs = [rule_stated(r['answer']) for r in selfrec]; S.res['self_records'] = {'n': len(rs), 'mentions_time': sum(r['mentions_time'] for r in rs) / len(rs), 'mentions_infra': sum(r['mentions_infra'] for r in rs) / len(rs), 'mentions_both_needed': sum(r['mentions_both_needed'] for r in rs) / len(rs), 'mean_answer_chars': sum(len(r['answer']) for r in selfrec) / len(selfrec)}
    S.log('self records generated (%.0f s): %s' % (S.res['timings']['self_generation'], S.res['self_records'])); S.dump()
    def evaluate(tag):
        t = time.time(); res = {}
        for dom, its in items.items():
            res[dom] = {}
            for q in ALLQ: rows = score(tok, model, its, q); res[dom][q] = {'rate': sum(r['correct'] for r in rows) / len(rows), 'rows': rows}
        far3 = [res[d][q]['rate'] for d in ('clinic', 'software') for q in REL3]; res['far3_mean'] = sum(far3) / len(far3); res['plant3_mean'] = sum(res['plant'][q]['rate'] for q in REL3) / 3
        S.res['arms'][tag] = dict(S.res['arms'].get(tag, {}), **res); S.res['timings']['eval_' + tag] = time.time() - t
        S.log('%-18s far3 %.3f plant3 %.3f | %s (%.0f s)' % (tag, res['far3_mean'], res['plant3_mean'], {d: {q: round(res[d][q]['rate'], 2) for q in ALLQ} for d in items}, time.time() - t)); S.dump()
    LM.load_lora_state(model, init); evaluate('base')
    arms = {'episodes_only': ep_texts, 'oracle_abstraction': ep_texts + oracle, 'self_abstraction': ep_texts + [r['record'] for r in selfrec]}
    for i, (tag, texts) in enumerate(arms.items()):
        if time.time() - S.T0 > 60 * cfg['cap_minutes']: S.res['status'] = 'incomplete: cap'; S.dump(); return
        LM.load_lora_state(model, init); t = time.time(); trace, ntok = train_lora(S, tok, model, texts, cfg['seed'] + 10 + i, tag); S.res['timings']['train_' + tag] = time.time() - t
        torch.save(LM.lora_state(model), os.path.join(S.out, 'lora_%s.pt' % tag)); S.res['arms'][tag] = {'train_trace': trace, 'train_tokens': ntok, 'n_texts': len(texts), 'lora_B_norm': LM.lora_norms(model)['B']}; evaluate(tag)
    A = S.res['arms']; base, orc, slf = A['base']['far3_mean'], A['oracle_abstraction']['far3_mean'], A['self_abstraction']['far3_mean']
    S.res['criteria'] = {'oracle_gain': orc - base, 'oracle_pass': orc - base >= 0.15 and orc >= 0.55 and A['oracle_abstraction']['plant3_mean'] > orc, 'self_gain': slf - base, 'self_pass': (orc - base >= 0.15 and orc >= 0.55) and (slf - base >= 0.5 * (orc - base)), 'episodes_only_gain': A['episodes_only']['far3_mean'] - base}
    S.res['status'] = 'complete'; S.res['timings']['total'] = time.time() - S.T0; S.dump(); S.log('done: %s' % S.res['criteria'])


if __name__ == '__main__':
    main()
