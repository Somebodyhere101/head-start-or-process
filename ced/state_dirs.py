"""Stage ST1 helper (PROTOCOL.md): fit two candidate state directions on Qwen3-1.7B at the final prompt position of one layer.
THINK = mean residual under the thinking chat template minus the letter-only template (the same world prompts; the templates differ only in the
empty think block the letter-only prompt carries), i.e. the state in which G1t showed the model computes the relation.
CR = mean residual on computation questions minus retrieval questions (the compute-retrieve axis of the user's paper, refit here at one layer).
RAND = a random unit direction. Saves results/ST1/dirs.pt with unit directions, the mean final-position residual norm, cosines, and, for each
direction at the injection dose, the mean KL of the base letter distribution on world prompts and the in-domain letter accuracy shift.
    python state_dirs.py [--layer 12] [--dose 0.2]"""
import argparse, json, os, sys, random, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'retrogenesis'))
import torch
import local_models as LM, rlens as RL, exp_abstraction as EA, exp_consequence as XC

CAPS = [('France', 'Paris'), ('Japan', 'Tokyo'), ('Egypt', 'Cairo'), ('Canada', 'Ottawa'), ('Peru', 'Lima'), ('Kenya', 'Nairobi'), ('Norway', 'Oslo'), ('Thailand', 'Bangkok'), ('Chile', 'Santiago'), ('Greece', 'Athens'), ('Cuba', 'Havana'), ('Iran', 'Tehran')]
AUTH = [('Hamlet', 'Shakespeare'), ('1984', 'Orwell'), ('Dune', 'Herbert'), ('Emma', 'Austen'), ('Ulysses', 'Joyce'), ('Dracula', 'Stoker'), ('Frankenstein', 'Shelley'), ('Beloved', 'Morrison'), ('Lolita', 'Nabokov'), ('Candide', 'Voltaire'), ('Faust', 'Goethe'), ('the Odyssey', 'Homer')]
ELEM = [('gold', 'Au'), ('iron', 'Fe'), ('sodium', 'Na'), ('lead', 'Pb'), ('silver', 'Ag'), ('tin', 'Sn'), ('copper', 'Cu'), ('potassium', 'K'), ('mercury', 'Hg'), ('tungsten', 'W'), ('helium', 'He'), ('neon', 'Ne')]
YEARS = [('the first moon landing happen', 1969), ('the Berlin Wall fall', 1989), ('the French Revolution begin', 1789), ('the Titanic sink', 1912), ('World War I begin', 1914), ('the Magna Carta get signed', 1215), ('the Wright brothers first fly', 1903), ('Columbus reach the Americas', 1492), ('the Berlin airlift begin', 1948), ('the euro appear as cash', 2002), ('the Chernobyl disaster happen', 1986), ('the Apollo 13 mission fly', 1970)]


def retrieval_prompts():
    p = ['What is the capital of %s? Answer with one word.' % c for c, _ in CAPS] + ['Who wrote %s? Answer with the surname only.' % b for b, _ in AUTH]
    p += ['What is the chemical symbol for %s? Answer with the symbol only.' % e for e, _ in ELEM] + ['In what year did %s? Answer with the year only.' % y for y, _ in YEARS]
    return p


def computation_prompts(seed=3):
    rng = random.Random(seed); p = []
    for _ in range(12): a, b = rng.randrange(12, 99), rng.randrange(12, 99); p.append('What is %d times %d? Answer with the number only.' % (a, b))
    for _ in range(12): a, b = rng.randrange(1000, 9999), rng.randrange(100, 999); p.append('What is %d minus %d? Answer with the number only.' % (a, b))
    for _ in range(12): a, b = rng.randrange(3, 40), rng.randrange(20, 90); p.append('If x plus %d equals %d, what is x? Answer with the number only.' % (a, b))
    for _ in range(12): h, m = rng.randrange(2, 12), rng.randrange(5, 59); p.append('How many minutes are in %d hours and %d minutes? Answer with the number only.' % (h, m))
    return p


def final_resid(tok, model, texts, thinking, layer, batch=8):
    outs = []
    for i in range(0, len(texts), batch):
        msgs = [[{'role': 'system', 'content': EA.SYSTEM}, {'role': 'user', 'content': p}] for p in texts[i:i + batch]]
        seqs = [list(tok(tok.apply_chat_template(m, add_generation_prompt=True, enable_thinking=thinking, tokenize=False), add_special_tokens=False)['input_ids']) for m in msgs]
        ids, mask = LM._left_pad(seqs, tok.pad_token_id)
        with torch.no_grad(): hs = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), output_hidden_states=True).hidden_states[layer + 1][:, -1, :].float()
        outs.append(hs.cpu())
    return torch.cat(outs)


def hook_last(model, layer, vec):
    """Add vec to the residual at the final position (left padding: the last column is the final prompt token of every row) at the output of block `layer`."""
    def h(module, inputs, output):
        o = output[0] if isinstance(output, tuple) else output
        if o.shape[1] == 1: return output
        o = o.clone(); o[:, -1, :] += vec.to(o.dtype)
        return (o,) + tuple(output[1:]) if isinstance(output, tuple) else o
    return model.model.layers[layer].register_forward_hook(h)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--layer', type=int, default=12); ap.add_argument('--dose', type=float, default=0.2); ap.add_argument('--seed', type=int, default=20260920); a = ap.parse_args()
    out = os.path.join('results', 'ST1'); os.makedirs(out, exist_ok=True); T0 = time.time()
    tok, model = RL.load_eager(LM.TARGET_DIR); tok.padding_side = 'left'; model.set_attn_implementation('sdpa'); d = model.config.hidden_size; L = a.layer
    XC.QS = ['interpret', 'pursue']
    pl = XC.experience_items('plant', 32, a.seed + 1, a.seed + 2, True); cl = XC.experience_items('clinic', 32, a.seed + 2, a.seed + 12, True)
    world = ['Current situation. ' + it['situation'] + '\n\n' + it['questions'][q] for it in pl + cl for q in ('interpret', 'pursue')]
    h_think = final_resid(tok, model, world, True, L); h_letter = final_resid(tok, model, world, False, L)
    h_comp = final_resid(tok, model, computation_prompts(), False, L); h_retr = final_resid(tok, model, retrieval_prompts(), False, L)
    mean_norm = float(h_letter.norm(dim=-1).mean())
    dirs = {'think': (h_think.mean(0) - h_letter.mean(0)), 'cr': (h_comp.mean(0) - h_retr.mean(0)), 'rand': torch.randn(d, generator=torch.Generator().manual_seed(a.seed + 77))}
    raw_norm = {k: float(v.norm()) for k, v in dirs.items()}; dirs = {k: v / v.norm() for k, v in dirs.items()}
    def half(hA, hB):
        u1 = (hA[0::2].mean(0) - hB[0::2].mean(0)); u2 = (hA[1::2].mean(0) - hB[1::2].mean(0)); return float(torch.dot(u1, u2) / (u1.norm() * u2.norm()))
    rel = {'think': half(h_think, h_letter), 'cr': half(h_comp, h_retr)}
    cos = {'think_cr': float(torch.dot(dirs['think'], dirs['cr'])), 'think_rand': float(torch.dot(dirs['think'], dirs['rand'])), 'cr_rand': float(torch.dot(dirs['cr'], dirs['rand']))}
    letters = [tok(l, add_special_tokens=False)['input_ids'][0] for l in 'ABCD']; probe = world[:64]
    with torch.no_grad(): lp0 = XC.letter_logprobs(tok, model, probe, letters, 8)
    keys = [it['key'][q][it['common']] for it in pl + cl for q in ('interpret', 'pursue')][:64]; acc0 = float(sum('ABCD'[int(k)] == y for k, y in zip(lp0.argmax(1), keys)) / len(keys))
    strength = {}
    for name in ('think', 'cr', 'rand'):
        for sign in (1, -1):
            hk = hook_last(model, L, (sign * a.dose * mean_norm * dirs[name]).cuda())
            try:
                with torch.no_grad(): lp = XC.letter_logprobs(tok, model, probe, letters, 8)
            finally: hk.remove()
            kl = float((lp0.exp() * (lp0 - lp)).sum(1).mean()); acc = float(sum('ABCD'[int(k)] == y for k, y in zip(lp.argmax(1), keys)) / len(keys))
            strength['%s%s' % (name, '+' if sign > 0 else '-')] = {'letter_kl': kl, 'acc': acc}
    res = {'layer': L, 'dose': a.dose, 'mean_norm': mean_norm, 'raw_norm': raw_norm, 'split_half_cos': rel, 'cos': cos, 'base_acc': acc0, 'strength': strength, 'n_world': len(world), 'seconds': time.time() - T0}
    torch.save({'dirs': dirs, 'mean_norm': mean_norm, 'layer': L}, os.path.join(out, 'dirs.pt')); json.dump(res, open(os.path.join(out, 'dirs.json'), 'w'), indent=1); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
