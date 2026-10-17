"""SCAFFOLD control adapters (PREREGISTRATION.md section 5, Stage 2; ASSESSMENT 5.3b item 6), trained with the unchanged ced v1 recipe
(exp_state.consequence_pass_exact on interleaved plant + clinic experience items, interpret + pursue, 3 passes, LoRA seed = run seed):
  shuf   - outcomes permuted across situations (rule-free consequences, same items and update)          -> results/controls/shuf_<seed>.pt
  single - outcomes under the infrastructure-only rule (another experience-acquired abstraction)        -> results/controls/single_<seed>.pt
  perm   - A's own LoRA factors with rows of B and columns of A permuted (norm-matched, function destroyed) -> results/controls/perm_<seed>.pt
Each control is evaluated on fresh ced software interpret items (meaning accuracy) and for its normalised letter KL to base, beside A.
   python train_controls.py --kind shuf --seed 20260919        (kind single sets CED_RULE=infra before ced imports)"""
import argparse, json, os, random, sys, time
ap = argparse.ArgumentParser(); ap.add_argument('--kind', required=True, choices=['shuf', 'single', 'perm']); ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--model', default='q17', choices=['q17', 'q06'], help='Stage 3: q06 = Qwen3-0.6B (A adapters ST2s_<seed>, outputs prefixed q06)'); a = ap.parse_args()
if a.kind == 'single': os.environ['CED_RULE'] = 'infra'
if a.model == 'q06': os.environ['CED_MODEL_DIR'] = os.path.expanduser(r'~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca')
HERE = os.path.dirname(os.path.abspath(__file__)); IX = os.path.dirname(HERE); CED = os.path.join(IX, 'ced')
sys.path.insert(0, HERE); sys.path.append(os.path.join(IX, 'retrogenesis'))
cwd = os.getcwd(); sys.path.insert(1, CED)
import torch
import h2lib as H, local_models as LM
import exp_consequence as XC, exp_state as XS, exp_abstraction as EA      # exp_consequence chdirs into ced/ at import
os.chdir(HERE)
import stage0 as S0
OUT = os.path.join(HERE, 'results', 'controls'); os.makedirs(OUT, exist_ok=True)


def evaluate(tok, model, lets, items, enc):
    LP = H.lp_table(model, enc, H.batches_of(len(items), 16), lets); test = list(range(len(items)))
    return S0.meaning_acc(items, LP, test), LP


def main():
    T0 = time.time(); tag = ('q06' if a.model == 'q06' else '') + '%s_%d' % (a.kind, a.seed); logf = open(os.path.join(OUT, tag + '.log'), 'a')
    def log(m): s = '[%6.0fs] %s' % (time.time() - T0, m); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    tok, model = H.load(); lets = H.letters(tok)
    import world as WD; train_rule = WD.RULE; WD.RULE = 'and'                     # evaluation items always follow A's rule ('and'), whatever the control was trained on
    ev = [it for it in S0.build(888000, 32) if it['dom'] == 'software']; enc = H.Enc(tok, [it['prompt'] for it in ev]); WD.RULE = train_rule
    acc0, LP0 = evaluate(tok, model, lets, ev, enc)
    A_path = os.path.join(H.CED_RES, ('ST2s_%d' if a.model == 'q06' else 'ST2_%d') % a.seed if a.seed != 20260920 else 'ST2b_20260920', 'lora_none.pt')
    if a.kind == 'perm':
        st = torch.load(A_path); g = torch.Generator().manual_seed(a.seed + 99); new = {}
        for k, d in st.items():
            pb = torch.randperm(d['B'].shape[0], generator=g); pa = torch.randperm(d['A'].shape[1], generator=g); new[k] = {'A': d['A'][:, pa].clone(), 'B': d['B'][pb].clone()}
        torch.save(new, os.path.join(OUT, tag + '.pt')); trace = None
    else:
        XC.QS = ['interpret', 'pursue']; cfg = dict(XC.CFG); cfg.update(seed=a.seed, n_per_domain=192, batch=4, eval_batch=8, lr=5e-4, passes=3, kl_beta=0.03, balanced=True, update='exact', mass_gamma=0.1, tau=2.0, eps=0.1, ckpt_every=0)
        n = cfg['n_per_domain']; pl = XC.experience_items('plant', n, a.seed + 1, a.seed + 2, True); cl = XC.experience_items('clinic', n, a.seed + 2, a.seed + 12, True)
        exp = [x for pair in zip(pl, cl) for x in pair]; outs = [it['outcome'] for it in exp]
        if a.kind == 'shuf': random.Random(a.seed + 77).shuffle(outs)
        log('%s: %d situations, common rate %.3f (rule %s)' % (tag, len(exp), sum(XC.is_common(o) for o in outs) / len(outs), os.environ.get('CED_RULE', 'and')))
        os.chdir(CED); S = EA.Stage(cfg, os.path.join(OUT, tag + '_stage')); os.chdir(HERE); torch.manual_seed(a.seed)      # ced's Stage hashes its own source files relative to ced/
        LM.add_lora(model, 8, 16, ('q_proj', 'v_proj'), seed=a.seed); params = [p for _, m in LM.lora_modules(model) for n_, p in m.named_parameters() if not n_.startswith('base')]
        trace = XS.consequence_pass_exact(S, tok, model, params, exp, outs, a.seed + 31, tag)
        torch.save(LM.lora_state(model), os.path.join(OUT, tag + '.pt')); H.strip(model); del model; torch.cuda.empty_cache(); model = H.load()[1]
    H.merge_adapter(model, os.path.join(OUT, tag + '.pt')); acc_c, LPc = evaluate(tok, model, lets, ev, enc)
    del model; torch.cuda.empty_cache(); model = H.load()[1]; H.merge_adapter(model, A_path); acc_a, LPa = evaluate(tok, model, lets, ev, enc)
    nkl = lambda P, Q: float((torch.log_softmax(P, 1).exp() * (torch.log_softmax(P, 1) - torch.log_softmax(Q, 1))).sum(1).mean())
    res = {'kind': a.kind, 'seed': a.seed, 'software_interpret_meaning_acc': {'base': acc0, 'control': acc_c, 'A': acc_a}, 'normalised_letter_kl_to_base': {'control': nkl(LPc, LP0), 'A': nkl(LPa, LP0)},
           'final_vindicated': trace[-1]['vindicated_recent'] if trace else None, 'seconds': time.time() - T0}
    H.save_json(res, os.path.join(OUT, tag + '.json')); log(json.dumps(res))


if __name__ == '__main__':
    main()
