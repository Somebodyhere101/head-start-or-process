"""Stage R1 (PROTOCOL.md): the attention-allocation lens (R-space) and the prediction lens (P-space) on Qwen3-1.7B, compared with the
J-space (the user's wikitext Jacobian lens and a same-corpus refit with the jlens estimator).
R readout at layer l, position p: gradient of S(p) = total attention mass p receives from all later positions (all heads, all layers) w.r.t. the
residual at layer l, position p. P readout: gradient of the log-probability of the actual next token at p.   python rlens.py [--smoke]"""
import argparse, glob, hashlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'retrogenesis')); sys.path.insert(0, 'C:/Users/caabm/Documents/Codex/2026-09-06/co/work/jacobian-lens')
import torch
import local_models as LM

JLENS = os.environ.get('CED_JLENS') or 'C:/Users/caabm/Documents/Codex/2026-09-06/co/work/Qwen3-1.7B_jacobian_lens.pt'; K = 64; T = 128; SKIP = 16; N_POS = 8
CORPUS_FILE = os.environ.get('CED_CORPUS') or os.path.join(HERE, 'corpus_prompts.json')   # exported prompts; used when present so the corpus is identical across machines
CORPUS_ROOTS = ['C:/Users/caabm/Documents/Codex/2026-09-08/i-x20', 'C:/Users/caabm/Documents/Codex/2026-09-13']


def file_sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
def overlap(A, B):
    Qa, _ = torch.linalg.qr(A.float()); Qb, _ = torch.linalg.qr(B.float()); s = torch.linalg.svdvals(Qa.T @ Qb); return float((s ** 2).mean())
def top_right(M, k):
    U_, S, Vh = torch.linalg.svd(M.float(), full_matrices=False); return Vh[:k].T.contiguous(), ((S ** 2) / (S ** 2).sum())[:k].sum().item()
def span_of_rows(G, k):
    X = G.float(); X = X / (X.norm(dim=1, keepdim=True) + 1e-12); return top_right(X, k)


def corpus(tok, n_prompts, seed=20260914):
    if os.path.exists(CORPUS_FILE):
        D = json.load(open(CORPUS_FILE, encoding='utf-8')); assert len(D['prompts']) >= n_prompts, 'corpus file has %d prompts, %d requested' % (len(D['prompts']), n_prompts); return D['prompts'][:n_prompts], dict(D['stats'], source=CORPUS_FILE, n_prompts=n_prompts)
    files = sorted(f for root in CORPUS_ROOTS for f in glob.glob(os.path.join(root, '**', '*.md'), recursive=True) if os.path.getsize(f) > 8000 and 'node_modules' not in f)
    text = '\n\n'.join(open(f, encoding='utf-8', errors='replace').read() for f in files); ids = tok(text, add_special_tokens=False)['input_ids']
    chunks = [ids[i:i + T] for i in range(0, len(ids) - T, T)]; step = max(1, len(chunks) // n_prompts); sel = chunks[::step][:n_prompts]
    return [tok.decode(c) for c in sel], {'files': len(files), 'chars': len(text), 'tokens': len(ids), 'chunks': len(chunks)}


def load_eager(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path); model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='eager').cuda().eval()
    for p in model.parameters(): p.requires_grad_(False)
    return tok, model


def fit_RP(tok, model, prompts, log):
    """Per source layer l (block index 0..26 = hidden_states[l+1]): gradient rows for R and P at N_POS positions per prompt."""
    nl = len(model.model.layers); R = {l: [] for l in range(nl - 1)}; P = {l: [] for l in range(nl - 1)}; meta = []
    for i, text in enumerate(prompts):
        ids = tok(text, add_special_tokens=False)['input_ids'][:T]; x = torch.tensor([ids], device='cuda'); emb = model.get_input_embeddings()(x).detach().requires_grad_(True)
        out = model(inputs_embeds=emb, output_attentions=True, output_hidden_states=True); hs = [out.hidden_states[l + 1] for l in range(nl - 1)]
        n = len(ids); mask = torch.triu(torch.ones(n, n, device='cuda'), diagonal=1)                              # mask[p', p] = 1 iff p' > p
        S = sum((A[0].float() * mask.T.unsqueeze(0)).sum(dim=(0, 1)) for A in out.attentions)                       # S[p] = sum over heads, layers, later queries p' of A[p', p]
        lp = torch.log_softmax(out.logits[0].float(), -1); nxt = lp[torch.arange(n - 1), x[0, 1:]]                    # log-prob of the actual next token at each p
        positions = torch.linspace(SKIP, n - 2, N_POS).round().long().tolist()
        for j, p in enumerate(positions):
            gR = torch.autograd.grad(S[p], hs, retain_graph=True); gP = torch.autograd.grad(nxt[p], hs, retain_graph=(j < len(positions) - 1))
            for l in range(nl - 1): R[l].append(gR[l][0, p].detach().half().cpu()); P[l].append(gP[l][0, p].detach().half().cpu())
            meta.append((i, p))
        if i % 20 == 0: log('  R/P fit prompt %d/%d' % (i, len(prompts)))
    return {l: torch.stack(v) for l, v in R.items()}, {l: torch.stack(v) for l, v in P.items()}, meta


def refit_J(tok, model, prompts, budget_s, log):
    """Same-corpus J with the jlens estimator, as two half-fits (for split-half reliability), within a time budget."""
    import jlens; from jlens.fitting import jacobian_for_prompt
    lm = jlens.from_hf(model, tok); nl = len(model.model.layers); halves = [{}, {}]; counts = [0, 0]; t0 = time.time()
    for i, text in enumerate(prompts):
        if time.time() - t0 > budget_s: break
        Jp, _, _ = jacobian_for_prompt(lm, text, list(range(nl - 1)), dim_batch=32, max_seq_len=T); h = i % 2
        for l, J in Jp.items(): halves[h][l] = halves[h].get(l, 0) + J.float().cpu()
        counts[h] += 1
        if i % 8 == 0: log('  J refit prompt %d (%.0f s)' % (i, time.time() - t0))
    for h in (0, 1):
        for l in halves[h]: halves[h][l] = halves[h][l] / max(1, counts[h])
    full = {l: (halves[0][l] * counts[0] + halves[1][l] * counts[1]) / max(1, sum(counts)) for l in halves[0]} if halves[0] else {}
    return full, halves, counts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--n_prompts', type=int, default=160); ap.add_argument('--j_prompts', type=int, default=40); ap.add_argument('--j_budget_min', type=float, default=14.0); a = ap.parse_args()
    if a.smoke: a.n_prompts, a.j_prompts, a.j_budget_min = 6, 2, 2.0
    out = os.path.join('results', 'R1_smoke' if a.smoke else 'R1'); os.makedirs(out, exist_ok=True); T0 = time.time(); logf = open(os.path.join(out, 'log.txt'), 'a')
    def log(m): s = '[%6.0fs] %s' % (time.time() - T0, m); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    res = {'model': LM.TARGET_DIR, 'k': K, 'T': T, 'positions_per_prompt': N_POS, 'code': {f: file_sha(f) for f in ('rlens.py', 'PROTOCOL.md')}, 'jlens_wikitext': JLENS, 'status': 'running'}
    tok, model = load_eager(LM.TARGET_DIR); prompts, cstat = corpus(tok, a.n_prompts); res['corpus'] = dict(cstat, n_prompts=len(prompts)); log('model loaded (eager attention); corpus %s' % res['corpus'])
    t = time.time(); R, P, meta = fit_RP(tok, model, prompts, log); res['timings'] = {'RP_fit': time.time() - t}; log('R/P fit done: %d rows per layer (%.0f s)' % (R[0].shape[0], time.time() - t))
    torch.save({'R': R, 'P': P, 'meta': meta}, os.path.join(out, 'rp_rows.pt'))
    t = time.time(); Jloc, Jhalves, jcounts = refit_J(tok, model, prompts[:a.j_prompts], 60 * a.j_budget_min, log); res['timings']['J_refit'] = time.time() - t; res['J_refit_prompts'] = jcounts; log('J refit done on %s prompts (%.0f s)' % (jcounts, time.time() - t))
    Jw = torch.load(JLENS, map_location='cpu', weights_only=False)['J']
    nl = len(model.model.layers); rows = R[0].shape[0]; half = torch.arange(rows) % 2 == 0; table = {}
    for l in range(nl - 1):
        Rl, evR = span_of_rows(R[l], K); Pl, evP = span_of_rows(P[l], K); Jwl, evJw = top_right(Jw[l], K)
        entry = {'R_split_half': overlap(span_of_rows(R[l][half], K)[0], span_of_rows(R[l][~half], K)[0]), 'P_split_half': overlap(span_of_rows(P[l][half], K)[0], span_of_rows(P[l][~half], K)[0]),
                 'R_vs_Jwiki': overlap(Rl, Jwl), 'P_vs_Jwiki': overlap(Pl, Jwl), 'R_vs_P': overlap(Rl, Pl), 'R_explained': evR, 'P_explained': evP, 'Jwiki_explained': evJw}
        if Jloc:
            Jll, evJl = top_right(Jloc[l], K); entry.update({'R_vs_Jlocal': overlap(Rl, Jll), 'P_vs_Jlocal': overlap(Pl, Jll), 'Jwiki_vs_Jlocal': overlap(Jwl, Jll), 'Jlocal_explained': evJl,
                                                            'Jlocal_split_half': overlap(top_right(Jhalves[0][l], K)[0], top_right(Jhalves[1][l], K)[0]) if l in Jhalves[1] else None})
        table[l] = entry
    floor = K / Jw[0].shape[0]; res['random_floor'] = floor
    def reading(ov, rel):
        return 'inside' if ov >= 0.5 * rel else ('distinct' if ov <= 2 * floor else 'partial')
    res['readings'] = {l: {'R_vs_J': reading(table[l].get('R_vs_Jlocal', table[l]['R_vs_Jwiki']), table[l]['R_split_half']), 'P_vs_J': reading(table[l].get('P_vs_Jlocal', table[l]['P_vs_Jwiki']), table[l]['P_split_half']), 'R_vs_P': reading(table[l]['R_vs_P'], min(table[l]['R_split_half'], table[l]['P_split_half']))} for l in table}
    res['table'] = table; res['status'] = 'complete'; res['seconds'] = time.time() - T0; json.dump(res, open(os.path.join(out, 'results.json'), 'w'), indent=1, default=str)
    torch.save({'R_basis': {l: span_of_rows(R[l], K)[0].half() for l in R}, 'P_basis': {l: span_of_rows(P[l], K)[0].half() for l in P}, 'J_local': {l: v.half() for l, v in Jloc.items()}}, os.path.join(out, 'bases.pt'))
    log('layer  R_rel  P_rel  R-Jw  R-Jl  P-Jw  P-Jl  R-P   Jw-Jl  (floor %.3f)' % floor)
    for l in table:
        e = table[l]; log('%5d  %.2f   %.2f   %.2f  %s  %.2f  %s  %.2f  %s   %s' % (l, e['R_split_half'], e['P_split_half'], e['R_vs_Jwiki'], ('%.2f' % e['R_vs_Jlocal']) if 'R_vs_Jlocal' in e else '  - ', e['P_vs_Jwiki'], ('%.2f' % e['P_vs_Jlocal']) if 'P_vs_Jlocal' in e else '  - ', e['R_vs_P'], ('%.2f' % e['Jwiki_vs_Jlocal']) if 'Jwiki_vs_Jlocal' in e else '  - ', res['readings'][l]))
    log('done in %.0f s' % res['seconds'])


if __name__ == '__main__':
    main()
