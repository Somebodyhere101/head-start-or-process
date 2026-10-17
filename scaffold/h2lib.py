"""SCAFFOLD core library: model loading and adapter merging (ced conventions), letter log-probs, per-prompt offset tables (output matching and
prior shift), learning-only residual hooks, and the frozen v1 exact update (term for term as ced/exp_state.consequence_pass_exact with
QS = ['interpret']) with hooks for offsets, removal vectors, gradient transforms and periodic evaluation. Imports ced and retrogenesis read-only."""
from __future__ import annotations
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); IX = os.path.dirname(HERE)
for p in (HERE, os.path.join(IX, 'retrogenesis')):
    if p not in sys.path: sys.path.insert(0, p)
import torch
import torch.nn.functional as F
import local_models as LM
import bworld as BW

SYSTEM = 'You are an analyst reviewing incident records. Answer multiple-choice questions with the letter only.'   # = ced/exp_abstraction.SYSTEM
V1 = dict(lr=5e-4, batch=4, eps=0.1, tau=2.0, mass_gamma=0.1, kl_beta=0.03, clip=1.0, rank=8, alpha=16)          # frozen recipe v1 (ced PROTOCOL 2026-09-18)
CED_RES = os.path.join(IX, 'ced', 'results')


def load():
    tok, model = LM.load_causal_lm(LM.TARGET_DIR); tok.padding_side = 'left'; return tok, model


def letters(tok): return [tok(l, add_special_tokens=False)['input_ids'][0] for l in 'ABCD']


class Enc:
    """Tokenised chat prompts, cached; batches built by fixed index lists so padding is identical wherever a batch recurs."""
    def __init__(self, tok, prompts):
        self.tok = tok; self.seqs = [LM.chat_ids(tok, [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': p}]) for p in prompts]
    def batch(self, idx):
        ids, mask = LM._left_pad([self.seqs[i] for i in idx], self.tok.pad_token_id); return ids.cuda(), mask.cuda()


class ResidHook:
    """Adds per-row vectors to the output of block `layer` at the final position (ced convention: block l = hidden_states[l+1]).
    vec is set per batch (rows x d) or None. Skips single-token (generation) steps."""
    def __init__(self, model, layer):
        self.vec = None; self.layer = layer; self.h = model.model.layers[layer].register_forward_hook(self._fn)
    def _fn(self, module, inputs, output):
        o = output[0] if isinstance(output, tuple) else output
        if self.vec is None or o.shape[1] == 1: return output
        o = o.clone(); o[:, -1, :] = o[:, -1, :] + self.vec.to(o.dtype)
        return (o,) + tuple(output[1:]) if isinstance(output, tuple) else o
    def remove(self): self.h.remove()


def letter_lp(model, enc, idx, lets, hook=None, table=None):
    """Full-vocabulary log-softmax at the last position, gathered at the four letters (as ced exp_consequence.letter_logprobs)."""
    ids, mask = enc.batch(idx)
    if hook is not None: hook.vec = None if table is None else table[idx].cuda()
    out = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1, :].float()
    if hook is not None: hook.vec = None
    return torch.log_softmax(out, -1)[:, lets]


@torch.no_grad()
def lp_table(model, enc, batches, lets, hook=None, table=None):
    T = torch.zeros(len(enc.seqs), 4)
    for idx in batches: T[idx] = letter_lp(model, enc, idx, lets, hook, table).cpu()
    return T


@torch.no_grad()
def resid_table(model, enc, batches, layer):
    """Final-position residual after block `layer` (hidden_states[layer+1]) for every prompt."""
    H = None
    for idx in batches:
        ids, mask = enc.batch(idx); hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, logits_to_keep=1).hidden_states[layer + 1][:, -1].float().cpu()
        if H is None: H = torch.zeros(len(enc.seqs), hs.shape[1])
        H[idx] = hs
    return H


def merge_adapter(model, path, seed=0):
    """Merge a saved ced LoRA (rank 8, alpha 16, q/v) into the base weights in place (as ced later_learning.py)."""
    LM.add_lora(model, V1['rank'], V1['alpha'], ('q_proj', 'v_proj'), seed=seed); LM.load_lora_state(model, torch.load(path))
    for _, m in list(LM.lora_modules(model)):
        with torch.no_grad(): m.base.weight += (m.B @ m.A * m.scale).to(m.base.weight.dtype)
    strip(model)


def strip(model):
    for name, mod in list(model.named_modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, LM.LoRALinear): setattr(mod, cn, child.base)


def add_phi(model, seed):
    LM.add_lora(model, V1['rank'], V1['alpha'], ('q_proj', 'v_proj'), seed=seed)
    return [p for _, m in LM.lora_modules(model) for n, p in m.named_parameters() if not n.startswith('base')]


def batches_of(n, b): return [list(range(i, min(n, i + b))) for i in range(0, n, b)]


def rewards(items, idx):
    return torch.tensor([[1.0 if L == items[i]['key'][items[i]['common']] else -1.0 for L in 'ABCD'] for i in idx])


@torch.no_grad()
def evaluate(model, enc, items, lets, off=None, hook=None, table=None, batch=8):
    lp = torch.zeros(len(items), 4)
    for idx in batches_of(len(items), batch): lp[idx] = letter_lp(model, enc, idx, lets, hook, table).cpu()
    if off is not None: lp = lp + off
    chosen = ['ABCD'[int(k)] for k in lp.argmax(1)]; bal, by = BW.balanced_acc(items, chosen)
    return {'bal': bal, 'by_type': by, 'acc': sum(c == it['answer'] for c, it in zip(chosen, items)) / len(items)}


def train_b(model, params, enc, items, lets, passes, off=None, hook=None, rm=None, grad_tf=None, on_eval=None, eval_every=96, cfg=V1, log=print, step_hooks=None, off_fn=None, on_batch=None):
    """Frozen v1 exact policy-gradient update on the interpret question (ced consequence_pass_exact, QS=['interpret']):
    lp = renorm(lp_raw / tau); pm = (1-eps) p + eps/4; adv = R - moving baseline; loss = -sum pm adv lp + beta KL-anchor + gamma mass.
    off: per-prompt 4-vector added to lp_raw (output matching or prior shift); rm: per-prompt residual vectors applied through `hook` during
    training only; grad_tf(params): in-place gradient transform before clipping (PC-H2). Fixed item order every pass."""
    bts = batches_of(len(items), cfg['batch']); opt = torch.optim.Adam(params, lr=cfg['lr']); base = 0.0; trace = []
    with torch.no_grad():
        lp0 = lp_table(model, enc, bts, lets, hook, rm)
        if off is not None: lp0 = lp0 + off
        q0 = torch.log_softmax(lp0, 1)                                    # DEVIATIONS D1: anchor on the renormalised four-letter distribution
    for p in params: p.requires_grad_(True)
    seen = 0
    total = passes * len(items)
    for ps in range(passes):
        for idx in bts:
            if on_batch is not None: on_batch(seen, total)                       # e.g. the fade schedule (sets the A scale before this batch)
            opt.zero_grad()
            lp_model = letter_lp(model, enc, idx, lets, hook, rm)               # the model's own (raw, full-vocabulary) letter log-probs
            cur_off = off_fn(idx) if off_fn is not None else (off[idx] if off is not None else None)
            lp_raw = lp_model + cur_off.cuda() if cur_off is not None else lp_model
            lp = lp_raw / cfg['tau'] - torch.logsumexp(lp_raw / cfg['tau'], 1, keepdim=True)
            R = rewards(items, idx).cuda(); p = lp.exp().detach(); p = torch.where(torch.isfinite(p), p, torch.full_like(p, 0.25)); pm = (1 - cfg['eps']) * p + cfg['eps'] / 4
            er = (pm * R).sum(1); adv = R - base; base = 0.9 * base + 0.1 * float(er.mean())
            loss = -((pm * adv) * lp).sum(1).mean()
            qt = torch.log_softmax(lp_raw, 1)                                 # D1: proper KL(q_t || q_0) over the four letters; mass term on the raw model output
            loss = loss + cfg['kl_beta'] * (qt.exp() * (qt - q0[idx].cuda())).sum(1).mean() + cfg['mass_gamma'] * (-torch.logsumexp(lp_model, 1)).mean()
            loss.backward()
            if grad_tf is not None: grad_tf(params)
            gn = float(torch.nn.utils.clip_grad_norm_(params, cfg['clip'], error_if_nonfinite=True))
            if step_hooks is not None: step_hooks[0]()
            opt.step()
            if step_hooks is not None: step_hooks[1]()
            seen += len(idx)
            if on_eval is not None and seen % eval_every == 0:
                for q in params: q.requires_grad_(False)
                r = on_eval(seen); r.update(seen=seen, grad_norm=gn, vind=float((p * (R > 0).float()).sum(1).mean())); trace.append(r)
                for q in params: q.requires_grad_(True)
    for p in params: p.requires_grad_(False)
    return trace


def phi_state(model): return LM.lora_state(model)


class ScaledDelta:
    """A held as a float32 weight delta on q/v (dW = B A * scale per module) over a bf16 snapshot of the base weights; set(c) writes
    W = W0 + c * dW. Used for the gradual-fade arm (Yang et al. 2026). Must be built on a clean (unmerged) model, before phi is added."""
    def __init__(self, model, path):
        st = torch.load(path); mods = dict(model.named_modules()); self.items = []
        for name, d in st.items():
            lin = mods[name]; dW = (d['B'].float() @ d['A'].float() * (V1['alpha'] / V1['rank'])).to(lin.weight.device)
            self.items.append((lin, lin.weight.detach().clone(), dW))
        self.c = None
    @torch.no_grad()
    def set(self, c):
        if self.c is not None and abs(c - self.c) < 1e-9: return
        for lin, W0, dW in self.items: lin.weight.copy_((W0.float() + c * dW).to(lin.weight.dtype))
        self.c = c


def phi_B_params(model): return [m.B for _, m in LM.lora_modules(model)]


def calib_basis(model, enc, items, lets, k=8):
    """Top-k right singular vectors of per-item gradients of -log p(correct letter) w.r.t. phi's B matrices (PC-H2 calibration)."""
    Bs = phi_B_params(model); rows = []
    for p in Bs: p.requires_grad_(True)
    for i in range(len(items)):
        for p in Bs: p.grad = None
        lp = letter_lp(model, enc, [i], lets); lp = lp - torch.logsumexp(lp, 1, keepdim=True)
        (-lp[0, 'ABCD'.index(items[i]['answer'])]).backward(); rows.append(torch.cat([p.grad.flatten().float() for p in Bs]).cpu())
    for p in Bs: p.requires_grad_(False); p.grad = None
    G = torch.stack(rows); return torch.linalg.svd(G, full_matrices=False)[2][:k].T.contiguous()       # (dim x k), orthonormal columns


def make_amplifier(model, U, factor=4.0):
    """post-step hook: amplify the component of the latest update of phi's B matrices inside span(U) by `factor` (process-only change)."""
    Bs = phi_B_params(model); Ug = U.cuda(); state = {'old': None}
    def before(): state['old'] = torch.cat([p.detach().flatten().float() for p in Bs])
    @torch.no_grad()
    def after():
        new = torch.cat([p.detach().flatten().float() for p in Bs]); d = new - state['old']; extra = (factor - 1.0) * (Ug @ (Ug.T @ d)); o = 0
        for p in Bs: n_ = p.numel(); p.add_(extra[o:o + n_].view_as(p).to(p.dtype)); o += n_
    return before, after


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True); json.dump(obj, open(path, 'w'), indent=1, default=str)
