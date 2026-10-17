"""Local frozen-model runtime for Retrogenesis v0.1: model loading, hand-rolled LoRA, the goal-residual adapter (soft prefix),
the outcome head, batched chat generation with an optional soft prefix, the supervised inverse forward pass, a call ledger and
weight hashing. Only torch and transformers; no peft, no downloads."""
from __future__ import annotations
import hashlib, json, os, time
import torch, torch.nn as nn, torch.nn.functional as Fnn

HF = os.path.expanduser('~/.cache/huggingface/hub')
TARGET_DIR = os.environ.get('CED_MODEL_DIR') or os.path.join(HF, 'models--Qwen--Qwen3-1.7B', 'snapshots', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')   # CED_MODEL_DIR: any HF model id or local path
GENERATOR_DIR = os.path.join(HF, 'models--Qwen--Qwen3-0.6B', 'snapshots', 'c1899de289a04d12100db370d81485cdf75e47ca')
EOS_IDS = [151645, 151643]          # <|im_end|>, <|endoftext|>
PAD_ID = 151643
OUTCOME_DIM = 4                     # parse, complete keys, valid domains, fraction of constraints satisfied
BASE_PREFIX_TEXT = 'Strategy:'   # frozen token embeddings of this lead-in form the nonzero base prefix; the adapter adds a zero-initialized residual


def load_causal_lm(path, device='cuda'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = 'left'
    if tok.pad_token_id is None: tok.pad_token = tok.convert_ids_to_tokens(PAD_ID)
    try: model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa')
    except TypeError: model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model = model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return tok, model


def weights_sha256(model):
    """sha256 over every parameter's raw bytes in parameter order (bf16 viewed as int16)."""
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        t = p.detach().cpu().contiguous()
        h.update(name.encode()); h.update(t.view(torch.int16).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes())
    return h.hexdigest()


# ----- LoRA
class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x; A random (kaiming uniform), B zero, so the wrapped layer equals the base layer at initialization."""
    def __init__(self, base: nn.Linear, r=8, alpha=16, seed=0):
        super().__init__(); self.base = base; self.r, self.scale = r, alpha / r
        g = torch.Generator().manual_seed(seed)
        self.A = nn.Parameter((torch.rand(r, base.in_features, generator=g) * 2 - 1) * (1.0 / base.in_features ** 0.5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(self, x):
        return self.base(x) + (Fnn.linear(Fnn.linear(x.float(), self.A), self.B) * self.scale).to(x.dtype)


def add_lora(model, r=8, alpha=16, targets=('q_proj', 'v_proj'), seed=0):
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(mod, child_name, LoRALinear(child, r, alpha, seed + n).to(model.device)); n += 1
    return n


def lora_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def lora_state(model):
    return {n: {'A': m.A.detach().cpu().clone(), 'B': m.B.detach().cpu().clone()} for n, m in lora_modules(model)}


def load_lora_state(model, st):
    mods = dict(lora_modules(model))
    for n, d in st.items():
        with torch.no_grad(): mods[n].A.copy_(d['A'].to(mods[n].A.device)); mods[n].B.copy_(d['B'].to(mods[n].B.device))


def lora_norms(model):
    return {'A': float(sum(m.A.norm() ** 2 for _, m in lora_modules(model)) ** 0.5), 'B': float(sum(m.B.norm() ** 2 for _, m in lora_modules(model)) ** 0.5), 'n_modules': len(lora_modules(model))}


# ----- goal-residual adapter and outcome head
class GoalResidualAdapter(nn.Module):
    """concat(g, o, g - o) (12) -> 64 tanh -> n_prefix residual embeddings added to the base prefix; last layer zero-initialized (zero residual until trained)."""
    def __init__(self, hidden, n_prefix, seed=0):
        super().__init__(); self.hidden, self.n_prefix = hidden, n_prefix
        torch.manual_seed(seed); self.l1 = nn.Linear(3 * OUTCOME_DIM, 64); self.l2 = nn.Linear(64, n_prefix * hidden)
        nn.init.zeros_(self.l2.weight); nn.init.zeros_(self.l2.bias)
    def forward(self, g, o):
        x = torch.cat([g, o, g - o], -1).float(); return self.l2(torch.tanh(self.l1(x))).view(-1, self.n_prefix, self.hidden)


def base_prefix(tok, model):
    """Frozen input embeddings of BASE_PREFIX_TEXT (detached, model dtype): the nonzero base every arm shares."""
    ids = tok(BASE_PREFIX_TEXT, add_special_tokens=False)['input_ids']
    with torch.no_grad(): return model.get_input_embeddings()(torch.tensor(ids, device=model.device)).detach().clone(), len(ids)


def assert_finite(named_tensors, where):
    bad = [n for n, t in named_tensors if not torch.isfinite(t).all()]
    if bad: raise RuntimeError('non-finite values at %s: %s' % (where, bad[:8]))


def grad_report(groups):
    """groups: name -> list of parameters. Returns per-group gradient L2 norm and the count of non-finite gradient entries."""
    out = {}
    for name, ps in groups.items():
        gs = [p.grad for p in ps if p.grad is not None]
        out[name] = {'norm': float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs))) if gs else None, 'nonfinite': int(sum((~torch.isfinite(g)).sum() for g in gs)) if gs else 0, 'n_params_with_grad': len(gs)}
    return out


class OutcomeHead(nn.Module):
    """pooled generator features (mean last hidden state over directive tokens) -> 64 tanh -> 4 logits (diagnostic only in v0.1)."""
    def __init__(self, hidden, seed=0):
        super().__init__(); torch.manual_seed(seed); self.l1 = nn.Linear(hidden, 64); self.l2 = nn.Linear(64, OUTCOME_DIM)
    def forward(self, feats): return self.l2(torch.tanh(self.l1(feats.float())))


# ----- ledger
class Ledger:
    def __init__(self, path):
        self.path = path; self.rows = []
    def record(self, **row):
        row['t'] = time.time(); self.rows.append(row)
        with open(self.path, 'a') as f: f.write(json.dumps(row) + '\n')
    def totals(self, key='arm'):
        out = {}
        for r in self.rows:
            k = r.get(key, '?'); d = out.setdefault(k, {'calls': 0, 'prompt_tokens': 0, 'gen_tokens': 0, 'seconds': 0.0})
            d['calls'] += 1; d['prompt_tokens'] += r['prompt_tokens']; d['gen_tokens'] += r['gen_tokens']; d['seconds'] += r['seconds']
        return out


# ----- generation
def chat_ids(tok, messages):
    text = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    return list(tok(text, add_special_tokens=False)['input_ids'])


def _left_pad(seqs, pad):
    L = max(len(s) for s in seqs); ids = torch.full((len(seqs), L), pad, dtype=torch.long); mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs): ids[i, L - len(s):] = torch.tensor(s); mask[i, L - len(s):] = 1
    return ids, mask


def _gen_len(row):
    for j, t in enumerate(row.tolist()):
        if t in EOS_IDS: return j + 1
    return int(row.numel())


@torch.no_grad()
def generate(tok, model, messages_list, max_new_tokens, prefix=None, do_sample=False, temperature=1.0, top_p=1.0, seed=0, batch_size=8, ledger=None, meta=None, task_ids=None):
    """Batched chat generation. prefix: optional (N, n_prefix, hidden) soft embeddings placed after the prompt, immediately before the first
    generated token (attended, never padded; left padding stays on the left).
    Returns a list of dicts: text, prompt_tokens, gen_tokens, seconds (batch seconds shared by its rows)."""
    out = []; meta = meta or {}
    for b in range(0, len(messages_list), batch_size):
        chunk = messages_list[b:b + batch_size]; seqs = [chat_ids(tok, m) for m in chunk]; ids, mask = _left_pad(seqs, tok.pad_token_id)
        ids, mask = ids.to(model.device), mask.to(model.device); t0 = time.perf_counter()
        if do_sample: torch.manual_seed(seed + b); torch.cuda.manual_seed_all(seed + b)
        kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tok.pad_token_id, eos_token_id=EOS_IDS, attention_mask=mask)
        if do_sample: kw.update(temperature=temperature, top_p=top_p, top_k=0)
        else: kw.update(temperature=None, top_p=None, top_k=None)
        if prefix is not None:
            emb = model.get_input_embeddings()(ids); pre = prefix[b:b + len(chunk)].to(emb.dtype).to(emb.device)
            kw['attention_mask'] = torch.cat([mask, torch.ones(len(chunk), pre.shape[1], dtype=mask.dtype, device=mask.device)], 1)
            gen = model.generate(inputs_embeds=torch.cat([emb, pre], 1), **kw)     # with inputs_embeds only new tokens are returned
        else:
            gen = model.generate(input_ids=ids, **kw)[:, ids.shape[1]:]
        secs = time.perf_counter() - t0
        for i in range(len(chunk)):
            n = _gen_len(gen[i]); text = tok.decode(gen[i][:n], skip_special_tokens=True).strip()
            row = {'text': text, 'prompt_tokens': len(seqs[i]) + (0 if prefix is None else int(prefix.shape[1])), 'gen_tokens': n, 'seconds': secs / len(chunk), 'task_id': task_ids[b + i] if task_ids else None}; out.append(row)
            if ledger is not None: ledger.record(task_id=row['task_id'], prompt_tokens=row['prompt_tokens'], gen_tokens=n, seconds=row['seconds'], do_sample=do_sample, seed=(seed + b) if do_sample else None, max_new_tokens=max_new_tokens, **meta)
    return out


# ----- supervised inverse forward (one record)
def inverse_forward(tok, model, adapter, base, messages, directive, g, o):
    """Teacher-forced forward of prompt + prefix + directive, prefix = base (frozen token embeddings) + adapter(g, o). Returns (token-mean CE over
    the directive tokens and the closing <|im_end|>, pooled directive features (detached) for the outcome head, number of directive tokens)."""
    p_ids = chat_ids(tok, messages); d_ids = tok(directive, add_special_tokens=False)['input_ids'] + [EOS_IDS[0]]
    ids = torch.tensor([p_ids + d_ids], device=model.device); emb = model.get_input_embeddings()(ids)
    pre = (base.unsqueeze(0) + adapter(g.view(1, -1).to(model.device), o.view(1, -1).to(model.device))).to(emb.dtype); n_pre = pre.shape[1]
    x = torch.cat([emb[:, :len(p_ids)], pre, emb[:, len(p_ids):]], 1); out = model(inputs_embeds=x, output_hidden_states=True)
    start = len(p_ids) + n_pre; logits = out.logits[0, start - 1: start - 1 + len(d_ids)].float(); tgt = torch.tensor(d_ids, device=model.device)
    ce = Fnn.cross_entropy(logits, tgt); feats = out.hidden_states[-1][0, start: start + len(d_ids)].float().mean(0).detach()
    return ce, feats, len(d_ids)


@torch.no_grad()
def directive_features(tok, model, adapter, base, messages, directive, o):
    """Label-free features for the outcome head: a separate pass conditioned on the deployment goal g = ones (never the achieved outcome)."""
    _, feats, _ = inverse_forward(tok, model, adapter, base, messages, directive, torch.ones(OUTCOME_DIM), o); return feats
