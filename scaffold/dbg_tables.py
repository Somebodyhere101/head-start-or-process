"""Debug: are the base / A letter log-prob tables finite on B prompts, and how large are the matching offsets?"""
import os, sys, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import h2lib as H, bworld as BW

train = BW.item_set('bext', 16, 9001); random.Random(9002).shuffle(train)
tok, model = H.load(); lets = H.letters(tok); enc = H.Enc(tok, [BW.prompt(it) for it in train]); bt = H.batches_of(16, 4)
lp0 = H.lp_table(model, enc, bt, lets)
del model; torch.cuda.empty_cache(); model = H.load()[1]; H.merge_adapter(model, os.path.join(H.CED_RES, 'ST2_20260919', 'lora_none.pt'))
lpA = H.lp_table(model, enc, bt, lets)
print('base finite', bool(torch.isfinite(lp0).all()), 'min', float(lp0.min()), 'max', float(lp0.max()))
print('A finite', bool(torch.isfinite(lpA).all()), 'min', float(lpA.min()), 'max', float(lpA.max()))
off = -(lpA - lp0); print('offset min/max', float(off.min()), float(off.max()))
print('A rows (first 4):', lpA[:4]); print('base rows:', lp0[:4])
