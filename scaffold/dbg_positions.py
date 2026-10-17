"""Debug: decode the tokens at the option positions found by stage0.option_positions for a few prompts."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transformers import AutoTokenizer
import h2lib as H, stage0 as S0
tok = AutoTokenizer.from_pretrained(H.LM.TARGET_DIR)
items = S0.build(777000, 2)[:6]; P = S0.option_positions(tok, items)
for k, it in enumerate(items):
    ids = H.LM.chat_ids(tok, [{'role': 'system', 'content': H.SYSTEM}, {'role': 'user', 'content': it['prompt']}])
    print(it['rot'], 'optS ->', repr(tok.decode(ids[P['optS'][k] - 3:P['optS'][k] + 1])), '| optP ->', repr(tok.decode(ids[P['optP'][k] - 2:P['optP'][k] + 1])), '| next', repr(tok.decode(ids[P['optS'][k] + 1:P['optS'][k] + 2])))
