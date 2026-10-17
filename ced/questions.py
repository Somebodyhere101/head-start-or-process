"""Five forced-choice questions per situation: four relevance decisions (recall, connect, interpret, pursue) and one surface control (site).
Each question has a structure-determined answer under the hidden rule; option order is seeded. Prompts share the same situation text so the
gradient at the situation's last token is comparable across questions."""
from __future__ import annotations
import random
import world as WD

LETTERS = 'ABCD'


def _mc(question, options, correct_idx, rng):
    order = list(range(4)); rng.shuffle(order); opts = [options[i] for i in order]; ans = LETTERS[order.index(correct_idx)]
    return {'question': question + '\n' + '\n'.join('%s. %s' % (LETTERS[i], o) for i, o in enumerate(opts)) + '\n\nAnswer with the letter only.', 'answer': ans, 'options': opts, 'structure_option': LETTERS[order.index(correct_idx)]}


def q_recall(sit, pool, rng):
    it = WD.recall_item(sit, pool, rng); q = it['prompt'].split('\n\n', 1)[1]          # drop the situation sentence; it is added by the prompt builder
    return {'question': q, 'answer': it['answer'], 'options': it['options'], 'structure_option': it['answer']}


def q_connect(sit, rng):
    D = WD.DOMAINS[sit['domain']]
    opts = ['They share %s' % sit['infra'], 'They happened close in time', 'One %s is responsible for all three' % D['carrier'], 'Nothing links them; each stands alone']
    return _mc('What most plausibly links these anomalies?', opts, 0 if sit['common'] else 3, rng)


def q_interpret(sit, rng):
    opts = ['A single shared cause affecting all three', 'Three separate causes', 'A recording error in the log', 'Too early to say']
    return _mc('What is the best explanation?', opts, 0 if sit['common'] else 1, rng)


def q_pursue(sit, rng):
    D = WD.DOMAINS[sit['domain']]
    opts = ['Inspect %s for a shared fault' % sit['infra'], 'Examine each %s separately' % D['carrier'], 'Wait for more anomalies', 'Audit the logging clock']
    return _mc('What should be investigated first?', opts, 0 if sit['common'] else 1, rng)


def q_surface(sit, rng):
    D = WD.DOMAINS[sit['domain']]; site = sit.get('site') or sit['text'].split('At ', 1)[1].split(',', 1)[0]; others = [s for s in D['site'] if s != site]      # 'site' is stored by world.situation since 2026-09-18 (the harbour frame has no 'At '); the parse is the fallback for older dicts
    opts = [site] + rng.sample(others, 3); return _mc('Which site logged these anomalies?', opts, 0, rng)


QUESTIONS = {'recall': None, 'connect': q_connect, 'interpret': q_interpret, 'pursue': q_pursue, 'surface': q_surface}


def build_items(seed=20260914, n=96, k_pool=64):
    """Per domain: n test situations x five questions, with a same-domain pool of prior cases for recall."""
    out = {}
    for dom in ('plant', 'clinic', 'software'):
        sits = WD.test_situations(dom, n, seed + {'plant': 3, 'clinic': 4, 'software': 5}[dom]); pool = WD.balanced_situations(dom, k_pool, seed + 10 + {'plant': 3, 'clinic': 4, 'software': 5}[dom])
        rng = random.Random(seed + 20); items = []
        for s in sits:
            qs = {'recall': q_recall(s, pool, rng), 'connect': q_connect(s, rng), 'interpret': q_interpret(s, rng), 'pursue': q_pursue(s, rng), 'surface': q_surface(s, rng)}
            items.append({'situation_id': s['id'], 'domain': dom, 'common': s['common'], 'situation': s['text'], 'questions': qs})
        out[dom] = items
    return out


if __name__ == '__main__':
    I = build_items(n=4, k_pool=16)
    it = I['clinic'][0]; print(it['situation']); print()
    for k, q in it['questions'].items(): print('--', k, 'answer', q['answer']); print(q['question']); print()
