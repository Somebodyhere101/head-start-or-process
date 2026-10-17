"""B-task world for SCAFFOLD (PREREGISTRATION.md section 4). Software-domain situations with four cues: time window and infrastructure (as in
ced/world.py), reporting desk, and a new work-order cue. Every B item carries all four phrases, so B_ind and B_ext differ only in the rule.
  B_ind ('bind'): one shared cause iff same desk AND same work order; time and infrastructure drawn independently (A's rule is uninformative).
  B_ext ('bext'): one shared cause iff same time AND same infrastructure AND same desk (ced 'and3'); work order random.
Items are built for the interpret question only, with both answer keys (option order seeded), as ced/exp_consequence.experience_items does.
Does not import or modify ced/world.py state; reads its software vocabulary and time phrases as constants."""
from __future__ import annotations
import os, random, sys
CED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ced')
if CED not in sys.path: sys.path.append(CED)
import world as CW          # read-only: DOMAINS, WINDOWS, SPREAD, DESKS

LETTERS = 'ABCD'
D = CW.DOMAINS['software']
WINDOWS, SPREAD = list(CW.WINDOWS), list(CW.SPREAD)
DESKS = list(CW.DESKS)
WORKORDERS = ['work order 4411', 'work order 2307', 'work order 8812', 'work order 1506', 'work order 6630', 'work order 9274']
# phrase-shift pools (disjoint words and connectives) for the form-of-B evaluation
DESKS2 = ['the weekend rota', 'the escalation line', 'the vendor portal', 'the status page', 'the pager queue', 'the help channel']
WORKORDERS2 = ['change ticket CHG-31', 'change ticket CHG-58', 'change ticket CHG-77', 'change ticket CHG-12', 'change ticket CHG-94', 'change ticket CHG-46']
INTERPRET_Q = 'What is the best explanation?'
INTERPRET_OPTS = ['A single shared cause affecting all three', 'Three separate causes', 'A recording error in the log', 'Too early to say']

RULES = {'bind': lambda c: c['desk'] and c['wo'], 'bext': lambda c: c['time'] and c['infra'] and c['desk'], 'bpara': lambda c: c['batch'] and c['city']}
RULE_CUES = {'bind': ('desk', 'wo'), 'bext': ('time', 'infra', 'desk'), 'bpara': ('batch', 'city')}
ALL_CUES = ('time', 'infra', 'desk', 'wo')
# Stage 3 (PREREGISTRATION_STAGE3.md): B_para. Sameness is semantic, never a repeated string: each line names the batch value and the supplier
# city in a different surface form (numeral / word / arithmetic; name / nickname / description). 'Different' items use three different values
# (cities), with the same one-of-each form pattern, so the pattern of forms carries no information and only identity does.
BATCH_FORMS = {12: ('12', 'twelve', '3 times 4'), 7: ('7', 'seven', '3 plus 4'), 30: ('30', 'thirty', '5 times 6'), 15: ('15', 'fifteen', '3 times 5'),
               24: ('24', 'twenty-four', '4 times 6'), 18: ('18', 'eighteen', '2 times 9'), 9: ('9', 'nine', '3 times 3'), 40: ('40', 'forty', '5 times 8')}
CITY_FORMS = {'NY': ('New York', 'the Big Apple', 'NYC'), 'LA': ('Los Angeles', 'the City of Angels', 'LA'), 'CHI': ('Chicago', 'the Windy City', 'Chi-Town'),
              'PAR': ('Paris', 'the City of Light', 'the French capital'), 'LON': ('London', 'the British capital', 'the home of Big Ben'),
              'ROM': ('Rome', 'the Eternal City', 'the Italian capital')}
PARA_CUES = ('time', 'infra', 'batch', 'city')


def para_situation(rng, cues):
    """B_para text: time and infrastructure as in ced (infra = repeated string, independent of the rule), plus batch and city cues in three
    different surface forms per item; no batch or city string repeats within an item in either condition."""
    units = rng.sample(D['units'], 3); anomalies = rng.sample(D['anomalies'], 3); infras = rng.sample(D['infras'], 3)
    infra_of = [infras[0]] * 3 if cues['infra'] else infras
    when = rng.choice(WINDOWS) if cues['time'] else rng.choice(SPREAD)
    bvals = [rng.choice(list(BATCH_FORMS))] * 3 if cues['batch'] else rng.sample(list(BATCH_FORMS), 3)
    cvals = [rng.choice(list(CITY_FORMS))] * 3 if cues['city'] else rng.sample(list(CITY_FORMS), 3)
    bform = [0, 1, 2]; rng.shuffle(bform); cform = [0, 1, 2]; rng.shuffle(cform)
    btxt = [BATCH_FORMS[bvals[i]][bform[i]] for i in range(3)]; ctxt = [CITY_FORMS[cvals[i]][cform[i]] for i in range(3)]
    assert len(set(btxt)) == 3 and len(set(ctxt)) == 3
    lines = ['%s (%s on %s, batch %s, supplied from %s) %s' % (units[i].capitalize(), D['carrier'], infra_of[i], btxt[i], ctxt[i], anomalies[i]) for i in range(3)]
    site = rng.choice(D['site'])
    return 'At %s, three anomalies were logged %s: %s; %s; and %s.' % (site, when, lines[0], lines[1], lines[2])


def situation(rng, cues, shift=False):
    units = rng.sample(D['units'], 3); anomalies = rng.sample(D['anomalies'], 3); infras = rng.sample(D['infras'], 3)
    infra_of = [infras[0]] * 3 if cues['infra'] else infras
    when = rng.choice(WINDOWS) if cues['time'] else rng.choice(SPREAD)
    dpool, wpool = (DESKS2, WORKORDERS2) if shift else (DESKS, WORKORDERS)
    desks = rng.sample(dpool, 3); desk_of = [desks[0]] * 3 if cues['desk'] else desks
    wos = rng.sample(wpool, 3); wo_of = [wos[0]] * 3 if cues['wo'] else wos
    dverb, wverb = ('escalated by', 'filed against') if shift else ('reported via', 'under')
    lines = ['%s (%s on %s, %s %s, %s %s) %s' % (units[i].capitalize(), D['carrier'], infra_of[i], dverb, desk_of[i], wverb, wo_of[i], anomalies[i]) for i in range(3)]
    site = rng.choice(D['site'])
    return 'At %s, three anomalies were logged %s: %s; %s; and %s.' % (site, when, lines[0], lines[1], lines[2])


def item_set(rule, n, seed, shift=False):
    """n items: half positives (every rule cue true), half near-misses (exactly one rule cue false, types cycled); non-rule cues Bernoulli(0.5).
    Each item: situation text, interpret question with seeded option order, answer key for both outcomes, cue dict, item type."""
    rng = random.Random(seed); rc = RULE_CUES[rule]; out = []
    for i in range(n):
        cues = {c: rng.random() < 0.5 for c in (PARA_CUES if rule == 'bpara' else ALL_CUES)}
        if i % 2 == 0:
            for c in rc: cues[c] = True
            typ = 'pos'
        else:
            off = rc[(i // 2) % len(rc)]
            for c in rc: cues[c] = (c != off)
            typ = 'nm_' + off
        common = bool(RULES[rule](cues)); assert common == (typ == 'pos')
        text = para_situation(rng, cues) if rule == 'bpara' else situation(rng, cues, shift)
        order = list(range(4)); rng.shuffle(order); opts = [INTERPRET_OPTS[k] for k in order]
        q = INTERPRET_Q + '\n' + '\n'.join('%s. %s' % (LETTERS[j], o) for j, o in enumerate(opts)) + '\n\nAnswer with the letter only.'
        key = {True: LETTERS[order.index(0)], False: LETTERS[order.index(1)]}
        out.append({'situation': text, 'question': q, 'common': common, 'type': typ, 'cues': cues, 'key': key, 'answer': key[common]})
    return out


def prompt(it): return 'Current situation. ' + it['situation'] + '\n\n' + it['question']


def balanced_acc(items, chosen):
    """mean(accuracy on positives, mean over near-miss types of accuracy); constant policy 0.5, single-cue <= 0.75, conjunction 1.0."""
    by = {}
    for it, ch in zip(items, chosen): by.setdefault(it['type'], []).append(ch == it['answer'])
    pos = sum(by['pos']) / len(by['pos']); nms = [sum(v) / len(v) for k, v in by.items() if k != 'pos']
    return 0.5 * pos + 0.5 * sum(nms) / len(nms), {k: sum(v) / len(v) for k, v in by.items()}


def check_independence(rule='bind', n=4000, seed=1):
    """Empirical check that A's rule (time AND infra) is independent of the B_ind label in the generator."""
    its = item_set(rule, n, seed); a = [it['cues']['time'] and it['cues']['infra'] for it in its]; b = [it['common'] for it in its]
    pa = sum(a) / n; pb_a = sum(x and y for x, y in zip(a, b)) / max(1, sum(a)); pb_na = sum((not x) and y for x, y in zip(a, b)) / max(1, n - sum(a))
    return {'P(A)': pa, 'P(B|A)': pb_a, 'P(B|not A)': pb_na}


if __name__ == '__main__':
    for r in ('bind', 'bext'):
        its = item_set(r, 8, 7); print(r, [it['type'] for it in its]); print(prompt(its[0])); print('answer', its[0]['answer'], its[0]['key']); print()
    print('independence B_ind:', check_independence('bind')); print('B_ext dependence:', check_independence('bext'))
    print(prompt(item_set('bind', 2, 3, shift=True)[0]))
