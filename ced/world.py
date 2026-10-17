"""Three far domains sharing one hidden relation: anomalies share a common cause iff they began in the same time window AND their carriers
sit on the same shared infrastructure. Templated, seeded text; no vocabulary shared across domains. Episodes (domain A) reveal the outcome;
test items never do. Also builds the forced-choice recall items (structure-matched target, surface-matched foils)."""
from __future__ import annotations
import random

DOMAINS = {
    'plant': {'site': ['Northfield plant', 'Harbor works', 'Ridgeline refinery', 'Delta mill'], 'record': 'plant records', 'carrier': 'unit', 'infra': 'utility loop',
              'units': ['reactor', 'compressor', 'boiler', 'conveyor', 'centrifuge', 'condenser', 'kiln', 'extruder', 'scrubber', 'chiller'],
              'infras': ['the east steam loop', 'the west steam loop', 'the north power bus', 'the south power bus', 'the primary cooling circuit', 'the secondary cooling circuit'],
              'anomalies': ['pressure spiked', 'temperature drifted', 'vibration rose', 'flow dropped', 'an alarm tripped', 'output fell', 'noise increased', 'a valve stuck'],
              'common': ['a voltage sag on {infra}', 'contaminated feed through {infra}', 'a clock fault on {infra}', 'a leak in {infra}'],
              'separate': ['a worn bearing', 'a fouled sensor', 'an operator error', 'a clogged filter', 'a cracked seal', 'a loose coupling']},
    'clinic': {'site': ['St. Anne hospital', 'Riverside clinic', 'Lakeshore infirmary', 'Hillcrest ward'], 'record': 'case files', 'carrier': 'patient', 'infra': 'ward',
               'units': ['a farmer', 'a teacher', 'a nurse', 'a student', 'a mechanic', 'a retiree', 'a chef', 'a driver', 'a clerk', 'a musician'],
               'infras': ['ward 3', 'ward 5', 'the dialysis bay', 'the maternity wing', 'the outpatient hall', 'the surgical floor'],
               'anomalies': ['developed a fever', 'reported nausea', 'showed a rash', 'became dizzy', 'had chest pain', 'lost appetite', 'grew confused', 'coughed blood'],
               'common': ['a contaminated water line in {infra}', 'a faulty ventilation unit in {infra}', 'a mislabeled medication batch in {infra}', 'a mould outbreak in {infra}'],
               'separate': ['a viral infection', 'a food allergy', 'a medication side effect', 'dehydration', 'a chronic condition', 'a sprain']},
    'software': {'site': ['the Orion platform', 'the Vega backend', 'the Lyra cluster', 'the Atlas stack'], 'record': 'incident log', 'carrier': 'service', 'infra': 'dependency',
                 'units': ['the billing service', 'the search index', 'the login gateway', 'the report builder', 'the image resizer', 'the mailer', 'the cache layer', 'the scheduler', 'the export job', 'the chat relay'],
                 'infras': ['the config store', 'the certificate authority', 'the message broker', 'the DNS resolver', 'the metrics pipeline', 'the identity provider'],
                 'anomalies': ['returned errors', 'timed out', 'served stale data', 'crashed', 'leaked memory', 'dropped requests', 'logged warnings', 'stalled'],
                 'common': ['an expired credential in {infra}', 'a bad rollout of {infra}', 'a partition of {infra}', 'a quota exhausted on {infra}'],
                 'separate': ['a null pointer bug', 'a misconfigured timeout', 'a full disk', 'a bad deploy', 'a race condition', 'an off-by-one error']},
    # alien domain, written 2026-09-15 20:30 after every adapter in results/ had been trained; never used in development (stage A1)
    'fleet': {'site': ['the Westgate depot', 'the Millbrook hub', 'the Northern yard', 'the Eastside terminal'], 'record': 'fleet logs', 'carrier': 'vehicle', 'infra': 'supply point',
              'units': ['van 12', 'truck 7', 'the refrigerated lorry', 'the courier bike', 'the tanker', 'the shuttle coach', 'the tow rig', 'the crane truck', 'minibus 9', 'the flatbed'],
              'infras': ['the upper fuel station', 'the lower fuel station', 'the main charging dock', 'the auxiliary charging dock', 'the tyre contractor', 'the washing rig'],
              'anomalies': ['died on the ramp', 'went silent on the radio', 'ran hot', 'braked hard', 'strayed off route', 'sat on a soft tyre', 'missed a checkpoint', 'flagged a warning light'],
              'common': ['a tainted delivery from {infra}', 'a seized pump at {infra}', 'a rogue firmware push to {infra}', 'ice on {infra}'],
              'separate': ['a slipping clutch', 'crew fatigue', 'a flat battery', 'a bent mirror', 'a punctured tyre', 'a jammed door']},
    # second alien domain, written 2026-09-18 21:30 (stage O1b) after every adapter had been trained: its own vocabulary, its own frame sentence and its own
    # time phrasing (tides and watches, no minutes, hours, days or weeks), so a transferred rule cannot ride on the shared time words of the other four domains
    'harbor': {'site': ['the Port of Marrow', 'Kestrel Quay', 'the Saltwick basin', 'Greyhaven docks'], 'record': 'harbour log', 'carrier': 'vessel', 'infra': 'berth service',
               'units': ['the dredger', 'the pilot boat', 'the grain barge', 'the car ferry', 'the harbour tug', 'the stern trawler', 'the container feeder', 'the bunker lighter', 'the survey launch', 'the ice-class coaster'],
               'infras': ['the north mooring ring', 'the south mooring ring', 'the shore-power pontoon', 'the bunker line', 'the pilot station channel', 'the fresh-water bowser'],
               'anomalies': ['lost steerage', 'took on water', 'fouled a line', 'went dark on AIS', 'dragged anchor', 'blew a gasket', 'reported a fuel smell', 'sounded three short blasts'],
               'common': ['a contaminated bunker batch at {infra}', 'a power surge through {infra}', 'a chart datum error issued from {infra}', 'silting around {infra}'],
               'separate': ['a tired crew', 'a corroded fitting', 'a gnawed cable', 'an unsecured hatch', 'a misread signal', 'a worn propeller'],
               'windows': ['on the same tide', 'during one watch'], 'spread': ['over three separate tides', 'across a fortnight of sailings', 'on voyages a month apart'],
               'frame': 'Harbour master log, %s. Three entries %s: %s; %s; and %s.'}}
import os
RULES = {'infra': lambda st, si: si, 'time': lambda st, si: st, 'or': lambda st, si: st or si, 'and': lambda st, si: st and si, 'xor': lambda st, si: st != si}
RULE = os.environ.get('CED_RULE', 'and')
THIRD_CUE = os.environ.get('CED_THIRD_CUE') == '1'   # stage CR3d: an equally salient third cue (reporting desk) independent of the rule; RULES['td'] is the counterfactual rule over it
DESKS = ['the day desk', 'the night desk', 'the field radio', 'the mobile app', 'the control room', 'the paper log']
DESK_POOL = os.environ.get('CED_DESK_POOL') == '1'   # FORK-WINDOW: eight connectives for the desk phrase, drawn per line, so no single token carries the proxy; off by default (CR3d, D3 texts unchanged)
DESK_VERBS = ['reported via', 'logged through', 'called in by', 'flagged from', 'raised on', 'filed by', 'notified through', 'relayed via']
RULES['td'] = lambda st, si, sd=False: bool(st and sd)   # hidden relation over (same time window, same shared infrastructure); 'and' reproduces every run before 2026-09-14 20:00
RULES['and3'] = lambda st, si, sd=False: bool(st and si and sd)   # stage D3 (2026-09-18): the three-cue conjunction, the composition of 'and' (time, infra) and 'td' (time, desk)
THREE = RULE in ('td', 'and3')   # rules over the third cue; they require CED_THIRD_CUE=1 and are handled by the three-cue branches below, which leave every two-cue rule byte-identical
PROXY = False   # stage AL1 (2026-09-19): when True (set by exp_state.py --proxy while building the experience items only), the reporting desk is a perfect proxy for the rule: same desk iff the relation holds; evaluation items keep the desk random, so the two policies decouple there
WINDOWS = ['within the same ten minutes', 'in the same hour']
SPREAD = ['across three different days', 'spread over two weeks', 'a week apart']
if os.environ.get('CED_PHRASING') == 'distinct':   # stage D4 (2026-09-18 23:10): each training domain gets its own time phrasing, so the time cue cannot be learned as a shared phrase; default runs are untouched
    DOMAINS['plant'].update(windows=['during one shift change', 'before the same batch cleared'], spread=['over the course of a month', 'on three unconnected shifts', 'a fortnight between them'])
    DOMAINS['clinic'].update(windows=['on the same ward round', 'between one handover and the next'], spread=['on visits a season apart', 'across three admissions', 'with a long gap between them'])


def _common(st, si, sd): return bool(RULES[RULE](st, si, bool(sd))) if THREE else bool(RULES[RULE](st, si))


def situation(dom, rng, same_time, same_infra, same_desk=None):
    D = DOMAINS[dom]; units = rng.sample(D['units'], 3); anomalies = rng.sample(D['anomalies'], 3); infras = rng.sample(D['infras'], 3)
    infra_of = [infras[0]] * 3 if same_infra else infras
    when = rng.choice(D.get('windows', WINDOWS)) if same_time else rng.choice(D.get('spread', SPREAD))      # per-domain time phrasing (O1b); the four original domains keep the shared lists and are byte-identical
    desk_of = [''] * 3
    if THIRD_CUE:
        if same_desk is None: same_desk = bool(RULES[RULE](same_time, same_infra)) if (PROXY and not THREE) else (rng.random() < 0.5)
        desks = rng.sample(DESKS, 3); desk_of = [desks[0]] * 3 if same_desk else desks
    else: same_desk = None
    verbs = [rng.choice(DESK_VERBS) for _ in range(3)] if (THIRD_CUE and DESK_POOL) else ['reported via'] * 3
    lines = ['%s (%s on %s%s) %s' % (units[i].capitalize() if dom != 'clinic' else units[i], D['carrier'], infra_of[i], (', %s %s' % (verbs[i], desk_of[i])) if THIRD_CUE else '', anomalies[i]) for i in range(3)]
    site = rng.choice(D['site']); text = D.get('frame', 'At %s, three anomalies were logged %s: %s; %s; and %s.') % (site, when, lines[0], lines[1], lines[2])      # same single rng call as before, so the four original domains are unchanged
    return {'domain': dom, 'same_time': same_time, 'same_infra': same_infra, 'same_desk': same_desk, 'common': _common(same_time, same_infra, same_desk), 'text': text, 'units': units, 'anomalies': anomalies, 'infra': infra_of[0], 'site': site}


def outcome(sit, rng):
    D = DOMAINS[sit['domain']]
    if sit['common']:
        cause = rng.choice(D['common']).format(infra=sit['infra']); return 'Investigation outcome: all three traced to a single shared cause, %s. The anomalies were one event.' % cause
    causes = rng.sample(D['separate'], 3); return 'Investigation outcome: three separate causes, %s, %s and %s. The anomalies were unrelated.' % tuple(causes)


def episode(dom, rng, idx, same_time=None, same_infra=None, same_desk=None):
    st = rng.random() < 0.5 if same_time is None else same_time; si = rng.random() < 0.5 if same_infra is None else same_infra
    sit = situation(dom, rng, st, si, same_desk); return dict(sit, id='%s-%03d' % (dom[0].upper(), idx), outcome=outcome(sit, rng))


def episode_text(ep, out=None):
    return 'Case %s. %s %s' % (ep['id'], ep['text'], out if out is not None else ep['outcome'])


def balanced_situations(dom, n, seed):
    """n situations with the four cue combinations balanced (so common : separate = 1 : 3 by the rule) unless n is split explicitly."""
    rng = random.Random(seed); out = []
    for i in range(n):
        combo = i % 4; out.append(episode(dom, rng, i, same_time=combo in (0, 1), same_infra=combo in (0, 2)))
    return out


def test_situations(dom, n, seed):
    """n test situations, half common cause (same time and same infra), half separate with the near-miss cue patterns (one cue true, one false)."""
    rng = random.Random(seed); out = []
    if THREE:                                                                    # three-cue rules (D3): positives are every combination the rule accepts; near-misses have exactly one of the rule's cues false
        assert THIRD_CUE, 'CED_RULE=%s needs CED_THIRD_CUE=1' % RULE
        combos3 = [(a, b, c) for a in (True, False) for b in (True, False) for c in (True, False)]; pos = [c for c in combos3 if RULES[RULE](*c)]
        neg = [(False, True, True), (True, False, True), (True, True, False)] if RULE == 'and3' else [(False, None, True), (True, None, False)]
        for i in range(n):
            st, si, sd = pos[(i // 2) % len(pos)] if i % 2 == 0 else neg[(i // 2) % len(neg)]
            if si is None: si = rng.random() < 0.5
            out.append(episode(dom, rng, i, same_time=st, same_infra=si, same_desk=sd))
        return out
    combos = [(True, True), (False, False), (True, False), (False, True)]; pos = [c for c in combos if RULES[RULE](*c)]; neg = [c for c in combos if not RULES[RULE](*c)]
    if RULE == 'and': neg = [(True, False), (False, True)]                       # the original near-miss design
    for i in range(n):
        st, si = pos[(i // 2) % len(pos)] if i % 2 == 0 else neg[(i // 2) % len(neg)]
        out.append(episode(dom, rng, i, same_time=st, same_infra=si))
    return out


def surface_overlap(a, b):
    return len(set(a['units']) & set(b['units'])) + len(set(a['anomalies']) & set(b['anomalies']))


def recall_item(sit, pool, rng):
    """Four prior cases (cues only): one with the same hidden structure as `sit`, three with the other structure chosen to maximize surface overlap
    with `sit` among the pool (the foils look more similar on the surface than the target does)."""
    same = [p for p in pool if p['common'] == sit['common']]; other = [p for p in pool if p['common'] != sit['common']]
    target = min(same, key=lambda p: (surface_overlap(p, sit), rng.random())); foils = sorted(other, key=lambda p: (-surface_overlap(p, sit), rng.random()))[:3]
    opts = [target] + foils; rng.shuffle(opts); letters = 'ABCD'
    q = ('Current situation. %s\n\nWhich prior case from the %s is most relevant to the current situation?\n' % (sit['text'], DOMAINS[pool[0]['domain']]['record'])
         + '\n'.join('%s. Case %s: %s' % (letters[i], o['id'], o['text']) for i, o in enumerate(opts)) + '\n\nAnswer with the letter only.')
    return {'prompt': q, 'answer': letters[opts.index(target)], 'situation_id': sit['id'], 'common': sit['common'], 'options': [o['id'] for o in opts]}


def build(seed=20260914, n_train=192, n_pool=64, n_test=96):
    train = balanced_situations('plant', n_train, seed + 1)                       # experience episodes (with outcomes)
    pool = balanced_situations('plant', n_pool, seed + 2)                         # prior cases offered as recall options (cues only)
    testA = test_situations('plant', n_test, seed + 3); testB = test_situations('clinic', n_test, seed + 4); testC = test_situations('software', n_test, seed + 5)
    rng = random.Random(seed + 6)
    items = {k: [recall_item(s, pool, rng) for s in sits] for k, sits in (('A', testA), ('B', testB), ('C', testC))}
    return {'train': train, 'pool': pool, 'items': items}


if __name__ == '__main__':
    W = build(); print(episode_text(W['train'][0])); print(); print(episode_text(W['train'][1])); print(); print(W['items']['B'][0]['prompt']); print('answer', W['items']['B'][0]['answer'])
    print('train common fraction', sum(e['common'] for e in W['train']) / len(W['train']))
