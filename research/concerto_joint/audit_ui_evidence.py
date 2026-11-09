"""Audit manually reviewed video evidence, without promoting it to training data.

Video intervals are observations, not necessarily isolated game actions. Unknown
notes remain null. This deliberately has no converter to joint_note_transition_v1.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from bootstrap import ROOT  # Add repository root for direct script invocation.
from note_state import NOTE_IDS, load_catalog
from maa_value_plugin.critic.environment import sim

SCHEMA = 'joint_ui_evidence_v1'


def media_time(bundle, seconds):
    for row in bundle['media']['review_segments']:
        if row['review_start'] <= seconds < row['review_end']:
            return row['original_start'] + seconds - row['review_start']
    raise ValueError('Sample falls outside the documented review segments')


def partial_vector(value):
    if (not isinstance(value, list) or len(value) != 13
            or any(n is not None and (type(n) is not int or n < 0) for n in value)):
        raise ValueError('Require thirteen entries; unknown entries must be null')
    return value


def known_levels(notes, groups):
    """Compute a skill only when ALL of its requirements are observed.

    This does not fill the unrelated unknown inventory entries with zero and
    does not manufacture a complete NoteState.
    """
    counts = {n: q for n, q in zip(NOTE_IDS, partial_vector(notes)) if q is not None}
    catalog = load_catalog()
    levels = []
    for group in groups:
        skill = catalog[group]
        required = {n for row in skill.requirements for n, _q in row}
        levels.append(skill.level(counts) if required <= counts.keys() else None)
    return levels


def audit(bundle, media_directory=None):
    if bundle.get('schema') != SCHEMA:
        raise ValueError('Unsupported evidence schema')
    if bundle.get('training_eligible') is not False or bundle.get('probability_eligible') is not False:
        raise ValueError('UI evidence must remain excluded from training and probability estimation')
    media_checks = {}
    for key in ('original', 'review'):
        info = bundle['media'][key]
        if len(info['sha256']) != 64 or info['bytes'] <= 0:
            raise ValueError('Missing source identity')
        if media_directory is not None:
            path = Path(media_directory) / info['file']
            hasher = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    hasher.update(block)
            digest = hasher.hexdigest()
            if path.stat().st_size != info['bytes'] or digest != info['sha256']:
                raise ValueError('Video identity mismatch: '+key)
            media_checks[key] = 'sha256_and_size_match'
        else:
            media_checks[key] = 'not_checked_media_not_supplied'
    states = {}
    for s in bundle['snapshots']:
        if s['id'] in states:
            raise ValueError('Duplicate snapshot ID')
        partial_vector(s['notes'])
        t = s['review_seconds']
        if not isinstance(t, (int, float)) or not math.isfinite(t):
            raise ValueError('Invalid timestamp')
        if abs(media_time(bundle, t)-s['original_seconds']) > .001:
            raise ValueError('Wrong source/review timestamp mapping')
        if s.get('notes_complete') is not False:
            raise ValueError('This evidence bundle must not claim complete inventory')
        for view in s.get('supporting_views', []):
            # These timestamps locate additional stable views of the SAME state;
            # they never import counts from a different run.
            if not 0 <= view['original_seconds'] < bundle['media']['original']['duration_seconds']:
                raise ValueError('Supporting view outside source video')
        states[s['id']] = s
    delta_sum, row_counts, ids, rows = [0]*13, Counter(), set(), []
    previous_end = -1
    for t in bundle['transitions']:
        if t['id'] in ids:
            raise ValueError('Repeated interval is not an independent observation')
        ids.add(t['id'])
        before, after = states[t['before']], states[t['after']]
        if not previous_end <= before['review_seconds'] < after['review_seconds']:
            raise ValueError('Unordered or overlapping intervals')
        previous_end = after['review_seconds']
        delta = [None if b is None or a is None else a-b
                 for b, a in zip(before['notes'], after['notes'])]
        if delta != t['observed_delta']:
            raise ValueError('Annotation disagrees with observed inventory: '+t['id'])
        if t.get('training_eligible') is not False:
            raise ValueError('An interval must not opt into training')
        if t['kind'] == 'exchange':
            if (t['displayed_note_cost'] != 5 or sum(n for n in delta if n is not None) != -5
                    or any(n > 0 for n in delta if n is not None)
                    or after['coins']-before['coins'] != 150):
                raise ValueError('Exchange does not reconcile')
        if t['kind'] == 'event_purchase':
            wallet_before = states[t['wallet_before_snapshot']]['coins']
            wallet_after = states[t['wallet_after_snapshot']]['coins']
            if (wallet_before-wallet_after != t['displayed_coin_cost']
                    or sum(n for n in delta if n is not None) != t['displayed_note_reward']
                    or any(n < 0 for n in delta if n is not None)):
                raise ValueError('Event purchase does not reconcile')
        for i, n in enumerate(delta):
            if n is not None:
                delta_sum[i] += n
        row_counts[t['kind']] += 1
        rows.append({'id': t['id'], 'start': before['original_seconds'], 'end': after['original_seconds'],
                     'kind': t['kind'], 'known_delta_sum': sum(n for n in delta if n is not None),
                     'description': t['description'], 'isolated': t['isolated_note_action']})
    first, last = bundle['snapshots'][0], bundle['snapshots'][-1]
    observed = [i for i, (b, a) in enumerate(zip(first['notes'], last['notes'])) if b is not None and a is not None]
    if any(delta_sum[i] != last['notes'][i]-first['notes'][i] for i in observed):
        raise ValueError('Intervals do not reconcile the visible start/end inventory')
    # Check continuity across intentionally unannotated waits/ordinary decisions.
    for a, b in zip(bundle['transitions'], bundle['transitions'][1:]):
        if states[a['after']]['notes'] != states[b['before']]['notes']:
            raise ValueError('Unaccounted visible inventory change between intervals')
    shelf_prices = Counter()
    for shelf in bundle['shelves']:
        goods = shelf['goods']
        if len(goods) != 8 or len({g['slot'] for g in goods}) != 8:
            raise ValueError('A complete shelf needs eight distinct slots')
        for g in goods:
            allowed = (100, 160, 200) if g['kind'] == 'drink' else {5: (45,72,90), 15: (200,320,400)}[g['quantity']]
            if g['price'] not in allowed:
                raise ValueError('Illegal observed shelf price')
            if g['kind'] == 'note':
                shelf_prices[g['price']] += 1
        discounted = [g for g in goods if g['price'] < (200 if g['kind']=='drink' else {5:90,15:400}[g['quantity']])]
        if len(discounted) != 4:
            raise ValueError('Observed shelf does not have four discounts')
        if sum(g['price'] == (100 if g['kind']=='drink' else {5:45,15:200}[g['quantity']]) for g in goods) > 2:
            raise ValueError('Observed shelf has more than two half-price slots')
        if sum(g['quantity'] == 15 for g in goods if g['kind'] == 'note') > 1:
            raise ValueError('More than one large bundle on a shelf')
        media_time(bundle, shelf['review_seconds'])
    for purchase in bundle['purchases']:
        shelf = next(s for s in bundle['shelves'] if s['id'] == purchase['shelf_id'])
        good = next(g for g in shelf['goods'] if g['slot'] == purchase['slot'])
        if any(good[k] != purchase[k] for k in ('note_id','quantity','price')):
            raise ValueError('Purchase does not match the observed shelf item')
        if purchase['coins_before']-purchase['coins_after'] != purchase['price']:
            raise ValueError('Purchase wallet delta disagrees with actual price')
        if purchase['stock_before'] != 1 or purchase['stock_after'] != 0:
            raise ValueError('Purchase is not confirmed by a stock change')
        for t in purchase['review_seconds']:
            media_time(bundle, t)
    for menu in bundle['menus']:
        media_time(bundle, menu['review_seconds'])
    conflicts = []
    for menu in bundle['menus']:
        if menu['id'] != 'm-gamble':
            continue
        displayed = menu['options'][1]['coin_outcomes']
        frozen = sim.EVENT_DATA[105][2][10502][1]
        expected_ui = sum(p*q for p,q in displayed)
        expected_frozen = sum(n/d*q for n,d,q,_rarity,_scope in frozen)
        if abs(expected_ui-expected_frozen) > 1e-8:
            conflicts.append({'evidence':menu['id'], 'review_seconds':menu['review_seconds'],
                'legacy_option_mapping_hypothesis':10502,
                'displayed_coin_expectation':expected_ui,
                'frozen_r12_coin_expectation':expected_frozen,
                'status':'Client/version binding unresolved. Displayed law conflicts with the frozen table; not an empirical server-frequency estimate.'})
    final = bundle['final_concerto']
    computed = known_levels(last['notes'], final['group_ids'])
    if computed != final['displayed_levels']:
        raise ValueError('Final UI concerto levels disagree with static requirements')
    return {'schema': 'joint_ui_evidence_audit_v1', 'run_id': bundle['run_id'],
            'media_checks': media_checks, 'snapshots': len(states),
            'unique_inventory_intervals': len(ids), 'interval_kinds': dict(row_counts),
            'visible_note_ids': [NOTE_IDS[i] for i in observed],
            'unknown_note_ids': [NOTE_IDS[i] for i in range(13) if i not in observed],
            'visible_initial_total': sum(first['notes'][i] for i in observed),
            'visible_final_total': sum(last['notes'][i] for i in observed),
            'visible_net_delta': sum(delta_sum[i] for i in observed),
            'inventory_reconciliation': 'passed_on_observed_components_only',
            'complete_shelves': len(bundle['shelves']), 'shelf_note_prices': dict(shelf_prices),
            'wallet_and_stock_verified_note_purchases': len(bundle['purchases']),
            'displayed_rule_conflicts': conflicts,
            'final_concerto_computed': computed, 'final_concerto_ui_match': True,
            'accepted_complete_training_transitions': 0,
            'training_eligible': False, 'probability_eligible': False,
            'limitations': bundle['limitations'], 'intervals': rows}


def summarize_coverage(bundles):
    """Report category coverage across runs; never merge their inventories."""
    seen, union, runs = set(), set(), []
    for bundle in bundles:
        if bundle['run_id'] in seen:
            raise ValueError('Repeated run does not add independent coverage')
        seen.add(bundle['run_id'])
        report = audit(bundle)
        union.update(report['visible_note_ids'])
        runs.append({'run_id':bundle['run_id'], 'visible_note_ids':report['visible_note_ids'],
                     'unknown_note_ids':report['unknown_note_ids'],
                     'inventory_intervals':report['unique_inventory_intervals']})
    return {'schema':'joint_ui_category_coverage_v1','runs':runs,
            'observed_category_union':sorted(union),
            'unobserved_categories':sorted(set(NOTE_IDS)-union),
            'runs_with_all_13_observed':sum(not r['unknown_note_ids'] for r in runs),
            'inventories_merged':False, 'training_approved':False,
            'scope':'Category coverage only. Different runs are never one state or one transition.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--csv', type=Path)
    p.add_argument('--media-directory', type=Path)
    args = p.parse_args()
    report = audit(json.loads(args.source.read_text(encoding='utf-8')), args.media_directory)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if args.csv:
        with args.csv.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report['intervals'][0]))
            writer.writeheader(); writer.writerows(report['intervals'])
    print(json.dumps({k:v for k,v in report.items() if k not in ('intervals','limitations')}, ensure_ascii=True))
