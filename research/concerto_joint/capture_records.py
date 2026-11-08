"""Validate future action-linked note captures; never infer holdings from OCR prose."""
import argparse
from collections import Counter
import json
from pathlib import Path

SCHEMA = 'joint_note_transition_v1'
NOTE_IDS = tuple(range(90011, 90024))


def vector(value):
    if isinstance(value, dict) and set(value) == set(map(str, NOTE_IDS)):
        value = [value[str(n)] for n in NOTE_IDS]
    if not isinstance(value, list) or len(value) != 13 or any(type(n) is not int or n < 0 for n in value):
        raise ValueError('thirteen observed nonnegative counts required; null is unknown, not zero')
    return value


def validate(record):
    errors = []
    if not isinstance(record, dict):
        return ['record_not_an_object']
    if record.get('schema') != SCHEMA:
        return ['unsupported_schema_no_automatic_legacy_conversion']
    for key in ('run_id', 'action_id'):
        if not isinstance(record.get(key), str) or not record[key]:
            errors.append('missing_'+key)
    if type(record.get('sequence')) is not int or record['sequence'] < 0:
        errors.append('missing_action_sequence')
    if record.get('event_type') not in ('battle', 'exchange', 'event_reward', 'purchase'):
        errors.append('unknown_event_type')
    for key in ('operation_confirmed', 'response_linked'):
        if record.get(key) is not True:
            errors.append('unconfirmed_'+key)
    if record.get('confirmation_source') not in ('server_response', 'verified_ui_change'):
        errors.append('software_success_is_not_confirmation')
    if record.get('intervening_note_actions') != []:
        errors.append('not_an_isolated_transition')
    for flag in ('training_eligible', 'probability_eligible', 'sequence_complete', 'source_complete', 'sequence_inference_eligible'):
        if record.get(flag) is False:
            errors.append('explicitly_excluded_'+flag)
    if record.get('invalid_fields') or record.get('recognition_status') == 'invalid':
        errors.append('invalid_observation_fields')
    counts = {}
    for side in ('before', 'after'):
        state = record.get(side) or {}
        if not isinstance(state, dict):
            errors.append(side+'_state_invalid'); continue
        if state.get('notes_complete') is not True:
            errors.append(side+'_notes_incomplete')
        try:
            counts[side] = vector(state.get('notes'))
        except ValueError:
            errors.append(side+'_notes_unknown_or_invalid')
    if len(counts) == 2:
        delta = [a-b for a,b in zip(counts['after'], counts['before'])]
        entries = record.get('note_infos')
        if entries is not None:
            actual = [0]*13
            if not isinstance(entries, list):
                errors.append('note_infos_not_array')
            else:
                for entry in entries:
                    if (not isinstance(entry, dict) or type(entry.get('Tid')) is not int
                            or entry['Tid'] not in NOTE_IDS or type(entry.get('Qty')) is not int):
                        errors.append('invalid_signed_note_info'); continue
                    actual[entry['Tid']-90011] += entry['Qty']
                if actual != delta:
                    errors.append('infos_disagree_with_observed_delta')
        if record.get('event_type') == 'exchange':
            cost = {10301: 5, 10401: 10}.get(record.get('option_id'))
            if cost is None or sum(delta) != -cost or any(n > 0 for n in delta):
                errors.append('exchange_result_not_verified')
    conditions = record.get('conditions') or {}
    if not isinstance(conditions, dict):
        return sorted(set(errors+['invalid_conditions']))
    for key in ('client_version', 'server_region', 'tower_id', 'floor', 'main_discs', 'auxiliary_discs'):
        if conditions.get(key) is None:
            errors.append('missing_condition_'+key)
    for field in ('main_discs', 'auxiliary_discs'):
        discs = conditions.get(field)
        if not isinstance(discs, list) or len(discs) > 3 or (field == 'main_discs' and len(discs) != 3):
            errors.append('incomplete_'+field); continue
        for disc in discs:
            if (not isinstance(disc, dict) or type(disc.get('disc_id')) is not int or disc['disc_id'] <= 0
                    or type(disc.get('phase')) is not int or not 0 <= disc['phase'] <= 8):
                errors.append('missing_'+field+'_id_or_phase')
    return sorted(set(errors))


def audit_records(records):
    groups, reasons, total = {}, Counter(), 0
    for record in records:
        total += 1
        errors = validate(record)
        if errors:
            reasons.update(errors); continue
        identity = (record['run_id'], record['action_id'])
        groups.setdefault(identity, []).append(record)
    accepted, repeats, conflicts = Counter(), 0, 0
    for rows in groups.values():
        # Equal observations of the same action are one sample; contradictory
        # observations invalidate that action instead of keeping whichever is last.
        signatures = {json.dumps({k: row.get(k) for k in ('event_type', 'option_id', 'sequence', 'before', 'after', 'note_infos', 'conditions')}, sort_keys=True)
                      for row in rows}
        repeats += len(rows)-1
        if len(signatures) != 1:
            conflicts += 1
        else:
            accepted[rows[0]['event_type']] += 1
    return {'schema': 'joint_note_capture_audit_v1', 'records': total,
            'accepted_unique_actions': dict(accepted), 'reobservations_not_independent_samples': repeats,
            'conflicting_action_groups_excluded': conflicts, 'rejection_reason_counts': dict(reasons),
            'scope': 'Capture completeness only; this does not identify a stochastic kernel or approve training.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.source.open(encoding='utf-8') as stream:
        report = audit_records(json.loads(line) for line in stream if line.strip())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
