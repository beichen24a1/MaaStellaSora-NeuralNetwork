"""Stream a local JSONL/ZIP; publish aggregate structural evidence, never rows."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

NOTE_IDS = set(range(90011, 90024))
NUMERIC_NOTE = re.compile(rb'(?<=[\[:,])\s*"?900(?:1[1-9]|2[0-3])"?\s*(?=[,}\]:])')


def profile(stream):
    digest, records, byte_count = hashlib.sha256(), 0, 0
    keys, kinds, vectors = Counter(), Counter(), Counter()
    rows_with_ids, note_entries, complete_maps = 0, 0, 0
    excluded = Counter()
    def walk(value):
        nonlocal note_entries, complete_maps
        if isinstance(value, dict):
            ids = {int(k) for k in value if str(k).isdigit()} & NOTE_IDS
            if ids == NOTE_IDS and all(type(value.get(str(n), value.get(n))) is int
                                      and value.get(str(n), value.get(n)) >= 0 for n in NOTE_IDS):
                complete_maps += 1
            tid = value.get('Tid', value.get('tid'))
            qty = value.get('Qty', value.get('qty'))
            if type(tid) is int and tid in NOTE_IDS and type(qty) is int:
                note_entries += 1
            for key, item in value.items():
                if re.search(r'note|lucky|inventory|infos|before|after', str(key), re.I):
                    # Only field names, never player identifiers or record contents.
                    keys[str(key)] += 1
                if isinstance(item, list) and len(item) == 13 and 'note' in str(key).lower():
                    vectors['complete_numeric' if all(type(n) is int and n >= 0 for n in item)
                            else 'incomplete_or_nonnumeric'] += 1
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    for line in stream:
        records += 1; byte_count += len(line); digest.update(line)
        row = json.loads(line)
        kinds[row.get('event_type', 'unknown')] += 1
        rows_with_ids += bool(NUMERIC_NOTE.search(line))
        for flag in ('training_eligible', 'probability_eligible', 'operation_confirmed', 'sequence_complete'):
            if row.get(flag) is False:
                excluded[flag] += 1
        walk(row)
    return {'records': records, 'bytes': byte_count, 'sha256': digest.hexdigest(),
            'records_by_event_type': dict(kinds), 'explicit_false_flags': dict(excluded),
            'rows_with_standalone_numeric_note_ids': rows_with_ids,
            'structured_note_tid_qty_entries': note_entries,
            'complete_13_id_numeric_maps': complete_maps,
            'note_named_13_element_lists': dict(vectors),
            'relevant_field_name_counts': dict(keys),
            'scope': 'Structural scan, not semantic verification. Text/menu/shop mentions are not holdings. '
                     'A nonzero candidate count still requires source/completeness/action-link review.',
            'usable_training_transitions': None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.source.suffix == '.zip':
        with ZipFile(args.source) as archive:
            names = [n for n in archive.namelist() if n.endswith('/data/observations.jsonl')]
            if len(names) != 1:
                raise ValueError('Expected exactly one canonical observations file')
            with archive.open(names[0]) as stream:
                result = profile(stream)
        result['archive_sha256'] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    else:
        with args.source.open('rb') as stream:
            result = profile(stream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
