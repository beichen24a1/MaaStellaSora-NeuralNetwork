import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from bootstrap import HERE
from audit_ui_evidence import audit, known_levels, media_time, summarize_coverage
from capture_records import validate


class VideoEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.bundle = json.loads((HERE/'data/ui_evidence_20261109_run02.json').read_text(encoding='utf-8'))

    def test_observed_ledger_reconciles_without_becoming_training_data(self):
        result = audit(self.bundle)
        self.assertEqual((result['unique_inventory_intervals'], result['visible_net_delta']), (20,121))
        self.assertEqual(result['accepted_complete_training_transitions'], 0)
        self.assertEqual(result['media_checks']['original'], 'not_checked_media_not_supplied')
        self.assertAlmostEqual(result['displayed_rule_conflicts'][0]['displayed_coin_expectation'],55)
        self.assertAlmostEqual(result['displayed_rule_conflicts'][0]['frozen_r12_coin_expectation'],22700/99)
        self.assertEqual(validate(self.bundle), ['unsupported_schema_no_automatic_legacy_conversion'])
        self.bundle['training_eligible'] = True
        with self.assertRaisesRegex(ValueError, 'excluded'):
            audit(self.bundle)

    def test_unknown_unrelated_notes_do_not_prevent_a_known_skill_level(self):
        final = self.bundle['final_concerto']
        notes = self.bundle['snapshots'][-1]['notes']
        self.assertEqual(known_levels(notes, final['group_ids']), [1,1,2,1,2,1])
        self.assertIsNone(notes[8])
        notes[5] = None  # Skill damage is required by the first two groups.
        self.assertEqual(known_levels(notes, final['group_ids'])[:2], [None,None])

    def test_omitted_wait_has_a_different_original_time(self):
        self.assertEqual(media_time(self.bundle, 599), 599)
        self.assertEqual(media_time(self.bundle, 600), 16740)
        final = self.bundle['snapshots'][-1]
        self.assertEqual(final['original_seconds'], 16876)
        final['original_seconds'] = final['review_seconds']
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            audit(self.bundle)

    def test_repeated_and_overlapping_intervals_are_rejected(self):
        self.bundle['transitions'].insert(1, copy.deepcopy(self.bundle['transitions'][0]))
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            audit(self.bundle)
        self.bundle['transitions'][1]['id'] = 'different-label-same-action'
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            audit(self.bundle)

    def test_wrong_exchange_delta_and_wallet_are_rejected(self):
        exchange = next(t for t in self.bundle['transitions'] if t['kind']=='exchange')
        exchange['observed_delta'][0] = 0
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            audit(self.bundle)
        self.setUp()
        next(s for s in self.bundle['snapshots'] if s['id']=='s374.5')['coins'] = 1690
        with self.assertRaisesRegex(ValueError, 'Exchange'):
            audit(self.bundle)

    def test_ocr_price_concatenation_and_wrong_purchase_are_rejected(self):
        self.bundle['shelves'][0]['goods'][5]['price'] = 9045
        with self.assertRaisesRegex(ValueError, 'price'):
            audit(self.bundle)
        self.setUp()
        self.bundle['purchases'][0]['price'] = 90
        with self.assertRaisesRegex(ValueError, 'shelf item'):
            audit(self.bundle)

    def test_ledger_rejects_an_unaccounted_gap(self):
        self.bundle['transitions'].pop(2)
        with self.assertRaisesRegex(ValueError, 'reconcile|Unaccounted'):
            audit(self.bundle)

    def test_documented_cli_works_without_test_import_path_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(HERE/'audit_ui_evidence.py'),
                str(HERE/'data/ui_evidence_20261109_run02.json'), '--output', str(Path(directory)/'audit.json')],
                cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((Path(directory)/'audit.json').read_text(encoding='utf-8'))
            self.assertEqual(report['media_checks']['original'], 'not_checked_media_not_supplied')

    def test_more_towers_cover_categories_without_merging_states(self):
        extras = [json.loads((HERE/f'data/ui_evidence_20261110_{name}.json').read_text(encoding='utf-8'))
                  for name in ('light','fire')]
        result = summarize_coverage([self.bundle]+extras)
        self.assertEqual(result['observed_category_union'], list(range(90011,90024)))
        self.assertEqual(result['runs_with_all_13_observed'], 0)
        self.assertFalse(result['inventories_merged'])
        self.assertFalse(result['training_approved'])
        self.assertEqual(extras[0]['snapshots'][-1]['notes'][7],7)  # Off-tower water is actually visible.
        self.assertIsNone(extras[1]['snapshots'][-1]['notes'][7])  # Never backfilled into another run.
        for bundle in extras:
            self.assertTrue(audit(bundle)['final_concerto_ui_match'])
        with self.assertRaisesRegex(ValueError,'Repeated run'):
            summarize_coverage([extras[0],extras[0]])

    def test_conditional_transfer_is_only_a_requirement_calculation(self):
        b=json.loads((HERE/'data/ui_evidence_20261110_light.json').read_text(encoding='utf-8'))
        example=b['conditional_transfer_example']
        self.assertFalse(example['applied_in_game'])
        self.assertFalse(example['entry_and_materials_verified'])
        self.assertEqual(known_levels(example['after_notes'],b['final_concerto']['group_ids']),example['expected_levels'])
        self.assertEqual(sum(example['expected_levels'])-sum(b['final_concerto']['displayed_levels']),3)
        self.assertEqual(example['before_notes'],b['snapshots'][-1]['notes'])
        for i,(before,after) in enumerate(zip(example['before_notes'],example['after_notes'])):
            if i not in (1,7):
                self.assertEqual(before,after)  # Includes every unknown component.


if __name__ == '__main__':
    unittest.main()
