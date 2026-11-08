import copy
import json
from dataclasses import replace
from random import Random
import unittest
from unittest.mock import patch

from joint_env import JointEpisode, random_scenarios, sim
from joint_model import delta_notes
from rule_alignment import (auxiliary_supply, phase_catalog, MissingEvidence,
                            require_training_ready, SANDBOX_PROFILE, sample_visible_menu)
from capture_records import validate, audit_records


class AlignmentTests(unittest.TestCase):
    def env(self, eid, options, seed=37):
        env = JointEpisode(random_scenarios(1, 901)[0], seed, assumptions=SANDBOX_PROFILE)
        env.ep.state = replace(env.ep.state, phase='EVENT', eventId=eid, seen=(eid,),
                               eventOptions=tuple(options), coins=1000)
        return env

    def test_wrong_answer_has_no_notes_stays_in_event_and_disables_it(self):
        for oid in (11401, 11402, 11404, 11405):
            env = self.env(114, (0, oid, 11403))
            action = next(a for a in env.actions if a.base.index == oid)
            self.assertEqual(sum(delta_notes(env, action)), 0)
            before = env.notes, env.ep.state.coins, env.ep.state.node
            env.step(action)
            self.assertEqual((env.notes, env.ep.state.coins, env.ep.state.node), before)
            self.assertEqual(env.phase, 'EVENT')
            self.assertNotIn(oid, {a.base.index for a in env.actions})
            self.assertIn(11403, {a.base.index for a in env.actions})
            branch = JointEpisode.from_public(json.loads(json.dumps(env.snapshot())), 2)
            self.assertNotIn(oid, {a.base.index for a in branch.actions})

    def test_12002_branches_and_feature_expectation(self):
        for draw, healing in ((.1, True), (.8, False)):
            env = self.env(120, (0, 12002))
            env.crossed = len(sim.ROUTE)
            env.ep.state = replace(env.ep.state, node=env.crossed+1, floor=20)
            action = next(a for a in env.actions if a.base.index == 12002)
            self.assertAlmostEqual(float(sum(delta_notes(env, action))), 2.5, places=5)
            before = sum(env.notes)
            with patch.object(env.event_rng, 'random', return_value=draw):
                env.step(action)
            self.assertEqual(env.healing_events, int(healing))
            self.assertEqual(sum(env.notes)-before, 0 if healing else 5)

    def test_cost_is_random_not_optimizer_selected_and_snapshot_reproducible(self):
        env = self.env(103, (0, 10301, 10302))
        env.notes = (10, 30) + (0,)*11
        snapshot = env.snapshot()
        a, b = [JointEpisode.from_public(snapshot, 67) for _ in range(2)]
        with patch.object(a.event_rng, 'choices', return_value=[90011]) as choose:
            a._consume(5)
        self.assertEqual(a.notes[:2], (5, 30))
        self.assertEqual(choose.call_count, 5)
        b._consume(5)
        c = JointEpisode.from_public(snapshot, 67); c._consume(5)
        self.assertEqual(b.notes, c.notes)
        self.assertEqual(sum(b.notes), 35)

    def test_menus_have_observed_structure_and_public_state_is_not_expanded(self):
        for eid in (107, 108):
            for seed in range(30):
                ids = sample_visible_menu(eid, Random(seed))
                self.assertEqual(len([n for n in ids if eid*100+1 <= n <= eid*100+7]), 2)
                self.assertIn(eid*100+8, ids)
                self.assertEqual(len(set(ids)), len(ids))
                self.assertEqual(eid*100+9 in ids, eid == 108)
        env = self.env(107, (0, 10702, 10705, 10708))
        env._sync()
        self.assertEqual(set(env.ep.state.eventOptions), {0, 10702, 10705, 10708})

    def test_supply_uses_one_phase_row_and_observed_inventory_replaces_it(self):
        catalogue = phase_catalog()
        self.assertEqual(len(catalogue['discs']), 104)
        self.assertEqual(sum(len(v) for v in catalogue['supply_groups'].values()), 1044)
        for disc_id, definition in catalogue['discs'].items():
            rows = catalogue['supply_groups'][str(definition['supply_group'])]
            for phase in range(9):
                selected = rows[str(max(int(p) for p in rows if int(p) <= phase))]
                expected = tuple(selected.get(str(n), 0) for n in range(90011, 90024))
                self.assertEqual(auxiliary_supply([{'disc_id': int(disc_id), 'phase': phase}]), expected)
        observed = [0]*13
        self.assertEqual(auxiliary_supply([{'disc_id': 211001, 'phase': 8}], observed), tuple(observed))
        with self.assertRaises(ValueError):
            auxiliary_supply([{'disc_id': 211001}])

    def test_training_and_implicit_sandbox_are_blocked(self):
        from train import generate, fit
        from evaluate import evaluate
        from build_handoff import build
        with self.assertRaises(MissingEvidence):
            require_training_ready()
        for call in (lambda: generate(None, 1, 1, 1), lambda: fit(None, None, None, 1, 1),
                     lambda: evaluate(None), lambda: build()):
            with self.assertRaises(MissingEvidence):
                call()
        with self.assertRaises(MissingEvidence):
            JointEpisode(random_scenarios(1, 901)[0], 3)


class CaptureTests(unittest.TestCase):
    def record(self):
        return {'schema':'joint_note_transition_v1', 'run_id':'synthetic-run', 'action_id':'synthetic-action',
                'sequence':1, 'event_type':'battle', 'operation_confirmed':True,
                'response_linked':True, 'confirmation_source':'verified_ui_change', 'intervening_note_actions':[],
                'before':{'notes':[0]*13, 'notes_complete':True},
                'after':{'notes':[0]*13, 'notes_complete':True}, 'note_infos':None,
                'conditions':{'client_version':'synthetic', 'server_region':'synthetic', 'tower_id':1,
                              'floor':1, 'main_discs':[{'disc_id':n, 'phase':0} for n in (211001,211002,211003)],
                              'auxiliary_discs':[{'disc_id':211001, 'phase':0}]}}

    def test_true_zero_is_valid_unknown_and_software_success_are_not(self):
        r = self.record(); self.assertEqual(validate(r), [])
        r['before']['notes'][0] = None
        self.assertIn('before_notes_unknown_or_invalid', validate(r))
        r = self.record(); r['operation_confirmed'] = False
        self.assertIn('unconfirmed_operation_confirmed', validate(r))
        self.assertEqual(validate({'event_type':'shop_observed'}), ['unsupported_schema_no_automatic_legacy_conversion'])
        self.assertEqual(validate(None), ['record_not_an_object'])

    def test_exclusion_infos_consistency_and_dedup_conflicts(self):
        r = self.record(); r['note_infos'] = [{'Tid':90011,'Qty':5}]
        self.assertIn('infos_disagree_with_observed_delta', validate(r))
        r = self.record(); r['training_eligible'] = False
        self.assertIn('explicitly_excluded_training_eligible', validate(r))
        r = self.record(); r['conditions']['main_discs'] = []
        self.assertIn('incomplete_main_discs', validate(r))
        r = self.record(); report = audit_records([r, copy.deepcopy(r)])
        self.assertEqual(report['accepted_unique_actions'], {'battle':1})
        changed = copy.deepcopy(r); changed['after']['notes'][0] = 1
        report = audit_records([r, changed])
        self.assertEqual(report['accepted_unique_actions'], {})
        self.assertEqual(report['conflicting_action_groups_excluded'], 1)
