import copy
from dataclasses import replace
import json
from random import Random
import unittest

from bootstrap import HERE
import numpy as np
from joint_env import JointEpisode, JointAction, catalog, random_scenarios, remaining_discount_tiers
from maa_value_plugin.critic.environment import Episode, sim
from joint_model import Baseline, JointCritic, encode_joint, play, delta_notes
from joint_env import note_progress
from rule_alignment import SANDBOX_PROFILE

def sandbox_episode(*args, **kwargs):
    return JointEpisode(*args, **kwargs, assumptions=SANDBOX_PROFILE)


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = random_scenarios(1, 901)[0]

    def test_requirements_supply_mapping(self):
        self.assertEqual(len(self.config["groups"]), 6)
        self.assertTrue(all(group in catalog() for group in self.config["groups"]))
        self.assertEqual(len(self.config["supply"]), 13)
        self.assertGreater(sum(self.config["supply"]), 0)

    def test_drop_distribution_independent_of_objective(self):
        other = copy.deepcopy(self.config)
        other["groups"] = list(catalog())[:6]
        a, b = sandbox_episode(self.config, 31), sandbox_episode(other, 31)
        a._random_notes(80, a.note_rng)
        b._random_notes(80, b.note_rng)
        self.assertEqual(a.notes, b.notes)
        before = a.notes
        a._sync(); a._sync()
        self.assertEqual(a.notes, before)

    def test_explicit_shop_budget_stock_and_branch(self):
        env = sandbox_episode(self.config, 32)
        policy = Baseline()
        while env.phase != "SHOP":
            env.step(policy(env))
        self.assertEqual(sum(map(sum, env.ep.state.stock)) + sum(i.stock for i in env.shelf), 8)
        snapshot = json.loads(json.dumps(env.snapshot()))
        branches = [JointEpisode.from_public(snapshot, 71) for _ in range(2)]
        action = next(a for a in env.actions if a.kind == "NOTE")
        before_coins, before_total = env.ep.state.coins, sum(env.notes)
        item = env.shelf[action.index]
        for branch in branches:
            branch.step(action)
        self.assertEqual(branches[0].notes, branches[1].notes)
        self.assertEqual(branches[0].ep.state.coins, before_coins - item.price)
        self.assertEqual(sum(branches[0].notes), before_total + sum(q for _, q in item.contents))
        self.assertEqual(branches[0].shelf[action.index].stock, 0)
        self.assertEqual(sum(env.notes), before_total)  # Branching leaves root untouched.
        with self.assertRaises(ValueError):
            branches[0].step(action)
        x, prior = encode_joint(env)
        env.note_rng = Random(999); env.shelf_rng = Random(999); env.event_rng = Random(999)
        env.ep._rng = Random(999)
        xx, pp = encode_joint(env)
        np.testing.assert_array_equal(x, xx)
        np.testing.assert_array_equal(prior, pp)
        self.assertFalse(any("rng" in key or "latent" in key for key in snapshot))

    def test_full_run_and_swap_conservation(self):
        a, b = sandbox_episode(self.config, 33), sandbox_episode(self.config, 33)
        one, two = play(a, Baseline()), play(b, Baseline())
        self.assertEqual(one, two)
        self.assertEqual(one["core"], 6)
        self.assertLessEqual(one["ordinary"], a.preset.total)
        self.assertTrue(all(n >= 0 for n in one["notes"]))
        a.finished, a.swaps_left = False, 1
        before = sum(a.notes), a.ep.state.coins
        action = next(action for action in a.actions if action.kind == "SWAP")
        a.step(action)
        self.assertEqual((sum(a.notes), a.ep.state.coins), before)
        self.assertEqual(a.swaps_left, 0)
        self.assertFalse(any(action.kind == "SWAP" for action in a.actions))

    def test_event_cannot_spend_missing_notes(self):
        env = self.event(103, (0, 10301, 10302), notes=0)
        self.assertFalse(any(a.base and a.base.index == 10301 for a in env.actions))
        self.assertEqual({a.base.index for a in env.actions}, {0, 10302})
        env.step(next(a for a in env.actions if a.base.index == 10302))
        self.assertEqual(sum(env.notes), 0)

    def event(self, eid, options, notes=50, coins=1000):
        env = sandbox_episode(self.config, 34)
        env.notes = (notes,) + (0,)*12
        env.ep.state = replace(env.ep.state, phase="EVENT", eventId=eid,
                               eventOptions=tuple(options), seen=(eid,), coins=coins,
                               node=len(sim.ROUTE)+1, floor=20, source=None, offer=())
        env.crossed = len(sim.ROUTE)
        return env

    def test_every_exposed_note_event_executes_and_charges_exactly(self):
        for eid, ids, quantity in ((107, range(10701, 10709), 5),
                                  (108, range(10801, 10809), 10),
                                  (114, (11403,), 10)):
            for oid in ids:
                with self.subTest(option=oid):
                    env = self.event(eid, (0, oid))
                    snapshot = json.loads(json.dumps(env.snapshot()))
                    branch = JointEpisode.from_public(snapshot, 98)
                    action = next(a for a in branch.actions if a.base.index == oid)
                    branch.step(action)
                    self.assertEqual(sum(branch.notes), 50+quantity)
                    self.assertEqual(branch.ep.state.coins, 1000-sim.EVENT_DATA[eid][2][oid][0])
                    self.assertEqual(branch.phase, "ADJUST")
                    self.assertEqual(sum(env.notes), 50)
                    if eid in (107, 108) and oid % 100 <= 7:
                        self.assertEqual(branch.notes[oid % 100-1], env.notes[oid % 100-1]+quantity)

    def test_event_visibility_wallet_and_note_cost_boundaries(self):
        for notes in (0, 4, 5, 9, 10, 50):
            env = self.event(103, (0, 10301, 10302), notes)
            self.assertEqual({a.base.index for a in env.actions},
                             {0, 10302} | ({10301} if notes >= 5 else set()))
            if notes >= 5:
                env.step(next(a for a in env.actions if a.base.index == 10301))
                self.assertEqual(sum(env.notes), notes-5)
                self.assertEqual(env.ep.state.coins, 1150)
            env = self.event(104, (0, 10401, 10402, 10403), notes, 200)
            self.assertIn(10402, {a.base.index for a in env.actions})
            self.assertEqual(10401 in {a.base.index for a in env.actions}, notes >= 10)
            if notes >= 10:
                env.step(next(a for a in env.actions if a.base.index == 10401))
                self.assertEqual(sum(env.notes), notes-10)
            paid = self.event(104, (0, 10401, 10402, 10403), notes, 200)
            paid.step(next(a for a in paid.actions if a.base.index == 10402))
            self.assertEqual(sum(paid.notes), notes)
            self.assertEqual(paid.ep.state.coins, 0)
        for coins in (89, 90, 139, 140):
            env = self.event(108, (0, 10801, 10808), coins=coins)
            ids = {a.base.index for a in env.actions}
            self.assertEqual(ids, {0} | ({10808} if coins >= 90 else set()) |
                             ({10801} if coins >= 140 else set()))
            self.assertNotIn(10802, ids)  # Not visible: no fabricated action.
        env = self.event(105, (0, 10501, 10502, 10503), coins=99)
        self.assertEqual({a.base.index for a in env.actions}, {0, 10503})
        with self.assertRaises(ValueError):
            env.step(JointAction("BASE", base=sim.Action("EVENT_OPTION", 10501, 0, 105)))

    def test_joint_shelf_constraints_initial_and_restock(self):
        env = sandbox_episode(self.config, 32)
        policy = Baseline()
        generations = 0
        while not env.done:
            if env.phase == "SHOP":
                stock = env.ep.state.stock
                if sum(map(sum, stock)) + sum(i.stock for i in env.shelf) == 8:
                    half = sum(row[0] for row in stock)
                    discounted = sum(row[0]+row[1] for row in stock)
                    for item in env.shelf:
                        quantity = sum(q for _, q in item.contents)
                        discounted += item.price < {5: 90, 15: 400}[quantity]
                        half += item.price == {5: 45, 15: 200}[quantity]
                    self.assertEqual(discounted, 4)
                    self.assertLessEqual(half, 2)
                    self.assertLessEqual(sum(sum(q for _, q in item.contents) == 15 for item in env.shelf), 1)
                    generations += 1
                restock = next((a for a in env.actions if a.base and a.base.kind == "RESTOCK"), None)
                if restock:
                    env.step(restock)
                    continue
            env.step(policy(env))
        self.assertGreaterEqual(generations, 6)  # All initial shops plus refreshes.

    def test_discount_conditioning_and_frozen_kernel_unchanged(self):
        # Three 20%-off drinks reveal that at most one of the four discounted
        # positions can be half-price. Bayes gives P(H=1)=.189/(.189+4*.027).
        rng = Random(451)
        samples = [remaining_discount_tiers(((0, 3, 0), (0, 0, 0)), rng) for _ in range(3000)]
        self.assertTrue(all(len(t) == 5 and sum(x < 2 for x in t) == 1 for t in samples))
        observed = sum(0 in t for t in samples)/len(samples)
        self.assertAlmostEqual(observed, .189/(.189+4*.027), delta=.03)
        middle = [remaining_discount_tiers(((0, 3, 0), (0, 0, 0)), rng,
                                         middle_initial=True) for _ in range(3000)]
        self.assertAlmostEqual(sum(0 in t for t in middle)/len(middle),
                               .140625/(.140625+4*.015625), delta=.03)
        # The ordinary Episode still applies its original pruning.
        joint = self.event(103, (0, 10301, 10302))
        ordinary = Episode.from_public(joint.ep.state, 4, joint.preset)
        self.assertEqual({a.index for a in ordinary.actions}, {10301})
        self.assertEqual({a.base.index for a in joint.actions}, {0, 10301, 10302})

    def test_zero_residual_reproduces_greedy_preferences(self):
        env = sandbox_episode(self.config, 32)
        baseline = Baseline()
        while env.phase != "SHOP":
            env.step(baseline(env))
        x, prior = encode_joint(env)
        params = {"W0": np.zeros((x.shape[1], 1), np.float32), "b0": np.zeros(1, np.float32),
                  "W1": np.zeros((1, 1), np.float32), "b1": np.zeros(1, np.float32),
                  "W2": np.zeros((1, 2), np.float32), "b2": np.zeros(2, np.float32)}
        model = JointCritic(params, {"preference_weight": 3., "residual_scale": 1.})
        before = note_progress(env.notes, env.skills)
        gains = [note_progress(tuple(np.asarray(env.notes) + delta_notes(env, action)), env.skills) - before
                 for action in env.actions]
        np.testing.assert_allclose(model.scores(env), prior + 3 * np.asarray(gains), atol=2e-6)


if __name__ == "__main__":
    unittest.main()
