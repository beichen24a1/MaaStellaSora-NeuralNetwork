import copy
from dataclasses import replace
import json
from random import Random
import unittest

from bootstrap import HERE
import numpy as np
from joint_env import JointEpisode, JointAction, catalog, random_scenarios
from joint_model import Baseline, JointCritic, encode_joint, play, delta_notes
from joint_env import note_progress


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
        a, b = JointEpisode(self.config, 31), JointEpisode(other, 31)
        a._random_notes(80, a.note_rng)
        b._random_notes(80, b.note_rng)
        self.assertEqual(a.notes, b.notes)
        before = a.notes
        a._sync(); a._sync()
        self.assertEqual(a.notes, before)

    def test_explicit_shop_budget_stock_and_branch(self):
        env = JointEpisode(self.config, 32)
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
        a, b = JointEpisode(self.config, 33), JointEpisode(self.config, 33)
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
        env = JointEpisode(self.config, 34)
        env.notes = (0,) * 13
        env.ep.state = replace(env.ep.state, phase="EVENT", eventId=103,
                               eventOptions=(0, 10301, 10302), seen=(103,))
        self.assertFalse(any(a.base and a.base.index == 10301 for a in env.actions))

    def test_zero_residual_reproduces_greedy_preferences(self):
        env = JointEpisode(self.config, 32)
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
