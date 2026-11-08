"""Two-return joint critic, with frozen beta transfer for ordinary decisions."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from bootstrap import ROOT, HERE
import numpy as np

from joint_env import JointAction, NOTE_IDS, RULES, note_progress
from note_state import encode_context
from maa_value_plugin.critic.environment import sim, legal_actions
from maa_value_plugin.critic.features import encode_actions, encode_state, FEATURE_DIM, STATE_DIM
from maa_value_plugin.critic.network import QNetwork

BETA = HERE / "models/ordinary_beta_20261105.npz"
FEATURE_VERSION = "joint_public_action_v2_greedy_residual"
GREEDY_PROGRESS_INDEX = FEATURE_DIM + 496 + 128 + 6


def source_identity():
    here = Path(__file__).resolve().parent
    paths = [here / name for name in ("joint_env.py", "joint_model.py", "note_state.py", "price_data.py", "rule_alignment.py", "data/phase_supply.json", "data/inputs.json", "data/note_system.json", "data/reference_scenarios.json")]
    paths.extend(ROOT / "maa_value_plugin/critic" / name
                 for name in ("features.py", "environment.py", "presets.py", "network.py", "vendor/stella_sim.py"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}


@lru_cache(maxsize=1)
def beta():
    return QNetwork.load(BETA, frozen=True)


def beta_choice(env, actions):
    bases = [a for a in actions if a.base is not None]
    if not bases:
        raise ValueError("No ordinary action")
    if len(bases) == 1:
        return bases[0]
    x = encode_actions(env.ep.state, [a.base for a in bases], env.preset)
    return bases[int(np.argmax(beta().predict(x)))]


def delta_notes(env, action):
    delta = np.zeros(13, dtype=np.float32)
    if action.kind == "NOTE":
        for note, quantity in env.shelf[action.index].contents:
            delta[note - 90011] += quantity
    elif action.kind == "SWAP":
        delta[action.donor - 90011] -= 1
        delta[action.target - 90011] += 1
    elif env.note_events and action.base and action.base.kind == "EVENT_OPTION":
        oid = action.base.index
        if 10701 <= oid <= 10707:
            delta[oid % 100 - 1] += 5
        elif 10801 <= oid <= 10807:
            delta[oid % 100 - 1] += 10
        elif oid in (10708, 12702, 12001, 12002, 10808, 11403):
            quantity = 10 if oid in (10808, 11403) else 2.5 if oid == 12002 else 5
            for note in env.allowed:
                delta[note - 90011] += quantity / len(env.allowed)
        elif oid in (10301, 10401):
            # Expected vector under the explicit count-weighted sandbox law;
            # not the actual server outcome, and not a deterministic donor.
            quantity = 5 if oid == 10301 else 10
            delta -= np.asarray(env.notes) * quantity / sum(env.notes)
    return delta


def encode_joint(env, actions=None):
    actions = env.actions if actions is None else tuple(actions)
    context = np.asarray(encode_context(env.note_state()), dtype=np.float32)
    shelf = np.zeros((8, 16), dtype=np.float32)
    for i, item in enumerate(env.shelf):
        shelf[i, :3] = (1., item.stock, item.price / 400)
        for note, quantity in item.contents:
            shelf[i, 3 + note - 90011] = quantity / 15
    ordinary = [a.base for a in actions if a.base is not None]
    if ordinary:
        ordinary_x = encode_actions(env.ep.state, ordinary, env.preset)
        ordinary_q = beta().predict(ordinary_x)
        lookup = {a: (x, float(q)) for a, x, q in zip(ordinary, ordinary_x, ordinary_q)}
    else:
        lookup = {}
    base_state = np.zeros(FEATURE_DIM, dtype=np.float32)
    base_state[:STATE_DIM] = encode_state(env.ep.state, env.preset)
    price_priors, rows, priors = {}, [], []
    before_progress = note_progress(env.notes, env.skills)
    for action in actions:
        if action.base is not None:
            ordinary_row, prior = lookup[action.base]
            price = action.base.price
        else:
            ordinary_row, prior = base_state, 0.
            price = env.shelf[action.index].price if action.kind == "NOTE" else 0
            if action.kind == "NOTE":
                if price not in price_priors:
                    from dataclasses import replace
                    after = replace(env.ep.state, coins=env.ep.state.coins - price)
                    legal = legal_actions(after)
                    price_priors[price] = float(np.max(beta().predict(encode_actions(after, legal, env.preset))))
                prior = price_priors[price]
        delta = delta_notes(env, action)
        after_counts = dict(zip(NOTE_IDS, np.asarray(env.notes) + delta))
        progress = note_progress(tuple(np.asarray(env.notes) + delta), env.skills) - before_progress
        custom = np.zeros(46, dtype=np.float32)
        custom[:7] = (action.kind == "NOTE", action.kind == "SWAP", action.kind == "FINISH",
                      action.kind == "BASE", price / 400, env.swaps_left / 15, progress / 5)
        custom[7:20] = delta / 15
        if action.kind == "SWAP":
            custom[20 + action.donor - 90011] = 1
            custom[33 + action.target - 90011] = 1
        rows.append(np.concatenate((ordinary_row, context, shelf.ravel(), custom, [prior / 20])))
        priors.append(prior)
    return np.asarray(rows, dtype=np.float32), np.asarray(priors, dtype=np.float32)


def greedy_adjust(env):
    actions = env.actions
    before = note_progress(env.notes, env.skills)
    candidates = []
    for action in actions:
        after = tuple(np.asarray(env.notes) + delta_notes(env, action))
        candidates.append(note_progress(after, env.skills) - before)
    return actions[int(np.argmax(candidates))]


class Baseline:
    def __init__(self, mode="greedy", weight=3.):
        self.mode, self.weight = mode, weight

    def __call__(self, env):
        if env.phase == "ADJUST":
            return greedy_adjust(env)
        if env.phase not in ("SHOP", "EVENT"):
            return beta_choice(env, env.actions)
        if self.mode == "beta_only":
            return beta_choice(env, env.actions)
        actions = env.actions
        _, priors = encode_joint(env, actions)
        before = note_progress(env.notes, env.skills)
        gains = [note_progress(tuple(np.asarray(env.notes) + delta_notes(env, a)), env.skills) - before
                 for a in actions]
        return actions[int(np.argmax(priors + self.weight * np.asarray(gains)))]


class JointCritic:
    def __init__(self, params, metadata):
        self.params, self.metadata = params, metadata

    def predict(self, x):
        h = np.asarray(x, dtype=np.float32)
        for i in range(2):
            z = h @ self.params[f"W{i}"] + self.params[f"b{i}"]
            h = z / (1 + np.exp(-np.clip(z, -50, 50)))
        return h @ self.params["W2"] + self.params["b2"]

    def choose(self, env, weight=None):
        if env.phase == "ADJUST":
            return greedy_adjust(env)
        if env.phase not in ("SHOP", "EVENT"):
            return beta_choice(env, env.actions)
        actions = env.actions
        return actions[int(np.argmax(self.scores(env, actions, weight)))]

    def scores(self, env, actions=None, weight=None):
        x, prior = encode_joint(env, actions)
        values = self.predict(x)
        weight = self.metadata["preference_weight"] if weight is None else weight
        scale = self.metadata.get("residual_scale", 1.)
        return (prior + weight * 5 * x[:, GREEDY_PROGRESS_INDEX]
                + scale * (20 * values[:, 0] + weight * 5 * values[:, 1]))

    __call__ = choose

    def save(self, path):
        metadata = {**self.metadata, "feature_version": FEATURE_VERSION, "rules_version": RULES["version"],
                    "beta_sha256": hashlib.sha256(BETA.read_bytes()).hexdigest(),
                    "source_sha256": source_identity(),
                    "parameter_sha256": parameter_hash(self.params)}
        np.savez_compressed(path, **self.params, metadata=np.array(json.dumps(metadata)))
        path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            params = {key: data[key].copy() for key in data.files if key != "metadata"}
        if metadata["feature_version"] != FEATURE_VERSION or metadata["rules_version"] != RULES["version"]:
            raise ValueError("Checkpoint contract mismatch")
        if metadata["parameter_sha256"] != parameter_hash(params):
            raise ValueError("Corrupt joint checkpoint")
        if metadata["beta_sha256"] != hashlib.sha256(BETA.read_bytes()).hexdigest():
            raise ValueError("Ordinary teacher changed")
        if metadata["source_sha256"] != source_identity():
            raise ValueError("Joint source/data changed; reproduce or retrain for the current contract")
        return cls(params, metadata)


def parameter_hash(params):
    digest = hashlib.sha256()
    for key in sorted(params):
        digest.update(key.encode()); digest.update(params[key].tobytes())
    return digest.hexdigest()


def play(env, policy):
    while not env.done:
        env.step(policy(env))
    return env.outcome()
