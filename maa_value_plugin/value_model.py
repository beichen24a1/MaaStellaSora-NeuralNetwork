"""Portable NumPy action critic used by the Maa integration.

The fixed distribution checkpoint is loaded once per agent process.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from .critic.environment import PublicState, Action, legal_actions, sim
from .critic.features import FEATURE_VERSION
from .critic.network import MODEL_VERSION
from .critic.policy import QPolicy
from .critic.presets import Preset
from .deployment_preset import DeploymentPreset


class ValueModel:
    def __init__(self, plugin_dir=None, checkpoint=None):
        self.root = Path(plugin_dir or Path(__file__).resolve().parent)
        self.checkpoint_override = Path(checkpoint) if checkpoint else None
        self._load()

    def _load(self):
        selection = json.loads((self.root / "model.json").read_text(encoding="utf-8-sig"))
        checkpoint = self.checkpoint_override or Path(selection["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = self.root / checkpoint
        model_id = selection.get("model_id", checkpoint.stem)
        actor = QPolicy.load(checkpoint, model_id)
        checkpoint_rules = actor.model.metadata.get("rules_version")
        if checkpoint_rules and checkpoint_rules != sim.MODEL_VERSION:
            raise ValueError(f"模型规则 {checkpoint_rules} 与模块规则 {sim.MODEL_VERSION} 不一致。")
        self.actor = actor
        self.model_id = model_id
        self.checkpoint = checkpoint
        self.versions = {
            "feature_version": FEATURE_VERSION, "network_version": MODEL_VERSION,
            "rules_version": sim.MODEL_VERSION, "policy": "direct_argmax",
            "checkpoint_rules_version": checkpoint_rules,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "model_selection": selection,
        }

    @staticmethod
    def preset(value):
        if value is None or isinstance(value, Preset):
            return value
        if not isinstance(value, dict):
            value = {"name": "Maa preset", "pools": value}
        preset_type = DeploymentPreset if value.get("deployment_only", False) else Preset
        return preset_type.from_dict(value)

    @classmethod
    def initial(cls, preset):
        return sim.initial_from_pools(cls.preset(preset).pools)

    @staticmethod
    def state(value):
        return value if isinstance(value, PublicState) else sim.state_from_dict(value)

    @staticmethod
    def actions(state):
        return legal_actions(ValueModel.state(state))

    def values(self, state, actions=None, preset=None):
        state = self.state(state)
        actions = legal_actions(state) if actions is None else tuple(actions)
        return self.actor.values(state, actions, self.preset(preset))

    def choose(self, state, actions=None, preset=None):
        return self.actor.choose(self.state(state), actions, self.preset(preset))

    def decide(self, state, actions=None, preset=None):
        state = self.state(state)
        actions = legal_actions(state) if actions is None else tuple(actions)
        if not actions:
            return {"action": None, "action_record": None, "terminal": True,
                    "ordinary_score": sim.score(state), "model_id": self.model_id}
        q = self.values(state, actions, preset)
        index = int(np.argmax(q))
        selected = actions[index]
        return {
            "action": sim.action_key(selected),
            "action_record": sim.to_jsonable(selected),
            "q": float(q[index]),
            "model_id": self.model_id,
            "checkpoint": str(self.checkpoint),
            "ordinary_score": sim.score(state),
            "terminal": False,
            "values": [
                {"action": sim.action_key(action), "action_record": sim.to_jsonable(action),
                 "expected_remaining_ordinary_return": float(value)}
                for action, value in zip(actions, q)
            ],
        }


__all__ = ["ValueModel", "PublicState", "Action", "Preset", "sim", "replace", "legal_actions"]
