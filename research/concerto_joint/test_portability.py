import json
from pathlib import Path
import unittest
from unittest.mock import patch

from bootstrap import HERE
from infer import decide
from joint_model import JointCritic, source_identity


class PortabilityTests(unittest.TestCase):
    def test_text_source_identity_accepts_crlf_checkout(self):
        expected = source_identity()
        original = Path.read_bytes

        def crlf_read(path):
            return original(path).replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")

        with patch.object(Path, "read_bytes", crlf_read):
            self.assertEqual(source_identity(), expected)

    @unittest.skipUnless((HERE / "runs/joint_v3_prices_clean/joint_critic.npz").exists()
                         and (HERE / "runs/joint_v3_prices_clean/example_decision.json").exists(),
                         "clean-price model/example is being regenerated")
    def test_clean_checkpoint_matches_published_public_example(self):
        directory = HERE / "runs/joint_v3_prices_clean"
        snapshot = json.loads((directory / "example_public_state.json").read_text(encoding="utf-8"))
        expected = json.loads((directory / "example_decision.json").read_text(encoding="utf-8"))
        actual = decide(snapshot, directory / "joint_critic.npz")
        self.assertEqual(actual["action_key"], expected["action_key"])
        self.assertEqual(len(actual["rankings"]), len(expected["rankings"]))
        for a, b in zip(actual["rankings"], expected["rankings"]):
            self.assertEqual(a["action_key"], b["action_key"])
            self.assertAlmostEqual(a["preference_score"], b["preference_score"], places=5)

    def test_invalidated_price_model_cannot_load_under_clean_rules(self):
        checkpoint = HERE / "runs/joint_v2/joint_critic.npz"
        if checkpoint.exists():
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                JointCritic.load(checkpoint)


if __name__ == "__main__":
    unittest.main()
