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

    def test_v4_checkpoint_and_public_example_are_historical_only(self):
        directory = HERE / "runs/joint_v4_env_fixed"
        snapshot = json.loads((directory / "example_public_state.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            decide(snapshot, directory / "joint_critic.npz")

    def test_invalidated_price_model_cannot_load_under_clean_rules(self):
        checkpoint = HERE / "runs/joint_v2/joint_critic.npz"
        if checkpoint.exists():
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                JointCritic.load(checkpoint)

    def test_environment_invalidated_v3_model_cannot_load(self):
        checkpoint = HERE / "runs/joint_v3_prices_clean/joint_critic.npz"
        if checkpoint.exists():
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                JointCritic.load(checkpoint)


if __name__ == "__main__":
    unittest.main()
