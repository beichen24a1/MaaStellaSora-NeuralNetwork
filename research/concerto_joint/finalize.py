"""Attach evaluation status without changing learned parameter values."""
import argparse
import hashlib
import json

from bootstrap import HERE
from joint_model import JointCritic, parameter_hash


def finalize(run="joint_v4_env_fixed"):
    directory = HERE / "runs" / run
    checkpoint = directory / "joint_critic.npz"
    report_path = directory / "evaluation/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError("Evaluation does not correspond to this checkpoint")
    critic = JointCritic.load(checkpoint)
    before = parameter_hash(critic.params)
    critic.metadata.update({"status": "evaluated_offline_research",
                            "evaluation_report": "evaluation/report.json",
                            "promotion": report["promotion"]})
    critic.save(checkpoint)
    if parameter_hash(JointCritic.load(checkpoint).params) != before:
        raise RuntimeError("Finalization unexpectedly changed model parameters")
    report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report["parameter_sha256"] = before
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": critic.metadata["status"], "parameters_unchanged": True,
                      "checkpoint_sha256": report["checkpoint_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="joint_v4_env_fixed")
    args = parser.parse_args()
    finalize(args.run)
