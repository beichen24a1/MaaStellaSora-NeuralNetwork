"""Create a simulated public inference example for the active clean model."""
import argparse
import json

from bootstrap import HERE
from joint_env import JointEpisode, random_scenarios
from joint_model import Baseline
from infer import decide


def make_example(run="joint_v4_env_fixed"):
    directory = HERE / "runs" / run
    env = JointEpisode(random_scenarios(1, 20361107)[0], 417)
    policy = Baseline()
    while env.phase != "SHOP":
        env.step(policy(env))
    snapshot = env.snapshot()
    snapshot["provenance"] = "simulated_example_not_real_game"
    result = decide(snapshot, directory / "joint_critic.npz")
    for name, value in (("example_public_state.json", snapshot), ("example_decision.json", result)):
        (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run": run, "ranked_actions": len(result["rankings"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="joint_v4_env_fixed")
    args = parser.parse_args()
    make_example(args.run)
