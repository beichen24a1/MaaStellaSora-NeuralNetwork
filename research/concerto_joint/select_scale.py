"""Choose residual strength on development episodes, never final test games."""
import argparse
import json
import time

from bootstrap import HERE
import numpy as np
from joint_env import JointEpisode
from joint_model import JointCritic, play
from maa_value_plugin.critic.presets import canonical_seed


def select(run="joint_v4_env_fixed", rounds=2):
    directory = HERE / "runs" / run
    checkpoint = directory / "joint_critic.npz"
    model = JointCritic.load(checkpoint)
    metadata = json.loads((directory / "dev.json").read_text(encoding="utf-8"))
    scores = {}
    started = time.time()
    for scale in (0., .1, .25, .5, 1.):
        model.metadata["residual_scale"] = scale
        rows = []
        for config in metadata["configs"]:
            for i in range(rounds):
                seed = canonical_seed("development_scale_selection", 20271107, config["fingerprint"], i)
                result = play(JointEpisode(config, seed), model)
                rows.append({"config": config["fingerprint"], "seed": str(seed),
                             "ordinary": result["ordinary"], "concerto": result["concerto"],
                             "objective": result["ordinary"] + 3 * result["concerto"]})
        scores[str(scale)] = {"mean_objective": float(np.mean([r["objective"] for r in rows])),
                               "mean_ordinary": float(np.mean([r["ordinary"] for r in rows])),
                               "mean_concerto": float(np.mean([r["concerto"] for r in rows])),
                               "episodes": rows}
        print(f"dev scale={scale}: objective={scores[str(scale)]['mean_objective']:.3f}, elapsed={time.time()-started:.1f}s", flush=True)
    best = max(scores, key=lambda value: (scores[value]["mean_objective"], -float(value)))
    model.metadata["residual_scale"] = float(best)
    model.metadata["development_scale_selection"] = {"selected": float(best), "rounds_per_dev_config": rounds,
                                                      "seed_domain": "development_scale_selection",
                                                      "results": scores, "final_test_used": False}
    model.save(checkpoint)
    print(json.dumps({"selected_residual_scale": float(best), "development_mean": scores[best]["mean_objective"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="joint_v4_env_fixed")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    select(args.run, args.rounds)
