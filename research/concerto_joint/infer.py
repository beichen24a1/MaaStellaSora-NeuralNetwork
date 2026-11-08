"""Public JSON snapshot -> offline joint decision, without game control."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from bootstrap import HERE
import numpy as np
from joint_env import JointEpisode, RULES
from joint_model import JointCritic, encode_joint, GREEDY_PROGRESS_INDEX
from price_data import validate_note_price


def decide(snapshot, checkpoint):
    if snapshot["ordinary_state"]["phase"] not in ("SHOP", "EVENT", "PAGE", "ENHANCE_PAGE", "END"):
        raise ValueError("Only an already observed decision boundary can be evaluated")
    for offer in snapshot.get("shelf", []):
        validate_note_price(sum(quantity for _note, quantity in offer["contents"]), offer["price"])
    critic = JointCritic.load(checkpoint)
    env = JointEpisode.from_public(snapshot, 0)
    if env.done:
        return {"terminal": True, "action": None, "rules_version": RULES["version"]}
    action = critic.choose(env)
    output = {"terminal": False, "rules_version": RULES["version"],
              "action": asdict(action), "action_key": action.key(),
              "policy_component": "learned_joint_critic" if env.phase in ("SHOP", "EVENT") else
                                  "fixed_terminal_adjustment" if env.phase == "ADJUST" else "frozen_beta",
              "preference_weight": critic.metadata["preference_weight"],
              "residual_scale": critic.metadata.get("residual_scale", 1.),
              "rankings": []}
    if env.phase in ("SHOP", "EVENT"):
        actions = env.actions
        x, prior = encode_joint(env, actions)
        predictions = critic.predict(x)
        scores = critic.scores(env, actions)
        for i in np.argsort(-scores):
            output["rankings"].append({"action_key": actions[i].key(),
                                        "preference_score": float(scores[i]),
                                        "ordinary_prior": float(prior[i]),
                                        "ordinary_advantage_correction": float(predictions[i, 0] * 20),
                                        "public_concerto_progress": float(x[i, GREEDY_PROGRESS_INDEX] * 5),
                                        "learned_concerto_correction": float(predictions[i, 1] * 5)})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--run", required=True, help="Explicit run required; there is no active calibrated checkpoint")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = decide(json.loads(args.snapshot.read_text(encoding="utf-8")),
                    HERE / "runs" / args.run / "joint_critic.npz")
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
