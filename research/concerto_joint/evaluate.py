"""Held-out complete-episode evaluation with paired outcomes and sensitivity."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import time

from bootstrap import HERE
import numpy as np

from joint_env import JointEpisode, RULES, random_scenarios, reference_scenarios
from joint_model import Baseline, JointCritic, play
from maa_value_plugin.critic.presets import canonical_seed
from rule_alignment import require_training_ready


def paired_summary(rows, comparator, weight=3.):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["config"], row["seed"]), {})[row["policy"]] = row
    pairs = [(values["joint"], values[comparator]) for values in grouped.values()
             if "joint" in values and comparator in values]
    if not pairs:
        return {"pairs": 0}
    result = {"pairs": len(pairs)}
    rng = np.random.default_rng(20261107)
    for metric in ("potential_total", "concerto", "objective", "note_spent"):
        differences = np.array([a[metric] - b[metric] for a, b in pairs])
        bootstrap = differences[rng.integers(0, len(differences), (4000, len(differences)))].mean(1)
        result[metric] = {"paired_mean_delta": float(differences.mean()),
                          "paired_seed_bootstrap_ci95": np.quantile(bootstrap, [.025, .975]).tolist()}
    return result


def summarize(rows):
    result = {}
    for policy in sorted({r["policy"] for r in rows}):
        values = [r for r in rows if r["policy"] == policy]
        result[policy] = {"completed": len(values)}
        for metric in ("potential_total", "ordinary", "concerto", "objective", "note_spent", "note_purchases"):
            data = [row[metric] for row in values]
            result[policy][metric] = {"mean": float(np.mean(data)), "p5": float(np.quantile(data, .05)),
                                      "std": float(np.std(data))}
    return result


def evaluate(checkpoint, rounds=32, seed=20401107, stress_rounds=8):
    require_training_ready()
    critic = JointCritic.load(checkpoint)
    configs = reference_scenarios() + random_scenarios(3, seed)
    training_dir = checkpoint.parent
    excluded = set()
    episode_ids = set()
    for split in ("train", "dev"):
        metadata = json.loads((training_dir / f"{split}.json").read_text(encoding="utf-8"))
        excluded.update(c["fingerprint"] for c in metadata["configs"])
        episode_ids.update(root["seed"] for root in metadata["roots"])
    if excluded & {c["fingerprint"] for c in configs}:
        raise ValueError("Held-out configuration overlaps training/development")
    policies = {"beta_only": Baseline("beta_only"), "greedy": Baseline("greedy"), "joint": critic}
    rows, failures = [], []
    start = time.time()
    conditions = [("primary", 8, 15, True, rounds)]
    if stress_rounds:
        conditions.extend((("low_drops", 4, 15, True, stress_rounds),
                           ("high_drops", 12, 15, True, stress_rounds),
                           ("no_swaps", 8, 0, True, stress_rounds),
                           ("no_note_events", 8, 15, False, stress_rounds)))
    for condition, drops, swaps, events, count in conditions:
        panel = configs if condition == "primary" else configs[:3]
        names = policies if condition == "primary" else ("greedy", "joint")
        for ci, config in enumerate(panel):
            for i in range(count):
                game_seed = canonical_seed("held_out", seed, condition, ci, i)
                if str(game_seed) in episode_ids:
                    raise ValueError("Held-out episode seed leaked")
                for name in names:
                    try:
                        result = play(JointEpisode(config, game_seed, drops, swaps, events), policies[name])
                        row = {"condition": condition, "config": config["name"], "config_fingerprint": config["fingerprint"],
                               "seed": str(game_seed), "policy": name, **result}
                        row["objective"] = row["ordinary"] + 3 * row["concerto"]
                        rows.append(row)
                    except Exception as error:
                        if error.__class__.__name__ != "ModelGap":
                            raise
                        failures.append({"condition": condition, "config": config["name"], "seed": str(game_seed),
                                         "policy": name, "error": str(error)})
            print(f"eval {condition}: config {ci + 1}/{len(panel)}, episodes={len(rows)}, elapsed={time.time() - start:.1f}s", flush=True)
    out = training_dir / "evaluation"
    out.mkdir(exist_ok=True)
    report = {"schema": "joint_episode_evaluation_v1", "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "rules": RULES, "held_out_seed": seed, "rounds_per_primary_config": rounds,
              "config_count": len(configs), "configs": configs, "failures": failures,
              "status": "experimental_offline_model", "conditions": {},
              "interpretation": ["Primary score preference = ordinary potential + 3*sum(concerto levels); this is not measured damage",
                                 "beta_only gets the same events/drop/adjustment environment but cannot buy notes",
                                 "All three policies use the same fixed terminal greedy adjustment operator",
                                 "Shop/event choices are learned; ordinary card choices retain frozen beta",
                                 "Intervals resample paired seeds across the fixed configuration panel, not unseen games or real-world conditions",
                                 "Held-out configurations and episode seeds are disjoint from train/dev",
                                 "Uncertain drop/event/swap rules are evaluated as assumptions, not established game truth"]}
    for condition, *_ in conditions:
        panel = [r for r in rows if r["condition"] == condition]
        report["conditions"][condition] = {"policies": summarize(panel),
                                           "joint_vs_greedy": paired_summary(panel, "greedy")}
        if condition == "primary":
            report["conditions"][condition]["joint_vs_beta_only"] = paired_summary(panel, "beta_only")
            report["per_config"] = {config["name"]: summarize([r for r in panel if r["config_fingerprint"] == config["fingerprint"]])
                                    for config in configs}
    primary = report["conditions"]["primary"]["joint_vs_greedy"]
    lower = primary.get("objective", {}).get("paired_seed_bootstrap_ci95", [-float("inf")])[0]
    report["promotion"] = {"recommended_over_greedy_in_primary_simulator": bool(lower > 0 and not failures),
                           "live_game_validated": False,
                           "reason": "Requires positive paired 95% lower bound and no simulation gaps; real-game transfer remains unvalidated"}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = ["condition", "config", "seed", "policy", "potential_total", "ordinary", "concerto", "objective", "note_spent", "note_purchases"]
    with (out / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"report": str(out / "report.json"), "primary": report["conditions"]["primary"],
                      "promotion": report["promotion"], "failure_count": len(failures)}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="joint_v4_env_fixed")
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--stress-rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20401107)
    args = parser.parse_args()
    evaluate(HERE / "runs" / args.run / "joint_critic.npz", args.rounds, args.seed, args.stress_rounds)
