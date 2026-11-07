"""Reproduce known joint-environment defects without training another model."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from bootstrap import HERE
from joint_env import JointEpisode, RULES, random_scenarios
from joint_model import Baseline
from price_data import normalize_note_price


def observe(path):
    discounts, quantities, shelves = Counter(), Counter(), Counter()
    digest = hashlib.sha256()
    for line in path.open("rb"):
        digest.update(line)
        row = json.loads(line)
        if row.get("shop_event_type") != "stock_snapshot":
            continue
        payload = row["payload"]
        if not payload.get("all_slots_observed") or not payload.get("all_slots_parsed"):
            raise ValueError("Incomplete shelf: audit cannot assume eight observed goods")
        discounted = 0
        for item in payload["slots"]:
            name, quantity, price = item["internal_name"], item["quantity"], item["effective_price"]
            if name == "potential_drink":
                standard = 200
            elif name.startswith("melody_of_"):
                price = normalize_note_price(quantity, price)
                standard = {5: 90, 15: 400}[quantity]
                quantities[quantity] += 1
            else:
                raise ValueError("Unrecognized product; cannot infer discount status")
            discounted += price < standard
        discounts[discounted] += 1
        identity = (row.get("exploration_id") or row.get("run_id"), row.get("shop_entry_id"), row.get("shelf_generation_id"))
        shelves[identity] += 1
    return {"source_sha256": digest.hexdigest(), "stock_snapshots": sum(discounts.values()),
            "discounted_slot_histogram": dict(discounts), "note_quantity_counts": dict(quantities),
            "distinct_recorded_shelf_keys": len(shelves),
            "repeats_by_recorded_shelf_key": sum(n - 1 for n in shelves.values()),
            "independence_caveat": "Inferred IDs cannot prove physical-shop independence; this is not a transaction count."}


def semantic(config):
    return (tuple(tuple(pool) for pool in config["preset"]["pools"]),
            tuple(sorted(config["groups"])), tuple(config["supply"]), config["tower"])


def audit(observations=None, games=80):
    config = random_scenarios(1, 91)[0]
    result = {"schema": "joint_environment_audit_v1", "rules_version": RULES["version"],
              "status": "confirmed_implementation_issues_block_model_recommendation",
              "event_probes": [], "confirmed_issues": []}
    for eid, options, count in ((107, tuple(range(10701, 10709)) + (0,), 50),
                                (108, tuple(range(10801, 10810)) + (0,), 50),
                                (103, (0, 10301, 10302), 0), (103, (0, 10301, 10302), 50),
                                (104, (0, 10401, 10402, 10403), 50)):
        env = JointEpisode(config, 1)
        env.notes = (count,) + (0,) * 12
        env.ep.state = replace(env.ep.state, phase="EVENT", eventId=eid,
                               eventOptions=options, seen=(eid,), coins=1000)
        result["event_probes"].append({"event": eid, "note_total": count,
                                       "visible_known_options": list(options),
                                       "actual_joint_actions": [action.key() for action in env.actions]})
    if len(result["event_probes"][0]["actual_joint_actions"]) == 1:
        result["confirmed_issues"].append("note_event_options_filtered_by_ordinary_predictive_subset")
    if not result["event_probes"][2]["actual_joint_actions"]:
        result["confirmed_issues"].append("ordinary_dominance_prunes_joint_alternatives_before_note_affordability")
    generated = Counter()
    for seed in range(games):
        env, policy = JointEpisode(config, seed), Baseline()
        while env.phase != "SHOP":
            env.step(policy(env))
        discounted = sum(row[0] + row[1] for row in env.ep.state.stock)
        discounted += sum(item.price < {5: 90, 15: 400}[sum(q for _, q in item.contents)] for item in env.shelf)
        generated[discounted] += 1
    result["simulated_first_shops"] = {"games": games, "scenario_seed": 91,
                                        "discounted_slot_histogram": dict(generated),
                                        "not_four_discounted_slots": sum(n for k, n in generated.items() if k != 4)}
    if any(k != 4 for k in generated):
        result["confirmed_issues"].append("independent_note_price_sampling_breaks_reference_four_discount_slot_constraint")
    if observations is not None:
        result["raw_observation_audit"] = observe(observations)
    run = HERE / "runs/joint_v3_prices_clean"
    train = json.loads((run / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((run / "dev.json").read_text(encoding="utf-8"))
    test = json.loads((run / "evaluation/report.json").read_text(encoding="utf-8"))
    a, b, c = ({semantic(row) for row in data["configs"]} for data in (train, dev, test))
    result["numerical_configuration_overlap"] = {"train_dev": len(a & b), "train_test": len(a & c), "dev_test": len(b & c)}
    result["unverified_assumptions"] = ["battle-note quantity and distribution", "terminal transfer legality/timing",
                                         "some event reward branches, including note-or-heal", "concerto level sum as a proxy for combat value"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "handoff/environment_audit_latest.json")
    parser.add_argument("--games", type=int, default=80)
    args = parser.parse_args()
    result = audit(args.observations, args.games)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(1 if result["confirmed_issues"] else 0)
