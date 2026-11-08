"""Reproduce known joint-environment defects without training another model."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from bootstrap import HERE
from joint_env import JointEpisode, RULES, random_scenarios, sim
from joint_model import Baseline, source_identity
from price_data import normalize_note_price
from rule_alignment import SANDBOX_PROFILE


def observe(path):
    discounts, halves, quantities, shelves = Counter(), Counter(), Counter(), Counter()
    digest = hashlib.sha256()
    for line in path.open("rb"):
        digest.update(line)
        row = json.loads(line)
        if row.get("shop_event_type") != "stock_snapshot":
            continue
        payload = row["payload"]
        if not payload.get("all_slots_observed") or not payload.get("all_slots_parsed"):
            raise ValueError("Incomplete shelf: audit cannot assume eight observed goods")
        if len(payload["slots"]) != 8:
            raise ValueError("Complete-shelf evidence must contain exactly eight slots")
        discounted, half = 0, 0
        for item in payload["slots"]:
            name, quantity, price = item["internal_name"], item["quantity"], item["effective_price"]
            if name == "potential_drink":
                if price not in (100, 160, 200):
                    raise ValueError("Unknown drink price; cannot infer discount status")
                standard = 200
            elif name.startswith("melody_of_"):
                price = normalize_note_price(quantity, price)
                standard = {5: 90, 15: 400}[quantity]
                quantities[quantity] += 1
            else:
                raise ValueError("Unrecognized product; cannot infer discount status")
            discounted += price < standard
            half += price * 2 == standard
        discounts[discounted] += 1
        halves[half] += 1
        identity = (row.get("exploration_id") or row.get("run_id"), row.get("shop_entry_id"), row.get("shelf_generation_id"))
        shelves[identity] += 1
    return {"source_sha256": digest.hexdigest(), "stock_snapshots": sum(discounts.values()),
            "discounted_slot_histogram": dict(discounts), "note_quantity_counts": dict(quantities),
            "half_price_slot_histogram": dict(halves),
            "distinct_recorded_shelf_keys": len(shelves),
            "repeats_by_recorded_shelf_key": sum(n - 1 for n in shelves.values()),
            "independence_caveat": "Inferred IDs cannot prove physical-shop independence; this is not a transaction count."}


def semantic(config):
    return (tuple(tuple(pool) for pool in config["preset"]["pools"]),
            tuple(sorted(config["groups"])), tuple(config["supply"]), config["tower"])


def audit(observations=None, games=80, run=None):
    config = random_scenarios(1, 91)[0]
    result = {"schema": "joint_environment_audit_v1", "rules_version": RULES["version"],
              "source_sha256": source_identity(), "event_probes": [], "confirmed_issues": []}
    for eid, options, count in ((107, (10701, 10704, 10708, 0), 50),
                                (108, (10802, 10806, 10808, 10809, 0), 50),
                                (103, (0, 10301, 10302), 0), (103, (0, 10301, 10302), 50),
                                (104, (0, 10401, 10402, 10403), 50)):
        env = JointEpisode(config, 1, assumptions=SANDBOX_PROFILE)
        env.notes = (count,) + (0,) * 12
        env.ep.state = replace(env.ep.state, phase="EVENT", eventId=eid,
                               eventOptions=options, seen=(eid,), coins=1000,
                               node=len(sim.ROUTE)+1,
                               floor=20, offer=(), source=None)
        env.crossed = len(sim.ROUTE)
        result["event_probes"].append({"event": eid, "note_total": count,
                                       "visible_known_options": list(options),
                                       "actual_joint_actions": [action.key() for action in env.actions]})
    if len(result["event_probes"][0]["actual_joint_actions"]) != 4 or len(result["event_probes"][1]["actual_joint_actions"]) != 5:
        result["confirmed_issues"].append("note_event_options_filtered_by_ordinary_predictive_subset")
    if (len(result["event_probes"][2]["actual_joint_actions"]) != 2 or
            len(result["event_probes"][3]["actual_joint_actions"]) != 3 or
            len(result["event_probes"][4]["actual_joint_actions"]) != 4):
        result["confirmed_issues"].append("ordinary_dominance_prunes_joint_alternatives_before_note_affordability")
    generated, large_bundles = Counter(), Counter()
    for seed in range(games):
        env, policy = JointEpisode(config, seed, assumptions=SANDBOX_PROFILE), Baseline()
        while env.phase != "SHOP":
            env.step(policy(env))
        discounted = sum(row[0] + row[1] for row in env.ep.state.stock)
        discounted += sum(item.price < {5: 90, 15: 400}[sum(q for _, q in item.contents)] for item in env.shelf)
        generated[discounted] += 1
        large_bundles[sum(sum(q for _, q in item.contents) == 15 for item in env.shelf)] += 1
    result["simulated_first_shops"] = {"games": games, "scenario_seed": 91,
                                        "discounted_slot_histogram": dict(generated),
                                        "not_four_discounted_slots": sum(n for k, n in generated.items() if k != 4)}
    result['simulated_first_shops']['large_bundle_histogram'] = dict(large_bundles)
    if any(k > 1 for k in large_bundles):
        result['confirmed_issues'].append('more_than_one_large_bundle_per_shelf')
    if any(k != 4 for k in generated):
        result["confirmed_issues"].append("independent_note_price_sampling_breaks_reference_four_discount_slot_constraint")
    if observations is not None:
        result["raw_observation_audit"] = observe(observations)
    if run is not None:
        directory = HERE / "runs" / run
        train = json.loads((directory / "train.json").read_text(encoding="utf-8"))
        dev = json.loads((directory / "dev.json").read_text(encoding="utf-8"))
        test = json.loads((directory / "evaluation/report.json").read_text(encoding="utf-8"))
        a, b, c = ({semantic(row) for row in data["configs"]} for data in (train, dev, test))
        overlaps = {"train_dev": len(a & b), "train_test": len(a & c), "dev_test": len(b & c)}
        result["numerical_configuration_overlap"] = overlaps
        if any(overlaps.values()):
            result["confirmed_issues"].append("numerical_configuration_leakage")
        checks = {}
        for split in ("train", "dev"):
            roots = [json.loads(line) for line in (directory / f"{split}.roots.jsonl").read_text(encoding="utf-8").splitlines()]
            mismatch, bad_shelves, full_shelves = 0, 0, 0
            for record in roots:
                env = JointEpisode.from_public(record["snapshot"], int(record["root_seed"]))
                actions = [a.key() for a in env.actions]
                mismatch += not actions or actions != record["actions"]
                if env.phase != "SHOP":
                    continue
                stock = env.ep.state.stock
                slots = sum(map(sum, stock)) + sum(item.stock for item in env.shelf)
                discounted = sum(row[0]+row[1] for row in stock)
                half = sum(row[0] for row in stock)
                for item in env.shelf:
                    quantity = sum(q for _, q in item.contents)
                    discounted += item.stock * (item.price < {5: 90, 15: 400}[quantity])
                    half += item.stock * (item.price == {5: 45, 15: 200}[quantity])
                full_shelves += slots == 8
                bad_shelves += slots > 8 or discounted > 4 or half > 2 or (slots == 8 and discounted != 4)
            checks[split] = {"roots": len(roots), "action_set_mismatches_or_empty": mismatch,
                             "full_shelf_roots": full_shelves, "invalid_remaining_shelves": bad_shelves}
            if mismatch or bad_shelves:
                result["confirmed_issues"].append(f"invalid_{split}_root_actions_or_shelves")
        result["dataset_checks"] = checks
    result["unverified_assumptions"] = ["battle-note quantity and distribution", "random note deduction kernel",
                                         "menu pair and item/discount conditional probabilities",
                                         "health-state impacts and server outcome verification",
                                         "terminal modification legality/timing", "concerto level sum as a proxy for combat value"]
    result['joint_training_ready'] = False
    result['sandbox_profile'] = SANDBOX_PROFILE
    result["status"] = ("confirmed_implementation_issues_block_model_recommendation" if result["confirmed_issues"]
                        else "sandbox_rule_checks_pass_training_still_blocked")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "handoff/environment_audit_latest.json")
    parser.add_argument("--games", type=int, default=80)
    parser.add_argument("--run", help="Also verify train/dev/test semantic configuration separation")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    result = audit(args.observations, args.games, args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(1 if result["confirmed_issues"] else 0)
