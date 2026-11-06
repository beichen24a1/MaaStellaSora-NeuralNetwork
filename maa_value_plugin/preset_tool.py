"""Compile ID-based deployment targets using this package's local catalog."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def compile_preset(source, catalog):
    rows = {}
    for aliases in catalog["by_normalized_name"].values():
        for row in aliases:
            key = (row["potential_id"], tuple(row["roles"]))
            rows.setdefault(key, {}).update({k: v for k, v in row.items() if v is not None})
    pools, party, recognition = [], [], []
    for index, member in enumerate(source["party"]):
        role = "master" if index == 0 else "assist"
        available = sorted((r for r in rows.values() if r["character_id"] == member["character_id"] and role in r["roles"]), key=lambda r: r["potential_id"])
        core = {int(k): v for k, v in member["core"].items()}
        ordinary = {int(k): v for k, v in member["ordinary"].items()}
        targets = []

        def target(row, level, rarity):
            return {"potential_id": row["potential_id"], "name_ja_jp": row.get("name_ja_jp"), "name_zh_cn": row.get("name_zh_cn"), "rarity_class": rarity, "target_level": level}

        for row in available:
            if row["item_stype"] == 42:
                item = target(row, core.get(row["potential_id"], 0), "core")
                if item["target_level"]:
                    targets.append(item)
                else:
                    recognition.append(dict(item, role=index + 1, character_id=member["character_id"]))
        for rarity, label in ((1, "rare"), (2, "common")):
            pool = []
            for row in available:
                if row["item_stype"] == 41 and row["item_rarity"] == rarity:
                    level = ordinary.get(row["potential_id"], 0)
                    pool.append(level)
                    targets.append(target(row, level, label))
            pools.append(pool)
        party.append({"role": index + 1, "character_id": member["character_id"], "name_ja_jp": member.get("name_ja_jp"), "name_zh_cn": member.get("name_zh_cn"), "targets": targets})
    totals = [sum(pools[i]) + sum(pools[i + 1]) for i in (0, 2, 4)]
    display = [total + sum(t["target_level"] for t in member["targets"] if t["rarity_class"] == "core") for total, member in zip(totals, party)]
    return {"name": source["name"], "preset_id": source.get("preset_id", source["name"]), "deployment_only": True,
            "training_scope_match": all(18 <= t <= 54 for t in totals) and 72 <= sum(totals) <= 144,
            "pools": pools, "party": party, "ordinary_total": sum(totals), "role_ordinary_totals": totals,
            "game_display_role_totals": display, "game_display_total": sum(display),
            "attribute_tower": source["attribute_tower"], "recognition_catalog": recognition,
            "source": {"target_authority": "preset_targets.json", "name_and_id_catalog": "potential_name_catalog.json"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("apply", "show"))
    parser.add_argument("input", nargs="?", type=Path, default=ROOT / "preset_targets.json")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    preset = compile_preset(read(args.input), read(ROOT / "potential_name_catalog.json"))
    if args.command == "apply":
        (ROOT / "preset.json").write_text(json.dumps(preset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        config_path = ROOT / "config.json"
        config = read(config_path) if config_path.exists() else {}
        config.update(enabled=True, preset_file="preset.json")
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"name": preset["name"], "attribute_tower": preset["attribute_tower"], "ordinary_total": preset["ordinary_total"], "role_ordinary_totals": preset["role_ordinary_totals"], "party": preset["party"]}
    if args.json:
        print(json.dumps(preset if args.command == "apply" else summary, ensure_ascii=False))
    else:
        print(f"{args.command}: {preset['name']} | {preset['attribute_tower']} | ordinary {preset['role_ordinary_totals']} = {preset['ordinary_total']}")
        for member in preset["party"]:
            print(f"  role {member['role']} character {member['character_id']} {member['name_ja_jp']}")
            for item in member["targets"]:
                print(f"    {item['potential_id']} {item['rarity_class']} {item['target_level']} {item['name_ja_jp']} / {item['name_zh_cn']}")


if __name__ == "__main__":
    main()
