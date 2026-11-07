"""Verify regenerated root snapshots, tensors and checkpoint against clean prices."""
import argparse
import hashlib
import json

from bootstrap import HERE
import numpy as np
from joint_env import RULES
from joint_model import BETA, FEATURE_VERSION, JointCritic
from maa_value_plugin.critic.features import FEATURE_DIM
from price_data import PRICE_SCHEMA, validate_note_goods, validate_note_price


def audit(run="joint_v4_env_fixed"):
    directory = HERE / "runs" / run
    raw = (HERE / "data/inputs.json").read_bytes().replace(b"\r\n", b"\n")
    data = json.loads(raw)
    price_audit = json.loads((HERE / "data/price_audit.json").read_text(encoding="utf-8"))
    validate_note_goods(data["note_goods"])
    if data.get("price_schema") != PRICE_SCHEMA:
        raise ValueError("Wrong note-price schema")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != price_audit["repaired_input_sha256_lf"]:
        raise ValueError("Inputs do not match audited repair")
    result = {"schema": "clean_price_training_audit_v1", "run": run,
              "rules_version": RULES["version"], "price_schema": PRICE_SCHEMA,
              "input_sha256_lf": digest, "splits": {}}
    for split in ("train", "dev"):
        path = directory / f"{split}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            x, y, root = arrays["x"], arrays["y"], arrays["root"]
            metadata = json.loads(str(arrays["metadata"].item()))
        if metadata["rules"] != RULES or metadata["feature_version"] != FEATURE_VERSION:
            raise ValueError("Old dataset reused under clean-price rule version")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite training features/labels")
        # Published feature contract: ordinary action features, then 496 note
        # context features, then eight 16-field visible-product descriptions.
        start = FEATURE_DIM + 496
        shelf = x[:, start:start + 8 * 16].reshape(-1, 8, 16)
        bad_rows = int(np.any((shelf[:, :, 2] < 0) | (shelf[:, :, 2] > 1.000001), axis=1).sum())
        if bad_rows:
            raise ValueError(f"Illegal price feature remains in {split}")
        roots = [json.loads(line) for line in path.with_suffix(".roots.jsonl").read_text(encoding="utf-8").splitlines()]
        for record in roots:
            for item in record["snapshot"]["shelf"]:
                validate_note_price(sum(q for _note, q in item["contents"]), item["price"])
        if len(roots) != len(set(root)):
            raise ValueError("Root/tensor grouping mismatch")
        result["splits"][split] = {"roots": len(roots), "action_rows": len(x),
                                   "rows_with_bad_price_features": bad_rows,
                                   "max_normalized_shelf_price": float(shelf[:, :, 2].max()),
                                   "all_snapshot_prices_legal": True,
                                   "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    model = JointCritic.load(directory / "joint_critic.npz")
    teacher_hash = hashlib.sha256(BETA.read_bytes()).hexdigest()
    if teacher_hash != "106b1271301dc05d702bc110b62a0b6b4f920b7144c7f61c98087e979a1d1070":
        raise ValueError("Frozen ordinary beta changed during price repair")
    result["frozen_beta_sha256"] = teacher_hash
    result["frozen_beta_retrained"] = False
    result["checkpoint_sha256"] = hashlib.sha256((directory / "joint_critic.npz").read_bytes()).hexdigest()
    result["parameter_sha256"] = model.metadata["parameter_sha256"]
    result["status"] = "price_domain_and_artifact_identity_only_verified"
    result["scope"] = "This does not verify event-action completeness, joint dominance pruning, shelf discount correlations or game fidelity."
    (directory / "clean_price_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="joint_v4_env_fixed")
    args = parser.parse_args()
    audit(args.run)
