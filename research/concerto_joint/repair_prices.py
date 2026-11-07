"""Reproduce the price repair from the immutable, explicitly invalidated input."""
import argparse
import hashlib
import json
from pathlib import Path

from bootstrap import HERE
from price_data import PRICE_SCHEMA, repair_note_goods


def repair(source, output, audit_path):
    raw = source.read_bytes().replace(b"\r\n", b"\n")
    payload = json.loads(raw)
    clean, audit = repair_note_goods(payload["note_goods"])
    audit["source_input_sha256_lf"] = hashlib.sha256(raw).hexdigest()
    audit["raw_observations_sha256"] = next((v for k, v in payload["source_sha256"].items()
                                             if k.endswith("observations.jsonl")), None)
    audit["repair_basis"] = "Collaborator confirmed original 90/45 and 90/72 price displays were OCR-concatenated; sampled raw texts are 09045 and 09072. No suffix guessing for other values."
    payload["note_goods"] = clean
    payload["schema"] = "joint_inputs_v2_clean_prices"
    payload["price_schema"] = PRICE_SCHEMA
    payload["price_normalization"] = {"source_input_sha256_lf": audit["source_input_sha256_lf"],
                                      "corrected_observations": audit["corrected_observations"],
                                      "audit": "price_audit.json"}
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    audit["repaired_input_sha256_lf"] = hashlib.sha256(encoded).hexdigest()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", type=Path, default=HERE / "data/inputs_unfiltered_v1.json")
    parser.add_argument("--output", type=Path, default=HERE / "data/inputs.json")
    parser.add_argument("--audit", type=Path, default=HERE / "data/price_audit.json")
    args = parser.parse_args()
    repair(args.source, args.output, args.audit)
