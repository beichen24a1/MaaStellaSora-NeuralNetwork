"""Create a local clean-price handoff; never upload or include player logs."""
import argparse
import hashlib
import json
from zipfile import ZipFile, ZIP_DEFLATED

from bootstrap import HERE, ROOT


def build(run="joint_v3_prices_clean", include_training_data=False):
    directory = HERE / "runs" / run
    validation = directory / "clean_price_validation.json"
    if not validation.exists():
        raise ValueError("Complete clean-price validation before packaging")
    audit = json.loads(validation.read_text(encoding="utf-8"))
    model = directory / "joint_critic.npz"
    if audit["checkpoint_sha256"] != hashlib.sha256(model.read_bytes()).hexdigest():
        raise ValueError("Checkpoint changed after the clean-price audit")
    selected = list(HERE.glob("*.py")) + list(HERE.glob("*.md"))
    selected += [HERE / "requirements.txt", HERE / ".gitattributes", HERE / ".gitignore"]
    selected += list((HERE / "data").glob("*.json")) + list((HERE / "models").glob("*"))
    selected += list(directory.glob("*.json")) + list(directory.glob("*.txt")) + [model]
    selected += list((directory / "evaluation").glob("*"))
    selected += [p for p in (ROOT / "maa_value_plugin/critic").rglob("*") if p.is_file()
                 and (p.suffix == ".py" or p.name in ("SOURCE.json", "README.txt", "LICENSE", "LICENSE.txt"))]
    selected += [ROOT / "maa_value_plugin/__init__.py", ROOT / "maa_value_plugin/licenses.json"]
    # Old weights are included only as explicitly invalidated audit fixtures.
    selected += [HERE / "runs/joint_v2" / name for name in ("joint_critic.npz", "INVALIDATED.json")]
    if include_training_data:
        selected += [directory / name for name in ("train.npz", "dev.npz", "train.roots.jsonl", "dev.roots.jsonl")]
    selected = sorted(set(p for p in selected if p.is_file()))
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}
    output = HERE / "handoff" / f"{run}_clean_price_handoff.zip"
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr("START_HERE.txt", "Clean-price offline joint critic. See research/concerto_joint/README.md and PRICE_AUDIT.md.\n"
                         "This archive contains no Maa application or raw player observations.\n"
                         "The old joint_v2 checkpoint is INVALIDATED and supplied only for the rejection regression test.\n"
                         f"Simulated training datasets included: {include_training_data}.\n")
    print(json.dumps({"path": str(output), "files": len(manifest), "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="joint_v3_prices_clean")
    parser.add_argument("--include-training-data", action="store_true")
    args = parser.parse_args()
    build(args.run, args.include_training_data)
