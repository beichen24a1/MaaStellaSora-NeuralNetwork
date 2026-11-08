"""Generate fresh joint rollouts and train a separate two-return action critic."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from random import Random
import time

from bootstrap import HERE
import numpy as np

from joint_env import JointEpisode, RULES, random_scenarios, fingerprint
from joint_model import Baseline, JointCritic, encode_joint, play, FEATURE_VERSION, GREEDY_PROGRESS_INDEX
from maa_value_plugin.critic.presets import canonical_seed
from rule_alignment import require_training_ready


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def generate(path, episodes, seed, rollouts, roots_per_episode=4):
    require_training_ready()
    configs = random_scenarios(12, seed)
    teacher = Baseline("greedy")
    all_x, all_y, all_ids, roots, failures = [], [], [], [], Counter()
    root_records = path.with_suffix(".roots.jsonl")
    with root_records.open("w", encoding="utf-8") as record_file:
        for game in range(episodes):
            game_seed = canonical_seed("collection", seed, game)
            config = configs[game % len(configs)]
            env = JointEpisode(config, game_seed)
            rng = Random(canonical_seed("exploration", seed, game))
            pending = {"SHOP": [], "EVENT": []}
            try:
                while not env.done:
                    if env.phase in pending and len(env.actions) > 1:
                        pending[env.phase].append(env.snapshot())
                        action = rng.choice(env.actions) if rng.random() < .15 else teacher(env)
                    else:
                        action = teacher(env)
                    env.step(action)
            except Exception as error:
                if error.__class__.__name__ != "ModelGap":
                    raise
                failures[f"collection:{error}"] += 1
                continue
            candidates = []
            for phase, count in (("SHOP", roots_per_episode - 1), ("EVENT", 1)):
                candidates.extend(rng.sample(pending[phase], min(count, len(pending[phase]))))
            for snapshot in candidates:
                root_id = len(roots)
                root_seed = canonical_seed("labels", seed, game, fingerprint(snapshot))
                root = JointEpisode.from_public(snapshot, root_seed)
                actions = root.actions
                x, prior = encode_joint(root, actions)
                before = (root.ep.score, sum(root.levels()))
                returns = np.zeros((len(actions), rollouts, 2), dtype=np.float32)
                failed = False
                for ai, action in enumerate(actions):
                    for rollout in range(rollouts):
                        branch_seed = canonical_seed("continuation", root_seed, rollout)
                        branch = JointEpisode.from_public(snapshot, branch_seed)
                        try:
                            branch.step(action)
                            result = play(branch, teacher)
                        except Exception as error:
                            if error.__class__.__name__ != "ModelGap":
                                raise
                            failures[f"label:{error}"] += 1
                            failed = True
                            break
                        returns[ai, rollout] = (result["ordinary"] - before[0], result["concerto"] - before[1])
                    if failed:
                        break
                if failed:
                    continue  # Exclude the WHOLE root, never just an unfavorable action.
                mean = returns.mean(1)
                target = np.stack(((mean[:, 0] - prior) / 20,
                                   mean[:, 1] / 5 - x[:, GREEDY_PROGRESS_INDEX]), axis=1)
                all_x.extend(x); all_y.extend(target); all_ids.extend([root_id] * len(actions))
                roots.append({"game": game, "seed": str(game_seed), "config_fingerprint": config["fingerprint"],
                              "phase": root.phase, "actions": [action.key() for action in actions],
                              "samples": len(actions), "root_seed": str(root_seed)})
                record_file.write(json.dumps({**roots[-1], "snapshot": snapshot}, ensure_ascii=False) + "\n")
            if (game + 1) % 4 == 0:
                log(f"data {path.stem}: episodes={game + 1}/{episodes}, roots={len(roots)}, rows={len(all_x)}")
    if not all_x:
        raise RuntimeError("No complete labeled roots")
    metadata = {"feature_version": FEATURE_VERSION, "rules": RULES, "seed": seed,
                "episodes_requested": episodes, "rollouts_per_action": rollouts,
                "roots": roots, "failures": dict(failures), "configs": configs}
    np.savez_compressed(path, x=np.asarray(all_x, dtype=np.float32), y=np.asarray(all_y, dtype=np.float32),
                        root=np.asarray(all_ids, dtype=np.int32), metadata=np.array(json.dumps(metadata)))
    path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def fit(train_path, dev_path, output, epochs, seed):
    require_training_ready()
    import torch
    from torch import nn
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def load(path):
        with np.load(path, allow_pickle=False) as data:
            return data["x"], data["y"], data["root"], json.loads(str(data["metadata"].item()))
    tx, ty, tr, tm = load(train_path)
    vx, vy, vr, vm = load(dev_path)
    if any(m["feature_version"] != FEATURE_VERSION or m["rules"] != RULES for m in (tm, vm)):
        raise ValueError("Dataset feature/rule contract changed; regenerate joint labels")
    train_configs = {c["fingerprint"] for c in tm["configs"]}
    dev_configs = {c["fingerprint"] for c in vm["configs"]}
    if train_configs & dev_configs:
        raise ValueError("Train/dev configuration leakage")
    if {r["seed"] for r in tm["roots"]} & {r["seed"] for r in vm["roots"]}:
        raise ValueError("Train/dev episode leakage")
    model = nn.Sequential(nn.Linear(tx.shape[1], 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 2)).to(device)
    nn.init.zeros_(model[-1].weight); nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    def batches(x, y, root):
        groups = [np.flatnonzero(root == r) for r in sorted(set(root))]
        width = max(map(len, groups))
        bx = np.zeros((len(groups), width, x.shape[1]), dtype=np.float32)
        by = np.zeros((len(groups), width, 2), dtype=np.float32)
        mask = np.zeros((len(groups), width), dtype=np.float32)
        for i, indices in enumerate(groups):
            n = len(indices); bx[i, :n] = x[indices]; by[i, :n] = y[indices]; mask[i, :n] = 1
        return [torch.tensor(a, device=device) for a in (bx, by, mask)]
    x, y, mask = batches(tx, ty, tr)
    dx, dy, dm = batches(vx, vy, vr)
    rng = np.random.default_rng(seed)
    best_regret, best_params, history = float("inf"), None, []
    for epoch in range(epochs + 1):
        if epoch:
            order = rng.permutation(len(x))
            model.train()
            for start in range(0, len(order), 16):
                idx = order[start:start + 16]
                bm = mask[idx]
                pred = model(x[idx])
                target = y[idx]
                count = bm.sum(1, keepdim=True).unsqueeze(-1)
                centered_pred = pred - (pred * bm[..., None]).sum(1, keepdim=True) / count
                centered_target = target - (target * bm[..., None]).sum(1, keepdim=True) / count
                loss = (((centered_pred - centered_target) ** 2) * bm[..., None]).sum() / (2 * bm.sum())
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(dx)
            # Last public feature is the frozen ordinary prior /20.
            public_prior = dx[..., -1] * 20 + dx[..., GREEDY_PROGRESS_INDEX] * 15
            true_value = public_prior + dy[..., 0] * 20 + dy[..., 1] * 15
            predicted = public_prior + pred[..., 0] * 20 + pred[..., 1] * 15
            predicted = predicted.masked_fill(dm == 0, -1e9)
            selected = predicted.argmax(1)
            best = true_value.masked_fill(dm == 0, -1e9).max(1).values
            regret = (best - true_value[torch.arange(len(dx), device=device), selected]).mean().item()
        history.append({"epoch": epoch, "dev_mc_regret": regret})
        if regret < best_regret:
            best_regret = regret
            layers = (model[0], model[2], model[4])
            best_params = {f"{kind}{i}": (layer.weight.detach().cpu().numpy().T.copy() if kind == "W" else layer.bias.detach().cpu().numpy().copy())
                           for i, layer in enumerate(layers) for kind in ("W", "b")}
            best_epoch = epoch
        if epoch % 20 == 0:
            log(f"fit {device}: epoch={epoch}/{epochs}, dev MC regret={regret:.3f}, best={best_regret:.3f}")
    metadata = {"architecture": [tx.shape[1], 256, 128, 2], "preference_weight": 3.,
                "head_semantics": ["ordinary advantage correction /20", "concerto advantage correction to public greedy progress /5"],
                "residual_scale": 1.,
                "continuation_teacher": "frozen beta plus public greedy joint baseline",
                "terminal_adjustment_policy": "shared deterministic greedy reference, not learned",
                "training_seed": seed, "best_epoch": best_epoch, "best_dev_mc_regret": best_regret,
                "train_roots": len(tm["roots"]), "dev_roots": len(vm["roots"]),
                "train_rows": len(tx), "dev_rows": len(vx), "history": history,
                "training_source": "fresh simulated joint branches, not historical ordinary-only labels",
                "status": "experimental_pending_episode_evaluation", "rules": RULES}
    critic = JointCritic(best_params, metadata)
    critic.save(output)
    # Ensure portable NumPy inference reproduces the actual selected Torch weights.
    portable = JointCritic.load(output)
    for i, layer in enumerate((model[0], model[2], model[4])):
        layer.weight.data.copy_(torch.tensor(best_params[f"W{i}"].T, device=device))
        layer.bias.data.copy_(torch.tensor(best_params[f"b{i}"], device=device))
    with torch.no_grad():
        expected = model(torch.tensor(vx[:64], device=device)).cpu().numpy()
    error = float(np.max(np.abs(expected - portable.predict(vx[:64]))))
    if error > 1e-4:
        raise RuntimeError(f"Torch/NumPy export mismatch: {error}")
    log(f"saved {output.name}; best epoch={best_epoch}; export max error={error:.2g}")
    return critic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--dev-episodes", type=int, default=16)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--run", default="joint_v4_env_fixed")
    parser.add_argument("--reuse-data", action="store_true")
    args = parser.parse_args()
    require_training_ready()
    from audit_environment import audit
    check = audit()
    if check["confirmed_issues"]:
        raise RuntimeError("Environment audit blocks training: " + repr(check["confirmed_issues"]))
    out = HERE / "runs" / args.run
    if (out / "KNOWN_ISSUES.json").exists() or (out / "INVALIDATED.json").exists():
        raise ValueError("Use a new run directory; do not overwrite invalidated historical evidence")
    out.mkdir(parents=True, exist_ok=True)
    for split, episodes, offset in (("train", args.train_episodes, 0), ("dev", args.dev_episodes, 100000)):
        path = out / f"{split}.npz"
        if not args.reuse_data or not path.exists():
            generate(path, episodes, args.seed + offset, args.rollouts)
    fit(out / "train.npz", out / "dev.npz", out / "joint_critic.npz", args.epochs, args.seed)


if __name__ == "__main__":
    main()
