"""One-command local reproduction; no game automation or network access."""
import argparse
import subprocess
import sys

from bootstrap import HERE
from rule_alignment import require_training_ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--run", default="joint_v4_env_fixed")
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--dev-episodes", type=int, default=16)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--stress-rounds", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=20401107)
    args = parser.parse_args()
    require_training_ready()
    if not (HERE / "data/inputs.json").exists():
        raise FileNotFoundError("Frozen research inputs are missing")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(HERE), "-v"], check=True)
    command = [sys.executable, str(HERE / "train.py"), "--run", args.run,
               "--train-episodes", str(args.train_episodes), "--dev-episodes", str(args.dev_episodes),
               "--rollouts", str(args.rollouts), "--epochs", str(args.epochs)]
    if args.reuse_data:
        command.append("--reuse-data")
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(HERE / "select_scale.py"), "--run", args.run], check=True)
    subprocess.run([sys.executable, str(HERE / "make_example.py"), "--run", args.run], check=True)
    subprocess.run([sys.executable, str(HERE / "evaluate.py"), "--run", args.run, "--rounds", str(args.rounds),
                    "--stress-rounds", str(args.stress_rounds), "--seed", str(args.eval_seed)], check=True)
    subprocess.run([sys.executable, str(HERE / "finalize.py"), "--run", args.run], check=True)
    subprocess.run([sys.executable, str(HERE / "audit_clean_run.py"), "--run", args.run], check=True)
    subprocess.run([sys.executable, str(HERE / "audit_environment.py"), "--run", args.run,
                    "--output", str(HERE / "runs" / args.run / "environment_validation.json")], check=True)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(HERE), "-v"], check=True)
    subprocess.run([sys.executable, str(HERE / "summarize.py"), "--run", args.run], check=True)
    subprocess.run([sys.executable, str(HERE / "build_handoff.py"), "--run", args.run], check=True)


if __name__ == "__main__":
    main()
