"""Launch one of the six DDPM paper training recipes (stdlib-only dry runs)."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", help="Configuration name or JSON path")
    result.add_argument("--list-configs", action="store_true")
    result.add_argument("--half", choices=("first", "second"), default="first",
                        help="Membership half (member/eval indices). The paper's targets use the second half, its shadow models the first")
    result.add_argument("--data-root", type=Path, help="Parent of cifar10/, cifar100/, stl10/, tiny-imagenet-200/")
    result.add_argument("--output-dir", type=Path, help="Checkpoint and TensorBoard directory")
    result.add_argument("--split-root", type=Path, help="Directory with the saved membership NPZ manifests")
    result.add_argument("--seed", type=int, help="Optional seed for Python, NumPy, and PyTorch")
    result.add_argument("--dry-run", action="store_true", help="Print command without importing ML packages or training")
    return result


def build_command(args):
    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        config_path = ROOT / "configs" / (args.config.removesuffix(".json") + ".json")
    if not config_path.is_file():
        raise ValueError(f"Unknown config {args.config!r} (see --list-configs)")
    config = json.loads(config_path.read_text())
    values = dict(config["flags"])
    values["train_split"] = {"first": "member", "second": "eval"}[args.half]
    # Paths are resolved before switching to the code directory.
    values["dataset_root"] = str((args.data_root or REPOSITORY_ROOT / "data" / "ddpm").expanduser().resolve())
    values["logdir"] = str((args.output_dir or REPOSITORY_ROOT / "artifacts" / "training" / "ddpm" / config_path.stem / args.half).expanduser().resolve())
    values["split_root"] = str((args.split_root or ROOT / "splits").expanduser().resolve())
    if args.seed is not None:
        values["seed"] = args.seed
    command = [sys.executable, str(ROOT / "main.py")]
    for key, value in values.items():
        for item in value if isinstance(value, list) else [value]:
            if item is not None:
                if isinstance(item, bool):
                    item = str(item).lower()
                command.append(f"--{key}={item}")
    return command, config


def main(argv=None):
    cli = parser()
    args, extra = cli.parse_known_args(argv)
    if args.list_configs:
        for path in sorted((ROOT / "configs").glob("*.json")):
            print(path.stem)
        return 0
    if not args.config:
        cli.error("--config is required unless --list-configs is given")
    if extra[:1] == ["--"]:
        extra = extra[1:]
    try:
        command, _ = build_command(args)
    except (ValueError, OSError) as error:
        cli.error(str(error))
    # Native absl flags can be supplied after '--', e.g. -- --total_steps=1.
    command.extend(extra)
    if args.dry_run:
        print(f"Preset: {args.config}")
        print(f"Working directory: {ROOT}")
        print(shlex.join(command))
        return 0
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
