"""Launch the six SD1.4 paper training recipes (standard library only)."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE.parents[1] / "data" / "sd14"
DEFAULT_OUTPUT_ROOT = HERE.parents[1] / "artifacts" / "training" / "sd14"
DEFAULT_MODEL = "CompVis/stable-diffusion-v1-4"


def load_configs():
    return json.loads((HERE / "configs.json").read_text())["configs"]


def data_path(args, recipe):
    if args.data_path is not None:
        return Path(args.data_path).expanduser().resolve()
    return DEFAULT_DATA_ROOT / recipe["imagefolders"][args.half]


def build_command(args, recipe, extra):
    command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", "1", "--mixed_precision", "fp16",
        str(HERE / "train_text_to_image.py"),
        "--pretrained_model_name_or_path", args.model_path,
        "--train_data_dir", str(data_path(args, recipe)),
        "--output_dir", str(Path(args.output_dir).expanduser().resolve()),
    ]
    for key, value in recipe["training"].items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    command.extend(extra[1:] if extra[:1] == ["--"] else extra)
    return command


def check_imagefolder(path, expected):
    metadata = path / "metadata.jsonl"
    if not metadata.is_file():
        return f"{path} must contain metadata.jsonl (see training/sd14/prepare_data.py)"
    rows = sum(1 for line in metadata.open(encoding="utf-8") if line.strip())
    if rows != expected:
        return f"{metadata} has {rows} rows, but this recipe and half expect {expected}"
    return None


def main(argv=None):
    configs = load_configs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--config", choices=sorted(configs))
    parser.add_argument("--half", choices=("first", "second"), default="first",
                        help="Training half. The paper's targets use the second half, its shadow models the first.")
    parser.add_argument("--data-path", help="Imagefolder with metadata.jsonl and images (default: data/sd14/<imagefolder>).")
    parser.add_argument("--model-path", default=DEFAULT_MODEL,
                        help=f"SD1.4 model directory or Diffusers model identifier (default {DEFAULT_MODEL}).")
    parser.add_argument("--output-dir", help="Defaults to artifacts/training/sd14/<config>/<half>.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if args.list_configs:
        print("\n".join(configs))
        return 0
    if args.config is None:
        parser.error("--config is required")
    if args.output_dir is None:
        args.output_dir = str(DEFAULT_OUTPUT_ROOT / args.config / args.half)
    recipe = configs[args.config]
    command = build_command(args, recipe, extra)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    error = check_imagefolder(data_path(args, recipe), recipe["expected_images"][args.half])
    if error:
        parser.error(error)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
