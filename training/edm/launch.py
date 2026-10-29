#!/usr/bin/env python3
"""Train CIFAR-100 flow matching with one of three EDM-codebase recipes."""

import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
CONFIG_NAMES = ("uncond_noaug", "uncond_aug", "cond_aug_dropout")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-configs", action="store_true")
    p.add_argument("--config", choices=CONFIG_NAMES)
    p.add_argument("--prediction", choices=("x", "e", "eps", "v"), default="e")
    p.add_argument("--loss-target", choices=("x", "e", "eps", "v"), default="e")
    p.add_argument("--half", choices=("first", "second"), default="first",
                   help="Per-class training half. The paper's targets use the second half, its shadow models the first")
    p.add_argument("--data-path", type=Path, help="CIFAR-100 training image ZIP/directory")
    p.add_argument("--output-dir", type=Path, help="Exact run directory")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--epochs", type=int, help="Override the recipe's training epochs")
    p.add_argument("--seed", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--batch-gpu", type=int, help="Per-GPU batch (smaller values use gradient accumulation)")
    p.add_argument("--resume", type=Path, help="Trusted checkpoint-last.pt or training-state-*.pt")
    p.add_argument("--dry-run", action="store_true", help="Print options without Torch, data access, or training")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def load_config(name):
    if name not in CONFIG_NAMES:
        raise ValueError("Unknown EDM recipe")
    return json.loads((ROOT / "configs" / (name + ".json")).read_text())


def effective_options(args, config):
    options = copy.deepcopy(config["training_options"])
    prediction = "e" if args.prediction == "eps" else args.prediction
    target = "e" if args.loss_target == "eps" else args.loss_target
    options["loss_kwargs"].update(prediction=prediction, loss_target=target)
    options["dataset_kwargs"]["class_subset"] = f"{args.half}-half-per-class"
    options["dataset_kwargs"]["path"] = str(
        (args.data_path or REPOSITORY_ROOT / "data" / "cifar100-32x32.zip").expanduser().resolve())
    output = args.output_dir or (REPOSITORY_ROOT / "artifacts" / "training" / "edm" / args.config / args.half
                                 / f"pred-{prediction}_loss-{target}")
    options["run_dir"] = str(output.expanduser().resolve())
    if args.resume:
        options["resume_state_dump"] = str(args.resume.expanduser().resolve())
    if args.epochs is not None:
        count = options["dataset_kwargs"]["max_size"]
        options["total_kimg"] = max((args.epochs * count + 999) // 1000, 1)
    if args.seed is not None:
        options["seed"] = args.seed
    if args.workers is not None:
        options["data_loader_kwargs"]["num_workers"] = args.workers
    if args.batch_gpu is not None:
        options["batch_gpu"] = args.batch_gpu
    return options


def execute(args, config, options, p):
    data = Path(options["dataset_kwargs"]["path"])
    if not data.exists():
        p.error("Dataset does not exist: " + str(data))
    if args.resume and not Path(options["resume_state_dump"]).is_file():
        p.error("Resume checkpoint does not exist")
    output = Path(options["run_dir"])
    if output.exists() and not output.is_dir():
        p.error("Output path is not a directory")
    if output.exists() and any(output.iterdir()) and not args.resume:
        p.error("Output directory is nonempty, so choose a new directory or pass --resume")

    # Config inspection works from a standard-library Python installation.
    import numpy as np
    import torch
    import dnnlib
    from torch_utils import distributed as dist
    from training import training_loop

    if not torch.cuda.is_available():
        p.error("Training requires an NVIDIA CUDA GPU")
    torch.multiprocessing.set_start_method("spawn", force=True)
    dist.init()
    batch = options["batch_size"]
    world_size = dist.get_world_size()
    batch_gpu = min(options.get("batch_gpu") or batch // world_size, batch // world_size)
    if batch_gpu < 1 or batch % (batch_gpu * world_size):
        p.error("Global batch must be divisible by per-GPU batch times world size")

    dataset = dnnlib.util.construct_class_by_name(**options["dataset_kwargs"])
    if len(dataset) != options["dataset_kwargs"]["max_size"]:
        p.error("The recipe requires a 25000-image half-per-class subset")
    labels = dataset._load_raw_labels()
    classes = config["dataset"]["num_classes"]
    if dataset.num_channels != config["dataset"]["image_channels"]:
        p.error("CIFAR-100 recipes require three-channel RGB images")
    if labels is None or labels.ndim != 1 or not np.array_equal(np.unique(labels), np.arange(classes)):
        p.error("The dataset must include integer labels for all 100 CIFAR-100 classes")
    if not np.all(np.bincount(labels, minlength=classes) == 500):
        p.error("Supply the full CIFAR-100 training archive: 500 images per class")
    dataset.close()

    if dist.get_rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "training_options.json").write_text(json.dumps(options, indent=2) + "\n")
        (output / "launch_config.json").write_text(json.dumps(dict(
            config=args.config,
            prediction=options["loss_kwargs"]["prediction"],
            loss_target=options["loss_kwargs"]["loss_target"],
        ), indent=2) + "\n")
        dnnlib.util.Logger(file_name=str(output / "log.txt"), file_mode="a", should_flush=True)
    torch.distributed.barrier()
    training_loop.training_loop(**options)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.list_configs:
        for name in CONFIG_NAMES:
            config = load_config(name)
            print(f"{name}: CIFAR-100, {len(config['allowed_pairs'])} prediction/loss pairs")
        return 0
    if not args.config:
        p.error("--config is required unless --list-configs is used")
    for name in ("gpus", "epochs", "workers", "batch_gpu"):
        value = getattr(args, name)
        if value is not None and value < 1:
            p.error("--" + name.replace("_", "-") + " must be positive")
    config = load_config(args.config)
    options = effective_options(args, config)
    pair = [options["loss_kwargs"]["prediction"], options["loss_kwargs"]["loss_target"]]
    if pair not in config["allowed_pairs"]:
        p.error("cond_aug_dropout supports only --prediction e --loss-target e")
    if args.dry_run:
        command_args = [a for a in (sys.argv[1:] if argv is None else argv) if a != "--dry-run"]
        print(json.dumps(dict(config=args.config, training_options=options,
                              command=shlex.join([sys.executable, str(ROOT / "launch.py"), *command_args])), indent=2))
        return 0
    if args.gpus > 1 and not args.worker and "WORLD_SIZE" not in os.environ:
        command_args = list(sys.argv[1:] if argv is None else argv)
        return subprocess.call([sys.executable, "-m", "torch.distributed.run", "--standalone",
                                "--nproc_per_node=" + str(args.gpus),
                                str(ROOT / "launch.py"), *command_args, "--worker"])
    execute(args, config, options, p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
