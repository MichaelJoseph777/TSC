"""Run TSC or RMC against a trained checkpoint and report AUC, ASR, and TPR@1%FPR.

    python -m attacks tsc --family ddpm --config cifar10 --checkpoint CKPT --half second
    python -m attacks rmc --family sd14 --checkpoint FINETUNED --members A --nonmembers B
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from . import scores
from .data import ImageSet, dataset_sets, half_sets, overlap

SWEEP = np.round(np.arange(-4.0, 4.0 + 1e-9, 0.25), 2)
HALF_WIDTH = 2.5
SD14_BASE = "CompVis/stable-diffusion-v1-4"


def parser():
    p = argparse.ArgumentParser(prog="python -m attacks", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("attack", choices=("tsc", "rmc"))
    p.add_argument("--family", choices=("ddpm", "edm", "sd14"), required=True)
    p.add_argument("--checkpoint", required=True, help="Target model checkpoint (the fine-tuned model for RMC)")
    p.add_argument("--reference",
                   help="RMC reference: the model before fine-tuning (SD1.4 default: the --model-path base model)")
    p.add_argument("--config", help="DDPM or SD1.4 training recipe, required with --half (optional for other DDPM data)")
    schedule = p.add_argument_group("DDPM without a recipe: architecture comes from checkpoint weights")
    schedule.add_argument("--beta-schedule", choices=("linear", "cosine"),
                          help="Training noise schedule (required without --config)")
    schedule.add_argument("--diffusion-steps", type=int, help="Check the number of training diffusion steps")
    schedule.add_argument("--beta-start", type=float, help="Linear beta start (default 1e-4)")
    schedule.add_argument("--beta-end", type=float, help="Linear beta end (default 0.02)")
    p.add_argument("--model-path", help=f"SD1.4 base model directory or Hugging Face identifier (default {SD14_BASE})")
    p.add_argument("--weights", choices=("ema", "raw"), default="ema")
    data = p.add_argument_group("data: choose --dataset, a pair of sets, or --half")
    data.add_argument("--dataset", type=Path, help="JSON file describing member and non-member image sets")
    data.add_argument("--members", help="SOURCE[@INDICES] of training images")
    data.add_argument("--nonmembers", help="SOURCE[@INDICES] of held-out images")
    data.add_argument("--half", choices=("first", "second"),
                      help="Training half of a model trained by this repository, which sets the members and non-members")
    data.add_argument("--data-root", help="DDPM or SD1.4 data root for --half (default data/ddpm or data/sd14)")
    data.add_argument("--data-path", help="EDM CIFAR-100 archive for --half (default data/cifar100-32x32.zip)")
    data.add_argument("--limit", type=int, help="Use only the first N members and non-members")
    method = p.add_argument_group("method")
    mode = method.add_mutually_exclusive_group()
    mode.add_argument("--logsnr", "--center", dest="logsnr", type=float,
                      help="Fixed log-SNR: TSC center or RMC level (default: the paper's setting)")
    mode.add_argument("--logsnrs", type=float, nargs=3, metavar=("LEFT", "CENTER", "RIGHT"),
                      help="TSC: three explicit, equally spaced log-SNR query points")
    mode.add_argument("--timestep", type=float,
                      help="Native TSC center or RMC query: zero-based DDPM/SD1.4 index or EDM flow time in (0,1)")
    mode.add_argument("--timesteps", type=float, nargs=3, metavar=("FIRST", "CENTER", "LAST"),
                      help="TSC: three exact native indices or flow times, in increasing or decreasing order")
    mode.add_argument("--sweep", type=float, nargs="*", metavar="VALUE",
                      help="Sweep log-SNR: default -4..4 step 0.25, or give START STOP STEP")
    method.add_argument("--delta", type=float, help="TSC log-SNR distance from center to each endpoint (default 2.5)")
    method.add_argument("--noises", type=int, help="Noise draws per level (TSC 5, RMC 1)")
    method.add_argument("--condition", choices=("label", "none"), default="label",
                        help="Class-conditional models: ground-truth label or the dropped condition")
    run = p.add_argument_group("run")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--batch-size", type=int)
    run.add_argument("--device")
    run.add_argument("--output", help="Write the report as JSON")
    run.add_argument("--save-losses", help="Write raw losses and selected per-image scores as NPZ")
    return p


def noise_seed(seed, split, index, stream):
    digest = hashlib.sha256(f"{seed}|{split}|{index}|{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


def collect(family, models, image_set, split, levels, args, torch):
    """Losses [images, levels, draws] for each model, with draws shared across levels and models."""
    from .models import to_tensor

    count = len(image_set)
    losses = {name: np.empty((count, len(levels), args.noises), dtype=np.float64) for name in models}
    source, started = image_set.source, time.monotonic()
    for start in range(0, count, args.batch_size):
        indices = image_set.indices[start:start + args.batch_size]
        images = torch.stack([to_tensor(source.image(i), family.resolution) for i in indices])
        labels = [source.label(i) for i in indices]
        latent_generators = [torch.Generator().manual_seed(noise_seed(args.seed, split, i, "latent")) for i in indices]
        with torch.inference_mode():
            clean, condition = family.prepare(images, labels, latent_generators)
            for draw in range(args.noises):
                noise = torch.stack([torch.randn(clean.shape[1:], generator=torch.Generator().manual_seed(
                    noise_seed(args.seed, split, i, draw))) for i in indices]).to(clean.device)
                for column, level in enumerate(levels):
                    for name, model in models.items():
                        values = family.loss(model, clean, condition, noise, level)
                        losses[name][start:start + len(indices), column, draw] = values.double().cpu().numpy()
        done = start + len(indices)
        if done == count or (start // args.batch_size) % 20 == 0:
            print(f"{split}s: {done}/{count} images, {time.monotonic() - started:.0f}s", file=sys.stderr, flush=True)
    for name, values in losses.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {name} losses")
    return losses


def requested_levels(args, family):
    """Requested log-SNR points and the attack's (left, center, right) or single query points."""
    low, high = (family.grid.min(), family.grid.max()) if hasattr(family, "grid") else (-np.inf, np.inf)
    delta = HALF_WIDTH if getattr(args, "delta", None) is None else args.delta
    native = getattr(args, "timesteps", None)
    if native is not None:
        if args.attack != "tsc":
            raise ValueError("--timesteps is a TSC option (RMC uses --timestep)")
        if getattr(args, "delta", None) is not None:
            raise ValueError("--timesteps already defines the window, so omit --delta")
        if len(native) != 3:
            raise ValueError("--timesteps requires three values: FIRST CENTER LAST")
        levels, logsnrs = family.native_levels(native)
        differences = np.diff(levels)
        if not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError("--timesteps requires three distinct, strictly increasing or decreasing native points")
        group = tuple(map(float, logsnrs))
        return sorted(group), {group[1]: group}
    explicit = getattr(args, "logsnrs", None)
    if explicit is not None:
        if args.attack != "tsc":
            raise ValueError("--logsnrs is a TSC option (RMC uses --logsnr)")
        if getattr(args, "delta", None) is not None:
            raise ValueError("--logsnrs already defines the spacing, so omit --delta")
        if len(explicit) != 3 or not np.isfinite(explicit).all():
            raise ValueError("--logsnrs requires three finite values: LEFT CENTER RIGHT")
        left, center, right = map(float, explicit)
        if not left < center < right or not np.isclose(center - left, right - center, rtol=1e-7, atol=1e-9):
            raise ValueError("--logsnrs requires strictly increasing, equally spaced points")
        if left < low or right > high:
            raise ValueError(f"TSC query points must lie in scheduler range [{low:.2f}, {high:.2f}]")
        return [left, center, right], {center: (left, center, right)}
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("--delta must be finite and positive")
    if args.sweep is not None:
        if not args.sweep:
            points = SWEEP
        elif len(args.sweep) == 3:
            start, stop, step = args.sweep
            if not np.isfinite([start, stop, step]).all() or step <= 0 or stop < start:
                raise ValueError("Sweep requires finite START <= STOP and positive STEP")
            points = np.round(np.arange(start, stop + step * 1e-9, step), 6)
        else:
            raise ValueError("Give --sweep alone or --sweep START STOP STEP")
    elif args.timestep is not None:
        _, realized = family.native_levels([args.timestep])
        points = [float(realized[0])]
    else:
        point = args.logsnr
        if point is None:
            point = family.default_center if args.attack == "tsc" else 1.0
        points = [point]
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Query log-SNR values must be finite and nonempty")
    if args.attack == "tsc":
        valid = [c for c in points if low <= c - delta and c + delta <= high]
        queries = {float(c): tuple(float(c + offset) for offset in (-delta, 0, delta)) for c in valid}
    else:
        queries = {float(point): (float(point),) for point in points if low <= point <= high}
    if not queries:
        raise ValueError(f"No valid {args.attack.upper()} queries in scheduler range [{low:.2f}, {high:.2f}]"
                         + (f" with a {delta:g} log-SNR window per side" if args.attack == "tsc" else ""))
    requested = sorted({point for group in queries.values() for point in group})
    return requested, queries


def query_levels(args, family):
    """Resolve query points, preserving exact native indices/times when supplied."""
    requested, queries = requested_levels(args, family)
    native = getattr(args, "timesteps", None)
    if native is not None or (args.timestep is not None and args.attack == "rmc"):
        values, logsnrs = family.native_levels(native if native is not None else [args.timestep])
        by_logsnr = dict(zip(map(float, logsnrs), values))
        levels, realized = [by_logsnr[point] for point in requested], np.asarray(requested)
    else:
        levels, realized = family.levels(requested)
        if args.timestep is not None:
            # TSC uses the exact native center, with endpoints chosen in log-SNR.
            values, logsnrs = family.native_levels([args.timestep])
            index = requested.index(float(logsnrs[0]))
            levels[index], realized[index] = values[0], logsnrs[0]
    column = {point: index for index, point in enumerate(requested)}
    if args.attack == "tsc":
        valid = {key: group for key, group in queries.items()
                 if len({levels[column[point]] for point in group}) == 3}
        if not valid or (args.sweep is None and len(valid) != len(queries)):
            raise ValueError("TSC requires three distinct scheduler query points")
        if len(valid) != len(queries):
            queries = valid
            requested = sorted({point for group in queries.values() for point in group})
            levels, realized = family.levels(requested)
    return requested, queries, levels, realized


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.attack == "rmc" and args.reference is None and args.family != "sd14":
        p.error("RMC needs --reference, the checkpoint from before training or fine-tuning")
    if args.attack == "tsc" and args.reference is not None:
        p.error("--reference is an RMC option")
    schedule_options = (args.beta_schedule, args.diffusion_steps, args.beta_start, args.beta_end)
    if any(value is not None for value in schedule_options):
        if args.family != "ddpm":
            p.error("DDPM schedule options apply only to --family ddpm")
        if args.config:
            p.error("Choose --config or explicit DDPM schedule options")
    if args.family == "ddpm" and not args.config and args.beta_schedule is None:
        p.error("DDPM needs --beta-schedule (architecture is read from the checkpoint), or --config")
    if args.model_path and args.family != "sd14":
        p.error("--model-path applies only to --family sd14")
    if args.family == "sd14" and not args.model_path:
        args.model_path = SD14_BASE
    if (args.members is None) != (args.nonmembers is None):
        p.error("Give both --members and --nonmembers")
    if sum((args.dataset is not None, args.members is not None, args.half is not None)) != 1:
        p.error("Choose --dataset, --members with --nonmembers, or --half")
    if args.half and args.family in ("ddpm", "sd14") and not args.config:
        p.error("--half for DDPM/SD1.4 needs --config (or use --dataset or a pair of image sets)")
    if args.config and (args.family == "edm" or (args.family == "sd14" and not args.half)):
        p.error("--config selects a DDPM recipe, or the SD1.4 recipe whose --half data to use")
    if (args.data_root or args.data_path) and not args.half:
        p.error("--data-root and --data-path locate the data for --half")
    if args.delta is not None and (args.attack != "tsc" or not np.isfinite(args.delta) or args.delta <= 0):
        p.error("--delta is a TSC option and must be finite and positive")
    if args.logsnrs is not None:
        if args.attack != "tsc":
            p.error("--logsnrs is a TSC option (RMC uses --logsnr)")
        if args.delta is not None:
            p.error("--logsnrs already defines the spacing, so omit --delta")
        left, center, right = args.logsnrs
        if (not np.isfinite(args.logsnrs).all() or not left < center < right
                or not np.isclose(center - left, right - center, rtol=1e-7, atol=1e-9)):
            p.error("--logsnrs requires three finite, strictly increasing, equally spaced points")
    if args.timesteps is not None:
        if args.attack != "tsc":
            p.error("--timesteps is a TSC option (RMC uses --timestep)")
        if args.delta is not None:
            p.error("--timesteps already defines the window, so omit --delta")
        differences = np.diff(args.timesteps)
        if not (np.all(differences > 0) or np.all(differences < 0)):
            p.error("--timesteps requires three strictly increasing or decreasing native points")
    native = args.timesteps if args.timesteps is not None else ([args.timestep] if args.timestep is not None else None)
    if native is not None:
        if not np.isfinite(native).all():
            p.error("Native timesteps must be finite")
        if args.family == "edm":
            if not np.all((np.asarray(native) > 0) & (np.asarray(native) < 1)):
                p.error("EDM native flow times must lie strictly between 0 and 1")
        elif any(value < 0 or not float(value).is_integer() for value in native):
            p.error("DDPM/SD1.4 native timesteps must be nonnegative integer indices")
    args.noises = (5 if args.attack == "tsc" else 1) if args.noises is None else args.noises
    args.batch_size = (8 if args.family == "sd14" else 128) if args.batch_size is None else args.batch_size
    if args.noises < 1 or args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        p.error("--noises, --batch-size, and --limit must be positive")
    if args.logsnr is not None and not np.isfinite(args.logsnr):
        p.error("--logsnr must be finite")
    if args.sweep is not None:
        if len(args.sweep) not in (0, 3):
            p.error("Give --sweep alone or --sweep START STOP STEP")
        if args.sweep and (not np.isfinite(args.sweep).all() or args.sweep[2] <= 0 or args.sweep[1] < args.sweep[0]):
            p.error("Sweep requires finite START <= STOP and positive STEP")
    outputs = [Path(path).expanduser().resolve() for path in (args.output, args.save_losses) if path]
    if len(set(outputs)) != len(outputs):
        p.error("--output and --save-losses must be different files")
    if any(path.exists() for path in outputs):
        p.error("Output already exists, so choose a new output filename")

    import torch
    from .models import DDPM, EDM, SD14

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = False
    try:
        if args.dataset is not None:
            members, nonmembers = dataset_sets(args.dataset)
        elif args.members is not None:
            sources = {}
            members, nonmembers = ImageSet.parse(args.members, sources), ImageSet.parse(args.nonmembers, sources)
        else:
            members, nonmembers = half_sets(args.family, args.half, config=args.config,
                                            data_root=args.data_root, data_path=args.data_path)
    except (ValueError, OSError, KeyError) as error:
        p.error(str(error))
    if overlap(members, nonmembers):
        p.error("Members and non-members share images")
    members.limit(args.limit), nonmembers.limit(args.limit)

    try:
        if args.family == "ddpm":
            family = DDPM(args.config, device, args.condition, beta_schedule=args.beta_schedule,
                          diffusion_steps=args.diffusion_steps, beta_start=args.beta_start, beta_end=args.beta_end)
        elif args.family == "edm":
            family = EDM(device, args.condition)
        else:
            family = SD14(args.model_path, device)
        models = {"target": family.load(args.checkpoint, args.weights)}
        if args.attack == "rmc":
            models["reference"] = family.load(args.reference, args.weights)
        requested, queries, levels, realized = query_levels(args, family)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        p.error(str(error))
    # Overlapping sweep windows reuse both requested points and scheduler levels.
    unique_levels = list(dict.fromkeys(levels))
    level_columns = {level: index for index, level in enumerate(unique_levels)}
    inverse = [level_columns[level] for level in levels]
    total_queries = len(unique_levels) * args.noises * len(models)
    mode = "sweep" if args.sweep is not None else "fixed"
    print(f"{args.attack.upper()} {mode}: {len(queries)} candidate(s), {total_queries} total queries per image",
          file=sys.stderr, flush=True)

    started = time.monotonic()
    losses = {split: collect(family, models, image_set, split, unique_levels, args, torch)
              for split, image_set in (("member", members), ("nonmember", nonmembers))}
    losses = {split: {name: values[:, inverse, :] for name, values in by_model.items()}
              for split, by_model in losses.items()}
    column = {point: index for index, point in enumerate(requested)}
    logs = {split: {name: scores.log_loss(values) for name, values in by_model.items()}
            for split, by_model in losses.items()}

    def residuals(group):
        if args.attack == "tsc":
            left, center, right = (column[point] for point in group)
            values = [scores.tsc(logs[s]["target"], left, center, right) for s in ("member", "nonmember")]
        else:
            values = [scores.rmc(logs[s]["target"], logs[s]["reference"], column[group[0]])
                      for s in ("member", "nonmember")]
        return values

    key = "center" if args.attack == "tsc" else "logsnr"
    rows = [{key: float(point), **scores.metrics(*residuals(group))} for point, group in sorted(queries.items())]
    result = max(rows, key=lambda row: row["auc"]) if args.sweep is not None else rows[0]
    main_key = result[key]
    selected_scores = residuals(queries[main_key])
    report = {"attack": args.attack.upper(), "family": args.family, "mode": mode,
              "query_space": "timestep" if native is not None else "logsnr",
              "score_convention": "lower_is_member",
              "members": len(members), "nonmembers": len(nonmembers), "noises_per_level": args.noises,
              "queries_per_image": len(queries[main_key]) * args.noises * len(models),
              "total_queries_per_image": total_queries,
              "points": [{"requested_logsnr": point, "realized_logsnr": float(realized[column[point]]),
                          ("timestep" if hasattr(family, "grid") else "time"): levels[column[point]]}
                         for point in queries[main_key]],
              **result, "seed": args.seed, "seconds": round(time.monotonic() - started, 1)}
    if args.attack == "tsc" and args.timesteps is None:
        group = queries[main_key]
        report["delta"] = group[1] - group[0]
    if args.sweep is not None:
        report["sweep"] = rows
        report["best_auc"] = result
        report["selection"] = "best AUC on the supplied dataset"
    if args.timestep is not None:
        report["timestep"] = int(args.timestep) if hasattr(family, "grid") else args.timestep
    if args.timesteps is not None:
        report["timesteps"] = [int(value) if hasattr(family, "grid") else value for value in args.timesteps]

    print_report(report, key)
    if args.output:
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.save_losses:
        arrays = {f"{split}_{'losses' if name == 'target' else 'reference_losses'}": values
                  for split, by_model in losses.items() for name, values in by_model.items()}
        path = Path(args.save_losses).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays, member_scores=selected_scores[0], nonmember_scores=selected_scores[1],
                                requested_logsnr=np.asarray(requested), realized_logsnr=np.asarray(realized),
                                member_indices=members.indices, nonmember_indices=nonmembers.indices,
                                report=json.dumps(report, allow_nan=False))
    return 0


def print_report(report, key):
    points = " / ".join(f"{row['requested_logsnr']:g} ({row['realized_logsnr']:.3f})" for row in report["points"])
    print(f"{report['attack']} on {report['family']}: {report['members']} members, {report['nonmembers']} non-members, "
          f"{report['queries_per_image']} queries per image")
    native_key = "time" if report["family"] == "edm" else "timestep"
    print(f"Native {native_key}: " + " / ".join(f"{row[native_key]:g}" for row in report["points"]))
    print(f"log-SNR (realized): {points}")
    print(f"AUC {100 * report['auc']:.2f}   ASR {100 * report['asr']:.2f}   TPR@1%FPR {100 * report['tpr_at_1pct_fpr']:.2f}")
    if "sweep" in report:
        print(f"Sweep used {report['total_queries_per_image']} total queries per image. Headline metrics use the best-AUC setting.")
        print(f"\n{key:>8}  {'AUC':>6}  {'ASR':>6}  {'TPR@1%':>6}")
        for row in report["sweep"]:
            print(f"{row[key]:8.2f}  {100 * row['auc']:6.2f}  {100 * row['asr']:6.2f}  {100 * row['tpr_at_1pct_fpr']:6.2f}")
        best = report["best_auc"]
        print(f"best AUC at {key} {best[key]:g}: AUC {100 * best['auc']:.2f}, ASR {100 * best['asr']:.2f}, "
              f"TPR@1%FPR {100 * best['tpr_at_1pct_fpr']:.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
