# Run TSC or RMC

TSC needs only the attacked model. RMC also needs the model it was fine-tuned
from. By default, this is SD1.4. Both report AUC, ASR, and TPR at 1% FPR on known
members (training images) and non-members (held-out images). Install the
requirements of the model family ([setup](../training/README.md#setup)) and run
from the repository root.

## Models trained with this repository

After training a model with the [training guide](../training/README.md), pass
its final checkpoint. The paths below are where the training launcher writes
the final checkpoint of a second-half model. Its EMA weights are used by default.

`--half` names the half the model was trained on. That half becomes the
members and the other half the non-members. DDPM and SD1.4 also need the
training recipe (`--config`). Data is read from the default `data/`
locations in the [training guide](../training/README.md#data). To read it
from elsewhere, pass `--data-root` (DDPM, SD1.4) or `--data-path` (EDM).

```bash
# DDPM
python -m attacks tsc --family ddpm --config cifar10 --half second \
  --checkpoint artifacts/training/ddpm/cifar10/second/ckpt-step800000.pt

# EDM flow matching: prediction and loss settings come from the checkpoint
python -m attacks tsc --family edm --half second \
  --checkpoint artifacts/training/edm/uncond_aug/second/pred-v_loss-v/weights-epoch-004000.pt

# SD1.4: TSC, and RMC against the base SD v1.4
python -m attacks tsc --family sd14 --config coco_real --half second \
  --checkpoint artifacts/training/sd14/coco_real/second/milestone-checkpoint-50000
python -m attacks rmc --family sd14 --config coco_real --half second \
  --checkpoint artifacts/training/sd14/coco_real/second/milestone-checkpoint-50000
```

SD1.4 loads the VAE, text encoder, scheduler, and RMC reference from
`--model-path`, by default `CompVis/stable-diffusion-v1-4`. `--reference`
gives RMC another reference model: an SD1.4 checkpoint, or a checkpoint the model 
was fine-tuned from.

## Other models and data

Give the images as a pair of sets:

```bash
python -m attacks rmc --family sd14 --checkpoint finetuned/ \
  --members images/train --nonmembers images/heldout
```

Each set is `SOURCE[@INDICES]`. A source can be:

- An image folder, read in sorted path order, or in row order when it
  contains a `metadata.jsonl` with `file_name` and an optional integer
  `label`.
- A CIFAR-10/100 or STL-10 download directory (the training set, or STL-10's
  unlabeled split).
- An EDM dataset ZIP.

A folder without `metadata.jsonl` whose images all sit in subfolders is
labelled by top-level subfolder, like torchvision's `ImageFolder`.
Class-conditional models need labels that match training. INDICES selects
images by position: `split.npz:key`, a `.npy`, `.txt` or `.json` file, or
`start:stop`.

An SD1.4 checkpoint can be a training milestone or a Diffusers pipeline or
UNet directory. An EDM checkpoint must be a `weights-epoch-*.pt` file written
by [the EDM trainer](../training/README.md#edm-flow-matching), which records
the prediction and loss settings.

The same sets can be described in a JSON file and passed with `--dataset
evaluation.json`. Paths are relative to the JSON file:

```json
{
  "members": {"source": "images", "indices": "split.npz:members"},
  "nonmembers": {"source": "images", "indices": [5, 6, 7]}
}
```

A DDPM without a recipe file needs its training noise schedule, which cannot
be read from the weights: `--beta-schedule linear` (with `--beta-start` and
`--beta-end`, defaults 1e-4 and 0.02) or `--beta-schedule cosine`. The UNet
architecture and the number of diffusion steps are read from the checkpoint.

```bash
python -m attacks tsc --family ddpm --beta-schedule linear \
  --checkpoint model.pt --dataset evaluation.json
```

## Query points

Without a query option, each attack runs once at the paper's settings. RMC
queries log-SNR 1. TSC queries a center and the points `delta` = 2.5 below and
above it, with the center chosen per family on shadow models:

| Family | TSC center log-SNR |
| --- | --- |
| DDPM | 0.75 |
| SD1.4 | 0.25 |
| EDM | by loss target: x −0.5, v 0.5, e 1.75 |

Choose at most one of the following options:

| Option | TSC | RMC |
| --- | --- | --- |
| `--logsnr L` (or `--center L`) | Center L, endpoints at L ± `delta` (set with `--delta D`) | Query at L |
| `--logsnrs A B C` | Three equally spaced points, center B | — |
| `--timestep T` | Native center T, endpoints at its log-SNR ± `delta` | Query at native T |
| `--timesteps T1 T2 T3` | Exactly these native points, center T2 | — |
| `--sweep [START STOP STEP]` | Every center in the range | Every level in the range |

Native timesteps are zero-based schedule indices for DDPM and SD1.4, and flow
times `0 < t < 1` for EDM, where `z = t * image + (1 - t) * noise` and
`t = sigmoid(logsnr / 2)`. Log-SNR values map to the nearest DDPM or SD1.4
schedule index, and the report lists the realized values.

A sweep uses −4 to 4 in steps of 0.25 by default. TSC centers whose window
leaves the schedule are skipped. It reports every candidate and takes its
headline metrics from the best AUC on the supplied data. That selection uses
membership labels. To choose settings fairly, sweep a shadow model, then run
the target at the chosen fixed setting.

## Results

Lower scores indicate members for both attacks. The log-loss is
`log(mean(loss over noise draws) + 1e-12)`. TSC scores the center log-loss
minus the mean of the two endpoint log-losses. RMC scores the target log-loss
minus the reference log-loss. TSC averages 5 noise draws per point (15 model
queries per image). RMC uses 1 draw per model (2 queries). `--noises` changes
the number of draws.

Results are printed. `--output report.json` also saves the metrics and query
points. `--save-losses losses.npz` saves the raw losses as `[images, points,
draws]`, the per-image scores of the reported setting, and the image indices.
Existing files are never overwritten.

## Other options

| Option | Default |
| --- | --- |
| `--weights ema\|raw` | `ema` |
| `--condition label\|none` | `label`: ground-truth classes. `none` uses the dropped label of EDM models trained with label dropout. |
| `--diffusion-steps N` | DDPM without a recipe: check the step count read from the checkpoint |
| `--limit N` | All images. Otherwise the first N members and the first N non-members. |
| `--seed N` | 0. Each image's noise draws are fixed by the seed and shared across query points and models. |
| `--batch-size N` | 8 for SD1.4, 128 otherwise |
| `--device DEVICE` | CUDA if available |

`python -m attacks --help` lists every option.
