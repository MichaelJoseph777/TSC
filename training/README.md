# Training

Recipes for every model in the paper. Each model trains on one half of a fixed
membership split, chosen with `--half first` or `--half second`, and the
other half is its held-out set. The paper attacks the second-half models and tunes
attack settings on the first-half (shadow) models. To attack a trained model,
see the [attack guide](../attacks/README.md).

## Setup

```bash
python -m pip install -r training/ddpm/requirements.txt   # Python 3.10+
python -m pip install -r training/edm/requirements.txt    # Python 3.9-3.11
python -m pip install -r training/sd14/requirements.txt   # Python 3.10+
```

## Data

The commands download everything into `data/`, where the launchers look by
default:

```bash
# DDPM: CIFAR-10, CIFAR-100, STL-10 (unlabeled), Tiny ImageNet
mkdir -p data/ddpm
python - <<'EOF'
from torchvision import datasets
datasets.CIFAR10("data/ddpm/cifar10", download=True)
datasets.CIFAR100("data/ddpm/cifar100", download=True)
datasets.STL10("data/ddpm/stl10", split="unlabeled", download=True)
EOF
curl -L -o data/ddpm/tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q data/ddpm/tiny-imagenet-200.zip -d data/ddpm

# EDM: CIFAR-100 in EDM's archive format
python training/edm/dataset_tool.py \
  --source data/ddpm/cifar100/cifar-100-python.tar.gz --dest data/cifar100-32x32.zip

# SD1.4: both halves of Pokémon, MS-COCO, and Flickr30k
huggingface-cli login
python training/sd14/prepare_data.py --output-dir data/sd14
```

The SD1.4 datasets come from Hugging Face
([Pokémon](https://huggingface.co/datasets/reach-vb/pokemon-blip-captions),
[MS-COCO, CLiD's split](https://huggingface.co/datasets/zsf/COCO_MIA_ori_split1),
[Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k)). Accept the
MS-COCO terms on its page before downloading. The membership splits index the
original files, so use these sources rather than re-encoded copies.

## Train

One command trains one model. For example, the second-half (target) models
for CIFAR-10 DDPM, CIFAR-100 EDM with v-prediction and v-loss, and SD1.4 on
MS-COCO:

```bash
python training/launch.py ddpm --config cifar10 --half second
python training/launch.py edm --config uncond_aug --half second --prediction v --loss-target v
python training/launch.py sd14 --config coco_real --half second
```

Swap in any recipe from the tables below, and use `--half first` for the
shadow model. Checkpoints go to `artifacts/training/<family>/<recipe>/<half>/`.
`--dry-run` prints the full trainer command (DDPM's `main.py`, the EDM
training options, or SD1.4's `accelerate launch train_text_to_image.py`)
without running it. `python training/launch.py <family> --help` lists the
options, such as other data or output locations, the seed, and resuming.
Hyperparameters are in the recipe files: `training/ddpm/configs/`,
`training/edm/configs/`, and `training/sd14/configs.json`.

### DDPM

| Recipe | Model | Final checkpoint |
| --- | --- | --- |
| `cifar10`, `cifar100` | unconditional | `ckpt-step800000.pt` |
| `stl10u`, `tinyin` | unconditional | `ckpt-step1600000.pt` |
| `cifar100_cond_linear`, `cifar100_cond_cosine` | class conditional, linear or cosine schedule | `ckpt-step800000.pt` |

### EDM flow matching

CIFAR-100 flow matching with the DDPM++ network. Choose the prediction and
loss target with `--prediction` and `--loss-target` (`x`, `e`, or `v`,
default `e`). Runs go to a `pred-<p>_loss-<l>/` subdirectory.

| Recipe | Model | Final checkpoint |
| --- | --- | --- |
| `uncond_noaug` | unconditional, no augmentation, any of the nine pairs | `weights-epoch-002000.pt` |
| `uncond_aug` | unconditional, with augmentation, any of the nine pairs | `weights-epoch-004000.pt` |
| `cond_aug_dropout` | class conditional, with augmentation and label dropout, `e/e` | `weights-epoch-004000.pt` |

### Stable Diffusion 1.4

Fine-tunes the SD v1.4 UNet on one caption per image, in an overfitting
setting and in a real-world setting with horizontal flips.

| Recipe | Images per half | Final checkpoint |
| --- | ---: | --- |
| `pokemon_overfit`, `pokemon_real` | 417 / 416 | `milestone-checkpoint-7500/` |
| `coco_overfit` | 2,500 | `milestone-checkpoint-150000/` |
| `coco_real` | 2,500 | `milestone-checkpoint-50000/` |
| `flickr_overfit` | 2,500 | `milestone-checkpoint-150000/` |
| `flickr_real` | 10,000 | `milestone-checkpoint-200000/` |

The base model defaults to `CompVis/stable-diffusion-v1-4`, and `--model-path`
selects another copy.
