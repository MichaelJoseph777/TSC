# Third-party code

The trainers in [training/](training) are adapted from the projects below.
Original copyright headers and license texts are kept in each family's
directory, and each license applies to the code taken from that project.
The DDPM and SD1.4 code came to this repository through the FreMIA codebase
(*Enhancing Membership Inference Attacks on Diffusion Models from a
Frequency-Domain Perspective*, ICML 2026). The attacks in
[attacks/](attacks) are new code that loads the model implementations below.

## DDPM: SecMI and pytorch-ddpm

- Upstream: [SecMI](https://github.com/jinhaoduan/SecMI), built on
  [pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm).
- License: SecMI is MIT licensed (copyright 2024 Jinhao Duan), see
  [training/ddpm/LICENSE](training/ddpm/LICENSE). pytorch-ddpm is WTFPL licensed.
- Imported: the trainer, UNet, diffusion implementation, dataset wrappers,
  and SecMI's CIFAR and STL10 membership splits. The splits are unchanged.
- Changes: class conditioning, the cosine schedule, configurable paths,
  optional seeding, CPU/CUDA device selection, and guards against the
  unsupported evaluation path and non-epsilon targets. The launcher, recipes,
  and class-balanced Tiny ImageNet split are new.

```bibtex
@inproceedings{duan2023diffusion,
  title={Are Diffusion Models Vulnerable to Membership Inference Attacks?},
  author={Duan, Jinhao and Kong, Fei and Wang, Shiqi and Shi, Xiaoshuang and Xu, Kaidi},
  booktitle={International Conference on Machine Learning},
  pages={8717--8730},
  year={2023}
}
```

## EDM: NVIDIA

- Upstream: <https://github.com/NVlabs/edm>
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0, see
  [training/edm/LICENSE.txt](training/edm/LICENSE.txt). NVIDIA's source
  headers are kept, and the files under `training/edm/` may be used for
  non-commercial purposes only.
- Changes: linear flow matching (`FMPrecond`, `FMLoss`) with independent
  `x`/`e`/`v` prediction and loss targets, which is not the original EDM
  objective, the per-class half split, weight snapshots that record the
  network and loss settings, explicit trusted checkpoint loading for PyTorch
  2.8, and validation of resume state. The launcher and recipes are new.

```bibtex
@inproceedings{karras2022elucidating,
  title={Elucidating the Design Space of Diffusion-Based Generative Models},
  author={Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## SD1.4: CLiD and Hugging Face Diffusers

- Upstream: <https://github.com/zhaisf/CLiD> (commit
  `b108e70b0fbca23bea6e2fa052e48ee032bb8698`), which adapts the Hugging Face
  Diffusers `train_text_to_image.py` example.
- License: Apache License 2.0, see [training/sd14/LICENSE](training/sd14/LICENSE).
  The Hugging Face copyright notice is kept in the trainer.
- Changes: milestone checkpoints, atomic checkpoint saves, optional caption
  dropout (disabled in every recipe), and a non-finite loss guard. The
  launcher, recipes, and data preparation script are new, and the recipes
  follow CLiD's fine-tuning settings.
- Data: the MS-COCO split for both halves is CLiD's
  [`zsf/COCO_MIA_ori_split1`](https://huggingface.co/datasets/zsf/COCO_MIA_ori_split1)
  (MIT license). `training/sd14/prepare_data.py` downloads it, and it is
  not redistributed here.

```bibtex
@article{zhai2024membership,
  title={Membership inference on text-to-image diffusion models via conditional likelihood discrepancy},
  author={Zhai, Shengfang and Chen, Huanran and Dong, Yinpeng and Li, Jiajun and Shen, Qingni and Gao, Yansong and Su, Hang and Liu, Yang},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={74122--74146},
  year={2024}
}
```
