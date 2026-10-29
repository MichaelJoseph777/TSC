"""Model families: noise levels, conditioning, and per-image denoising losses.

Each family maps a requested log-SNR to a noise level its model was trained on
and returns the squared reconstruction error of a noised batch, following the
paper's evaluation code: DDPM and SD1.4 measure the clean-image (x0) error,
summed over pixels or latents, and EDM flow matching measures the error in the
model's training-loss space, averaged over pixels.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from numbers import Real
from pathlib import Path
import re
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_tensor(image, resolution):
    """torchvision Resize(resolution) and CenterCrop(resolution), scaled to [-1, 1]."""
    width, height = image.size
    if (width, height) != (resolution, resolution):
        if width <= height:
            size = (resolution, int(resolution * height / width))
        else:
            size = (int(resolution * width / height), resolution)
        image = image.resize(size, Image.BILINEAR)
        left = int(round((image.width - resolution) / 2.0))
        top = int(round((image.height - resolution) / 2.0))
        image = image.crop((left, top, left + resolution, top + resolution))
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(array).permute(2, 0, 1)


def nearest_levels(grid, requested):
    """Map requested log-SNR values to the nearest discrete scheduler timesteps."""
    grid = np.asarray(grid, dtype=np.float64)
    requested = np.asarray(requested, dtype=np.float64)
    if grid.ndim != 1 or not grid.size or not np.isfinite(grid).all():
        raise ValueError("Scheduler log-SNR grid must be a nonempty finite vector")
    if requested.ndim != 1 or not requested.size or not np.isfinite(requested).all():
        raise ValueError("Requested log-SNR values must be a nonempty finite vector")
    low, high = grid.min(), grid.max()
    if requested.min() < low - 1e-6 or requested.max() > high + 1e-6:
        raise ValueError(f"Requested log-SNR must lie in the scheduler range [{low:.2f}, {high:.2f}]")
    timesteps = np.abs(grid[:, None] - requested[None, :]).argmin(axis=0)
    return timesteps.tolist(), grid[timesteps]


def _native_vector(requested):
    values = np.asarray(requested, dtype=object)
    if values.ndim != 1 or not values.size or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in values):
        raise ValueError("Native query values must be a nonempty vector of finite real numbers")
    try:
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Native query values must be finite real numbers") from error
    if not np.isfinite(values).all():
        raise ValueError("Native query values must be finite real numbers")
    return values


def discrete_levels(grid, requested):
    """Keep exact zero-based scheduler indices and return their log-SNR values."""
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or not grid.size or not np.isfinite(grid).all():
        raise ValueError("Scheduler log-SNR grid must be a nonempty finite vector")
    values = _native_vector(requested)
    if (values != np.floor(values)).any() or (values < 0).any() or (values >= len(grid)).any():
        raise ValueError(f"Native timesteps must be integer indices between 0 and {len(grid) - 1}")
    indices = values.astype(np.int64)
    return indices.tolist(), grid[indices]


class DDPM:
    """Epsilon-prediction DDPMs trained with training/ddpm."""

    resolution = 32
    default_center = 0.75

    def __init__(self, config, device, condition="label", *, beta_schedule=None,
                 diffusion_steps=None, beta_start=None, beta_end=None):
        self.device, self.condition = device, condition
        self.classes = None
        self._config_free = config is None
        self._architecture = None
        self._diffusion_steps = diffusion_steps
        overrides = (beta_schedule, diffusion_steps, beta_start, beta_end)
        if not self._config_free and any(value is not None for value in overrides):
            raise ValueError("Use a DDPM config or explicit diffusion settings, not both")
        if self._config_free:
            if beta_schedule not in ("linear", "cosine"):
                raise ValueError("DDPM without a config needs an explicit linear or cosine beta_schedule, "
                                 "since the noise schedule cannot be inferred from weights")
            if beta_schedule == "cosine" and (beta_start is not None or beta_end is not None):
                raise ValueError("DDPM beta_start and beta_end apply only to the linear beta schedule")
            if diffusion_steps is not None and (isinstance(diffusion_steps, bool)
                    or not isinstance(diffusion_steps, (int, np.integer)) or diffusion_steps < 1):
                raise ValueError("DDPM diffusion_steps must be a positive integer")
            start = 1e-4 if beta_start is None else beta_start
            end = 0.02 if beta_end is None else beta_end
            if not np.isfinite([start, end]).all() or not 0 < start <= end < 1:
                raise ValueError("DDPM beta endpoints must satisfy 0 < beta_start <= beta_end < 1")
            self.flags = dict(beta_schedule=beta_schedule, beta_1=start, beta_T=end,
                              mean_type="epsilon", dropout=0.0)
            return
        path = Path(config) if str(config).endswith(".json") else ROOT / "training/ddpm/configs" / f"{config}.json"
        self.flags = json.loads(path.read_text())["flags"]
        if self.flags.get("mean_type", "epsilon") not in ("epsilon", "eps"):
            raise ValueError("DDPM scoring requires an epsilon-prediction training recipe")
        self.alpha_bar, self.grid = self._schedule(self.flags)
        if self.flags.get("conditional"):
            self.classes = self.flags.get("num_classes") or {"CIFAR10": 10, "CIFAR100": 100,
                                                             "TINY-IN": 200}[self.flags["dataset"].upper()]
            if condition != "label":
                raise ValueError("Class-conditional DDPMs have no unconditional label, so use --condition label")

    @staticmethod
    def _schedule(flags):
        diffusion = _module("ddpm_diffusion", ROOT / "training/ddpm/diffusion.py")
        betas = diffusion.make_beta_schedule(flags.get("beta_schedule", "linear"),
                                             flags["beta_1"], flags["beta_T"], flags["T"])
        alpha_bar = torch.cumprod(1 - betas, dim=0).numpy()
        return alpha_bar, np.log(alpha_bar) - np.log1p(-alpha_bar)

    @staticmethod
    def _checkpoint_state(payload, weights):
        if not isinstance(payload, dict) or not payload:
            raise ValueError("Expected a DDPM checkpoint dictionary or tensor state dictionary")
        if all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in payload.items()):
            state = payload
        else:
            key = "ema_model" if weights == "ema" else "net_model"
            state = payload.get(key)
            if not isinstance(state, dict) or not state:
                raise ValueError(f"DDPM checkpoint does not contain {key} weights")
        cleaned = {}
        for key, value in state.items():
            if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                raise ValueError("DDPM state dictionary must contain named tensors only")
            key = key.removeprefix("module.")
            if key in cleaned:
                raise ValueError(f"DDPM state dictionary repeats {key}")
            cleaned[key] = value
        return cleaned

    @staticmethod
    def _infer_architecture(state):
        """Recover only the UNet topology implemented in training/ddpm/model.py."""
        def shape(key, rank):
            tensor = state.get(key)
            if tensor is None or tensor.ndim != rank:
                raise ValueError(f"Unsupported DDPM checkpoint: missing or invalid {key}")
            return tuple(tensor.shape)

        steps, channels = shape("time_embedding.timembedding.0.weight", 2)
        if steps < 1 or channels < 32 or channels % 32:
            raise ValueError("Unsupported DDPM timestep embedding or base channel count")
        if shape("head.weight", 4) != (channels, 3, 3, 3):
            raise ValueError("DDPM checkpoint must use the vendored RGB UNet head")
        blocks = sorted({int(match[1]) for key in state
                         if (match := re.match(r"downblocks\.(\d+)\.", key))})
        if not blocks or blocks != list(range(len(blocks))):
            raise ValueError("Unsupported DDPM checkpoint: downsampling blocks are missing or noncontiguous")
        groups, group = [], []
        for index in blocks:
            prefix = f"downblocks.{index}."
            if prefix + "block1.2.weight" in state:
                out_channels, _, height, width = shape(prefix + "block1.2.weight", 4)
                if (height, width) != (3, 3) or out_channels % channels:
                    raise ValueError("Unsupported DDPM residual block channel dimensions")
                group.append((out_channels // channels, prefix + "attn.proj_q.weight" in state))
            elif prefix + "main.weight" in state:
                if not group:
                    raise ValueError("Unsupported DDPM checkpoint: empty resolution level")
                groups.append(group)
                group = []
            else:
                raise ValueError(f"Unsupported DDPM checkpoint: unrecognized downsampling block {index}")
        if not group:
            raise ValueError("Unsupported DDPM checkpoint: trailing downsampling block")
        groups.append(group)
        num_res_blocks = len(groups[0])
        if any(len(level) != num_res_blocks or len(set(level)) != 1 or level[0][0] < 1 for level in groups):
            raise ValueError("Unsupported DDPM checkpoint: inconsistent residual blocks within resolution levels")
        classes = None
        if "label_embedding.weight" in state:
            classes, width = shape("label_embedding.weight", 2)
            if classes < 1 or width != 4 * channels:
                raise ValueError("Unsupported DDPM class embedding dimensions")
        return dict(T=steps, ch=channels, ch_mult=[level[0][0] for level in groups],
                    attn=[index for index, level in enumerate(groups) if level[0][1]],
                    num_res_blocks=num_res_blocks, num_classes=classes)

    def load(self, path, weights="ema"):
        if weights not in ("ema", "raw"):
            raise ValueError("DDPM weights must be ema or raw")
        state = self._checkpoint_state(torch.load(path, map_location="cpu", weights_only=True), weights)
        flags, classes = self.flags, self.classes
        architecture = None
        if self._config_free:
            architecture = self._infer_architecture(state)
            if self._diffusion_steps is not None and architecture["T"] != self._diffusion_steps:
                raise ValueError(f"DDPM diffusion_steps={self._diffusion_steps} conflicts with checkpoint T={architecture['T']}")
            if self._architecture is not None and architecture != self._architecture:
                raise ValueError("DDPM target and reference checkpoints must have the same UNet architecture")
            classes = architecture["num_classes"]
            if classes is not None and self.condition != "label":
                raise ValueError("Class-conditional DDPMs have no unconditional label, so use --condition label")
            flags = {**self.flags, **architecture, "conditional": classes is not None}
        model_module = _module("ddpm_model", ROOT / "training/ddpm/model.py")
        model = model_module.UNet(T=flags["T"], ch=flags["ch"], ch_mult=flags["ch_mult"],
                                  attn=flags["attn"], num_res_blocks=flags["num_res_blocks"],
                                  dropout=flags["dropout"], num_classes=classes)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError(f"Checkpoint does not match the supported DDPM UNet: {error}") from error
        if self._config_free and self._architecture is None:
            alpha_bar, grid = self._schedule(flags)
            self.flags, self.classes = flags, classes
            self.alpha_bar, self.grid = alpha_bar, grid
            self._architecture = architecture
        return model.to(self.device).eval().requires_grad_(False)

    def levels(self, requested):
        if not hasattr(self, "grid"):
            raise ValueError("Load the target DDPM checkpoint before querying its schedule")
        return nearest_levels(self.grid, requested)

    def native_levels(self, requested):
        if not hasattr(self, "grid"):
            raise ValueError("Load the target DDPM checkpoint before querying its schedule")
        return discrete_levels(self.grid, requested)

    def prepare(self, images, labels, generators):
        if self.classes is not None and any(label is None for label in labels):
            raise ValueError("A class-conditional DDPM needs labelled images")
        if self.classes is not None and any(label < 0 or label >= self.classes for label in labels):
            raise ValueError(f"DDPM class labels must lie in [0, {self.classes - 1}]")
        condition = torch.tensor(labels, device=self.device, dtype=torch.long) if self.classes is not None else None
        return images.to(self.device), condition

    def loss(self, model, clean, condition, noise, timestep):
        a = float(np.sqrt(self.alpha_bar[timestep]))
        s = float(np.sqrt(1 - self.alpha_bar[timestep]))
        noisy = a * clean + s * noise
        times = torch.full((len(clean),), timestep, device=self.device, dtype=torch.long)
        predicted = model(noisy, times, condition).float()
        reconstructed = (noisy - s * predicted) / a
        return (reconstructed - clean).square().flatten(1).sum(1)


class EDM:
    """Linear flow-matching models trained with training/edm."""

    resolution = 32
    # Shadow-selected TSC centers from the paper, by training-loss space.
    centers = {"x": -0.5, "v": 0.5, "e": 1.75}

    def __init__(self, device, condition="label"):
        sys.path.insert(0, str(ROOT / "training/edm"))
        self.device = device
        self.condition = condition
        self.loss_space = self.label_dim = self.t_eps = None

    @property
    def default_center(self):
        return self.centers[self.loss_space]

    def load(self, path, weights="ema"):
        import dnnlib

        payload = torch.load(path, map_location="cpu", weights_only=False)  # trusted EDM checkpoints only
        if not isinstance(payload, dict):
            raise ValueError("Expected an EDM checkpoint dictionary with model and loss metadata")
        state = payload.get("ema_state_dict" if weights == "ema" else "net_state_dict",
                            payload.get("ema" if weights == "ema" else "net"))
        if state is None:
            raise ValueError("Checkpoint does not contain the selected model weights")
        loss = payload.get("loss_kwargs", {})
        settings = dict(prediction=loss.get("prediction"), loss_space=loss.get("loss_target"),
                        t_eps=float(loss.get("t_eps", 0.05)))
        for key in ("prediction", "loss_space"):
            settings[key] = "e" if settings[key] == "eps" else settings[key]
            if settings[key] not in ("x", "e", "v"):
                raise ValueError("EDM checkpoints must record x/e/v prediction and loss_target metadata")
        if not np.isfinite(settings["t_eps"]) or settings["t_eps"] <= 0:
            raise ValueError("EDM t_eps must be finite and positive")
        if isinstance(state, torch.nn.Module):
            model = copy.deepcopy(state)
        else:
            if payload["network_kwargs"].get("class_name") != "training.networks.FMPrecond":
                raise ValueError("EDM scoring supports FMPrecond flow-matching checkpoints")
            label_dim = state["model.map_label.weight"].shape[1] if "model.map_label.weight" in state else 0
            model = dnnlib.util.construct_class_by_name(
                **payload["network_kwargs"], img_resolution=payload["dataset_kwargs"].get("resolution", 32),
                img_channels=3, label_dim=label_dim)
            model.load_state_dict(state, strict=True)
        if type(model).__name__ != "FMPrecond" or (model.img_resolution, model.img_channels) != (32, 3):
            raise ValueError("EDM scoring requires a 32x32 RGB FMPrecond model")
        if self.label_dim is not None:
            if model.label_dim != self.label_dim:
                raise ValueError("Target and reference models must have matching class conditioning")
            if settings["t_eps"] != self.t_eps:
                raise ValueError("Target and reference models must use the same t_eps")
        model.attack_settings = settings
        # A reference model is scored in the loss space of the first model loaded (the target).
        if self.label_dim is None:
            self.loss_space = settings["loss_space"]
            self.label_dim = model.label_dim
            self.t_eps = settings["t_eps"]
        return model.to(self.device).eval().requires_grad_(False)

    def levels(self, requested):
        requested = np.asarray(requested, dtype=np.float64)
        if requested.ndim != 1 or not requested.size or not np.isfinite(requested).all():
            raise ValueError("Requested log-SNR values must be a nonempty finite vector")
        times = np.empty_like(requested)
        positive = requested >= 0
        times[positive] = 1 / (1 + np.exp(-requested[positive] / 2))
        exponent = np.exp(requested[~positive] / 2)
        times[~positive] = exponent / (1 + exponent)
        try:
            return self.native_levels(times)
        except ValueError as error:
            raise ValueError("EDM log-SNR values must map to float32 times strictly between zero and one") from error

    def native_levels(self, requested):
        times = _native_vector(requested)
        if not np.all((times > 0) & (times < 1)):
            raise ValueError("EDM native times must lie strictly between zero and one")
        times = times.astype(np.float32)
        if not np.all((times > 0) & (times < 1)):
            raise ValueError("EDM native times must remain strictly between zero and one in float32")
        realized = 2 * (np.log(times.astype(np.float64)) - np.log1p(-times.astype(np.float64)))
        return times.tolist(), realized

    def prepare(self, images, labels, generators):
        condition = None
        if self.condition == "label" and self.label_dim:
            if any(label is None for label in labels):
                raise ValueError("A class-conditional model needs labelled images, or use --condition none")
            if any(label < 0 or label >= self.label_dim for label in labels):
                raise ValueError(f"EDM class labels must lie in [0, {self.label_dim - 1}]")
            condition = torch.nn.functional.one_hot(torch.tensor(labels), self.label_dim).float().to(self.device)
        return images.to(self.device), condition

    def loss(self, model, clean, condition, noise, time_value):
        settings = model.attack_settings
        prediction, space, t_eps = settings["prediction"], self.loss_space, settings["t_eps"]
        if condition is not None and model.label_dim == 0:
            condition = None
        time = torch.full((len(clean),), time_value, device=self.device, dtype=torch.float32)
        t = time[:, None, None, None]
        noisy = t * clean + (1 - t) * noise
        predicted = model(noisy, time, condition).float()
        safe_t, safe_1t = t.clamp_min(t_eps), (1 - t).clamp_min(t_eps)
        if prediction != space:
            if space == "x":
                predicted = (noisy - safe_1t * predicted) / safe_t if prediction == "e" else noisy + safe_1t * predicted
            elif space == "e":
                predicted = (noisy - t * predicted) / safe_1t if prediction == "x" else noisy - t * predicted
            else:
                predicted = (predicted - noisy) / safe_1t if prediction == "x" else (noisy - predicted) / safe_t
        target = {"x": clean, "e": noise, "v": (clean - noisy) / safe_1t}[space]
        return (predicted - target).square().flatten(1).mean(1)


class SD14:
    """Stable Diffusion v1.4 fine-tunes from training/sd14, scored in VAE latent space."""

    resolution = 512
    default_center = 0.25

    def __init__(self, model_path, device):
        from diffusers import AutoencoderKL, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer

        self.model_path, self.device = model_path, device
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32
        scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        if scheduler.config.prediction_type != "epsilon":
            raise ValueError("SD1.4 scoring expects an epsilon-prediction scheduler")
        self.alpha_bar = scheduler.alphas_cumprod.double().numpy()
        self.grid = np.log(self.alpha_bar) - np.log1p(-self.alpha_bar)
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=self.dtype)
        self.vae.to(device).eval().requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=self.dtype)
        self.text_encoder.to(device).eval().requires_grad_(False)

    def load(self, path=None, weights="ema"):
        """Load a UNet from a training checkpoint, a pipeline, or (``base``/None) the base model."""
        from diffusers import UNet2DConditionModel
        from diffusers.training_utils import EMAModel

        path = None if path in (None, "base") else Path(path)
        if path is None:
            model = UNet2DConditionModel.from_pretrained(self.model_path, subfolder="unet", torch_dtype=self.dtype)
        elif weights == "ema" and (path / "unet_ema").is_dir():
            model = UNet2DConditionModel.from_pretrained(self.model_path, subfolder="unet")
            EMAModel.from_pretrained(str(path / "unet_ema"), UNet2DConditionModel).copy_to(model.parameters())
            model = model.to(self.dtype)
        elif (path / "unet").is_dir():
            model = UNet2DConditionModel.from_pretrained(path, subfolder="unet", torch_dtype=self.dtype)
        else:
            model = UNet2DConditionModel.from_pretrained(path, torch_dtype=self.dtype)
        return model.to(self.device).eval().requires_grad_(False)

    def levels(self, requested):
        return nearest_levels(self.grid, requested)

    def native_levels(self, requested):
        return discrete_levels(self.grid, requested)

    def prepare(self, images, labels, generators):
        posterior = self.vae.encode(images.to(self.device, self.dtype)).latent_dist
        sample = torch.stack([torch.randn(posterior.mean.shape[1:], generator=g) for g in generators])
        latents = posterior.mean + posterior.std * sample.to(self.device, self.dtype)
        latents = latents * float(self.vae.config.scaling_factor)
        tokens = self.tokenizer([""] * len(images), max_length=self.tokenizer.model_max_length, padding="max_length",
                                truncation=True, return_tensors="pt").input_ids.to(self.device)
        return latents, self.text_encoder(tokens)[0]

    def loss(self, model, clean, condition, noise, timestep):
        a = torch.tensor(float(np.sqrt(self.alpha_bar[timestep])), device=self.device, dtype=self.dtype)
        s = torch.tensor(float(np.sqrt(1 - self.alpha_bar[timestep])), device=self.device, dtype=self.dtype)
        noisy = a * clean + s * noise.to(self.dtype)
        predicted = model(noisy, timestep, condition).sample
        reconstructed = (noisy - s * predicted) / a
        return (reconstructed.float() - clean.float()).square().flatten(1).sum(1)
