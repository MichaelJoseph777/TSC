"""Member and non-member image sets: a source plus optional indices.

A set is written ``SOURCE[@INDICES]``. SOURCE is detected from its contents:

- a directory containing ``cifar-10-batches-py`` or ``cifar-100-python``: the
  CIFAR training set, with its labels (CIFAR-100 uses the fine labels)
- a directory containing ``stl10_binary``: the 100,000 unlabeled STL-10 images
- an EDM dataset ``.zip``: images in sorted name order, labels from ``dataset.json``
- a directory with ``metadata.jsonl``: its rows in order (``file_name`` and an
  optional integer ``label``)
- any other directory: every image below it in sorted path order, labelled by
  its top-level subdirectory (torchvision ``ImageFolder`` order).

INDICES selects images by position: ``file.npz:key``, ``file.npy``, ``file.txt``
(one per line), ``file.json`` (a list), or a ``start:stop`` range.
``dataset_sets`` reads these selections from a JSON dataset manifest, resolving
source and index-file paths relative to the manifest.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import pickle
import re
import zipfile

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ROOT = Path(__file__).resolve().parents[1]


class ImageSource:
    """Indexable RGB images with optional integer labels."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self._array = self._zip = None
        self.labels = None
        if self.path.suffix == ".zip" and self.path.is_file():
            self._open_zip()
        elif (self.path / "cifar-10-batches-py").is_dir():
            batches = [_unpickle(self.path / "cifar-10-batches-py" / f"data_batch_{i}") for i in range(1, 6)]
            self._array = np.concatenate([b["data"] for b in batches]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.labels = np.concatenate([b["labels"] for b in batches]).astype(np.int64)
        elif (self.path / "cifar-100-python").is_dir():
            batch = _unpickle(self.path / "cifar-100-python" / "train")
            self._array = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.labels = np.asarray(batch["fine_labels"], dtype=np.int64)
        elif (self.path / "stl10_binary").is_dir():
            raw = np.memmap(self.path / "stl10_binary" / "unlabeled_X.bin", dtype=np.uint8, mode="r")
            # Same orientation as torchvision's STL10 loader.
            self._array = raw.reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1)
        elif (self.path / "metadata.jsonl").is_file():
            with (self.path / "metadata.jsonl").open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.files = [self.path / row["file_name"] for row in rows]
            labelled = ["label" in row for row in rows]
            if any(labelled) and not all(labelled):
                raise ValueError("metadata.jsonl must provide a label for every image or none")
            if labelled and all(labelled):
                labels = [row["label"] for row in rows]
                if any(isinstance(label, bool) or not isinstance(label, int) or label < 0
                       for label in labels):
                    raise ValueError("metadata.jsonl labels must be nonnegative integers")
                self.labels = _integer_indices(labels, "metadata.jsonl labels")
        elif self.path.is_dir():
            self.files = sorted(p for p in self.path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
            tops = [p.relative_to(self.path).parts[0] for p in self.files]
            if all(len(p.relative_to(self.path).parts) > 1 for p in self.files):
                classes = {name: index for index, name in enumerate(sorted(set(tops)))}
                self.labels = np.asarray([classes[name] for name in tops], dtype=np.int64)
        else:
            raise ValueError(f"Unrecognized image source: {path}")
        if len(self) == 0:
            raise ValueError(f"No images found in {path}")

    def _open_zip(self):
        self._zip = zipfile.ZipFile(self.path)
        names = set(self._zip.namelist())
        self.files = sorted(n for n in names if Path(n).suffix.lower() in IMAGE_EXTENSIONS)
        if "dataset.json" in names:
            labels = json.loads(self._zip.read("dataset.json")).get("labels")
            if labels:
                labels = dict(labels)
                self.labels = _integer_indices([labels[name.replace("\\", "/")] for name in self.files], "Class labels")
                if np.any(self.labels < 0):
                    raise ValueError("Class labels must be nonnegative integers")

    def __len__(self):
        return len(self._array) if self._array is not None else len(self.files)

    def image(self, index):
        if self._array is not None:
            return Image.fromarray(np.ascontiguousarray(self._array[index]))
        if self._zip is not None:
            return Image.open(io.BytesIO(self._zip.read(self.files[index]))).convert("RGB")
        with Image.open(self.files[index]) as image:
            return image.convert("RGB")

    def label(self, index):
        return None if self.labels is None else int(self.labels[index])


def _unpickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _integer_indices(values, name="Indices"):
    """Validate index values before casting, so fractions cannot be truncated."""
    if isinstance(values, (list, tuple)) and any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError(f"{name} must contain integers, not booleans")
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional integer array")
    if values.dtype.kind in "iu":
        if int(values.min()) < -(2 ** 63) or int(values.max()) >= 2 ** 63:
            raise ValueError(f"{name} are outside the supported integer range")
    elif values.dtype.kind == "f":
        if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
                or np.any(values < -(2 ** 63)) or np.any(values >= 2 ** 63)):
            raise ValueError(f"{name} must contain finite integers")
    else:
        raise ValueError(f"{name} must contain integers")
    return values.astype(np.int64, copy=False)


def read_indices(spec):
    """Parse an INDICES spec into a 1-D integer array."""
    spec = str(spec)
    match = re.fullmatch(r"(\d+):(\d+)", spec)
    if match:
        return _integer_indices(np.arange(int(match[1]), int(match[2]), dtype=np.int64))
    path, key = spec, None
    if ".npz:" in spec:
        path, key = spec.rsplit(":", 1)
    path = Path(path).expanduser()
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise ValueError(f"No index arrays found in {path}")
            values = archive[key or archive.files[0]]
    elif path.suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix == ".json":
        values = json.loads(path.read_text())
    else:
        values = np.loadtxt(path, dtype=np.float64, ndmin=1)
    return _integer_indices(values)


class ImageSet:
    """A source restricted to selected indices."""

    def __init__(self, source, indices=None, name=""):
        self.source, self.name = source, name
        self.indices = np.arange(len(source), dtype=np.int64) if indices is None else _integer_indices(indices, name or "Indices")
        if self.indices.size == 0 or self.indices.min() < 0 or self.indices.max() >= len(source):
            raise ValueError(f"Indices are empty or outside the source: {name}")
        if np.unique(self.indices).size != self.indices.size:
            raise ValueError(f"Indices repeat an image: {name}")

    @classmethod
    def parse(cls, spec, sources=None):
        """Build a set from ``SOURCE[@INDICES]``, sharing already opened sources."""
        source_spec, _, index_spec = spec.rpartition("@") if "@" in spec else (spec, "", "")
        sources = {} if sources is None else sources
        key = str(Path(source_spec).expanduser().resolve())
        if key not in sources:
            sources[key] = ImageSource(source_spec)
        return cls(sources[key], read_indices(index_spec) if index_spec else None, spec)

    def __len__(self):
        return len(self.indices)

    def limit(self, count):
        if count:
            self.indices = self.indices[:count]
        return self


def overlap(members, nonmembers):
    """Count shared indices or selected image files, including path aliases."""
    left, right = members.source, nonmembers.source
    if left is right or left.path.resolve() == right.path.resolve():
        return int(np.intersect1d(members.indices, nonmembers.indices).size)
    if left._array is None and right._array is None and left._zip is None and right._zip is None:
        left_files = {left.files[int(index)].resolve() for index in members.indices}
        right_files = {right.files[int(index)].resolve() for index in nonmembers.indices}
        return len(left_files.intersection(right_files))
    return 0


def dataset_sets(manifest_path):
    """Read member/non-member sources and optional indices from a JSON manifest.

    Each of ``members`` and ``nonmembers`` contains ``source`` and optionally
    ``indices`` (an integer list or an existing INDICES spec). Relative paths
    refer to the manifest directory, regardless of the command's working directory.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {"members", "nonmembers"}:
        raise ValueError("Dataset manifest must contain members and nonmembers")
    sources = {}
    sets = []
    for name in ("members", "nonmembers"):
        item = manifest[name]
        if not isinstance(item, dict) or "source" not in item or set(item) - {"source", "indices"}:
            raise ValueError(f"Dataset {name} must contain source and optional indices")
        if not isinstance(item["source"], str) or not item["source"].strip():
            raise ValueError(f"Dataset {name} source must be a nonempty path")
        path = Path(item["source"]).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        key = str(path.resolve())
        if key not in sources:
            sources[key] = ImageSource(path)
        indices = item.get("indices")
        if isinstance(indices, str):
            if not re.fullmatch(r"(\d+):(\d+)", indices):
                index_path, archive_key = (indices.rsplit(":", 1)
                                           if ".npz:" in indices else (indices, None))
                index_path = Path(index_path).expanduser()
                if not index_path.is_absolute():
                    index_path = manifest_path.parent / index_path
                indices = str(index_path) + (f":{archive_key}" if archive_key is not None else "")
            indices = read_indices(indices)
        elif "indices" in item:
            if not isinstance(indices, list) or any(isinstance(index, bool) or not isinstance(index, int)
                                                    for index in indices):
                raise ValueError(f"Dataset {name} indices must be an integer list or index-file spec")
        sets.append(ImageSet(sources[key], indices, name))
    if overlap(*sets):
        raise ValueError("Dataset members and nonmembers overlap")
    return tuple(sets)


def half_sets(family, half, *, config=None, data_root=None, data_path=None):
    """Member and non-member sets for a model this repository trained on ``half``."""
    other = "first" if half == "second" else "second"
    sources = {}
    if family == "ddpm":
        config_path = Path(config).expanduser()
        if config_path.suffix != ".json":
            config_path = ROOT / "training/ddpm/configs" / f"{config}.json"
        flags = json.loads(config_path.read_text())["flags"]
        dataset = flags["dataset"].upper()
        folder = {"CIFAR10": "cifar10", "CIFAR100": "cifar100", "STL10_U": "stl10",
                  "TINY-IN": "tiny-imagenet-200/train"}[dataset]
        split = {"STL10_U": "STL10_Unlabeled"}.get(dataset, dataset) + "_train_ratio0.5.npz"
        source = Path(data_root or ROOT / "data/ddpm") / folder
        manifest = ROOT / "training/ddpm/splits" / split
        keys = {"first": "mia_train_idxs", "second": "mia_eval_idxs"}
        return (ImageSet.parse(f"{source}@{manifest}:{keys[half]}", sources),
                ImageSet.parse(f"{source}@{manifest}:{keys[other]}", sources))
    if family == "edm":
        source = ImageSource(data_path or ROOT / "data/cifar100-32x32.zip")
        if source.labels is None:
            raise ValueError("EDM halves need integer labels in dataset.json")
        halves = {"first": [], "second": []}
        # The first and last 250 images of each class, as in training.
        for label in np.unique(source.labels):
            indices = np.nonzero(source.labels == label)[0]
            halves["first"].append(indices[:len(indices) // 2])
            halves["second"].append(indices[len(indices) // 2:])
        return (ImageSet(source, np.sort(np.concatenate(halves[half])), f"{half} half"),
                ImageSet(source, np.sort(np.concatenate(halves[other])), f"{other} half"))
    if family == "sd14":
        recipe = json.loads((ROOT / "training/sd14/configs.json").read_text())["configs"][config]
        root = Path(data_root or ROOT / "data/sd14")
        return (ImageSet.parse(str(root / recipe["imagefolders"][half]), sources),
                ImageSet.parse(str(root / recipe["imagefolders"][other]), sources))
    raise ValueError(f"Unknown family {family}")
