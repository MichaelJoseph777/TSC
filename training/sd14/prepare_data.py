"""Prepare single-caption COCO, Pokemon, and Flickr training imagefolders."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
MARKER = ".mia_diffusion_sd14"
# Hugging Face revisions used for the paper's models.
POKEMON_REPO, POKEMON_REVISION = "reach-vb/pokemon-blip-captions", "3790f29981f74ca26bf11662048342e19c5b615e"
FLICKR_REPO, FLICKR_REVISION = "nlphuji/flickr30k", "2b239befc81b6e3f035ce6bd52f5f4d60f5625f7"
CLID_COCO_REPO, CLID_COCO_REVISION = "zsf/COCO_MIA_ori_split1", "4af0207c955c893a49b7c2970db5ada414b37ed2"
CLID_COCO_ARCHIVE = "mia_COCO.zip"
CLID_COCO_SHA256 = "7b78861d38b07a9593f2615418ccde8b8d06cd9d8a6990496169a7d7a89ee587"
# CLiD's two disjoint 2,500-image halves of MS-COCO and their imagefolders.
CLID_COCO_HALVES = {"first": ("ori/coco_ori_train/", "coco_2500"),
                    "second": ("ori/coco_ori_test/", "coco_holdout_2500")}
POKEMON_SEED = 0


def create_folder(path):
    if path.exists():
        if not (path / MARKER).is_file():
            raise RuntimeError(f"Refusing to replace an unmarked directory: {path}")
        shutil.rmtree(path)
    (path / "images").mkdir(parents=True)
    (path / MARKER).write_text("Generated SD1.4 training imagefolder.\n")


def write_metadata(path, rows):
    with (path / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def pokemon_rows():
    """A fixed random split of the 833 rows: 417 for the first half and the other 416 for the second."""
    import numpy as np

    # RandomState keeps a fixed stream across NumPy versions.
    order = np.random.RandomState(POKEMON_SEED).permutation(833)
    return {"first": sorted(order[416:].tolist()), "second": sorted(order[:416].tolist())}


def prepare_pokemon(output, cache, halves):
    from datasets import load_dataset

    dataset = load_dataset(POKEMON_REPO, split="train", revision=POKEMON_REVISION, cache_dir=str(cache))
    if len(dataset) != 833:
        raise RuntimeError(f"Expected 833 Pokemon images, found {len(dataset)}")
    folders = {"first": "pokemon_417", "second": "pokemon_holdout_416"}
    for half in halves:
        target = output / folders[half]
        create_folder(target)
        rows = []
        for row in pokemon_rows()[half]:
            example = dataset[row]
            name = f"pokemon_{row:04d}.jpg"
            example["image"].convert("RGB").save(target / "images" / name, format="JPEG", quality=95)
            rows.append({"file_name": f"images/{name}", "text": example["text"]})
        write_metadata(target, rows)


def prepare_coco(output, cache, archive, halves):
    """Stage CLiD's released MS-COCO split: its train half first and its test half second."""
    if archive is None:
        from huggingface_hub import hf_hub_download

        archive = hf_hub_download(CLID_COCO_REPO, CLID_COCO_ARCHIVE, repo_type="dataset",
                                  revision=CLID_COCO_REVISION, cache_dir=str(cache))
    digest = hashlib.sha256()
    with open(archive, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    if digest.hexdigest() != CLID_COCO_SHA256:
        raise RuntimeError(f"{archive} is not the expected {CLID_COCO_ARCHIVE}")
    with zipfile.ZipFile(archive) as source:
        for half in halves:
            prefix, folder = CLID_COCO_HALVES[half]
            target = output / folder
            create_folder(target)
            rows = []
            for line in source.read(prefix + "metadata.jsonl").decode("utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                # Use only the base name so archive paths cannot escape the target folder.
                name = Path(row["file_name"]).name
                (target / "images" / name).write_bytes(source.read(prefix + name))
                rows.append({"file_name": f"images/{name}", "text": row["text"]})
            if len(rows) != 2500:
                raise RuntimeError(f"Expected 2500 COCO images in {prefix}, found {len(rows)}")
            write_metadata(target, rows)


def prepare_flickr(output, cache, halves):
    from datasets import load_dataset

    dataset = load_dataset(FLICKR_REPO, split="test", streaming=True, revision=FLICKR_REVISION, cache_dir=str(cache))
    # Streamed rows 0-9,999 form the first half and rows 10,000-19,999 the second.
    # The overfit folders use the first 2,500 rows of each half.
    layouts = {"first": ("flickr_10000", "flickr_2500", "flickr_{:05d}.jpg", 0),
               "second": ("flickr_holdout_10000", "flickr_holdout_2500", "flickr_holdout_{:05d}.jpg", 10000)}
    rows = {}
    for half in halves:
        full, small, _, _ = layouts[half]
        create_folder(output / full)
        create_folder(output / small)
        rows[half] = []
    stop = max(layouts[half][3] for half in halves) + 10000
    for index, example in enumerate(dataset):
        if index >= stop:
            break
        half = "first" if index < 10000 else "second"
        if half not in rows:
            continue
        full, small, pattern, start = layouts[half]
        position = index - start
        name = pattern.format(position)
        buffer = io.BytesIO()
        example["image"].convert("RGB").save(buffer, format="JPEG", quality=95)
        (output / full / "images" / name).write_bytes(buffer.getvalue())
        if position < 2500:
            (output / small / "images" / name).write_bytes(buffer.getvalue())
        caption = example["caption"][0] if isinstance(example["caption"], list) else example["caption"]
        rows[half].append({"file_name": f"images/{name}", "text": caption})
    for half, half_rows in rows.items():
        if len(half_rows) != 10000:
            raise RuntimeError(f"Expected 10000 Flickr images for the {half} half, found {len(half_rows)}")
        full, small, _, _ = layouts[half]
        write_metadata(output / full, half_rows)
        write_metadata(output / small, half_rows[:2500])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "coco", "pokemon", "flickr"], default="all")
    parser.add_argument("--half", choices=["both", "first", "second"], default="both")
    parser.add_argument("--output-dir", type=Path, default=HERE.parents[1] / "data/sd14")
    parser.add_argument("--coco-clid-archive", type=Path,
                        help=f"Local copy of {CLID_COCO_ARCHIVE} from {CLID_COCO_REPO}, downloaded if omitted.")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    cache = (args.cache_dir or output / "cache").expanduser().resolve()
    halves = ("first", "second") if args.half == "both" else (args.half,)
    if args.dry_run:
        print(f"Prepare {args.dataset} training imagefolders ({', '.join(halves)} half) under {output}")
        return
    if args.dataset in ("all", "pokemon"):
        prepare_pokemon(output, cache, halves)
    if args.dataset in ("all", "coco"):
        archive = args.coco_clid_archive.expanduser().resolve() if args.coco_clid_archive else None
        prepare_coco(output, cache, archive, halves)
    if args.dataset in ("all", "flickr"):
        prepare_flickr(output, cache, halves)


if __name__ == "__main__":
    main()
