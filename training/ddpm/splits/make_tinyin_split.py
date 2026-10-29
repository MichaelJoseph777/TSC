"""Generate the class-balanced Tiny ImageNet membership split.

torchvision's ImageFolder lists the 100,000 Tiny ImageNet training images class by
class, 500 per class, so index i belongs to class i // 500. Each half takes a random
250 images of every class, as in the CIFAR splits.
"""

from pathlib import Path

import numpy as np

CLASSES = 200
PER_CLASS = 500


def make_split(seed=0):
    # RandomState keeps a fixed stream across NumPy versions.
    rng = np.random.RandomState(seed)
    member, held_out = [], []
    for label in range(CLASSES):
        order = label * PER_CLASS + rng.permutation(PER_CLASS)
        member.append(np.sort(order[:PER_CLASS // 2]))
        held_out.append(np.sort(order[PER_CLASS // 2:]))
    return np.concatenate(member), np.concatenate(held_out)


if __name__ == "__main__":
    member, held_out = make_split()
    np.savez(Path(__file__).resolve().parent / "TINY-IN_train_ratio0.5.npz",
             mia_train_idxs=member, mia_eval_idxs=held_out, ratio=np.float64(0.5))
