"""TSC and RMC membership scores and their ROC metrics.

Losses are averaged over noise draws before the logarithm. Lower scores mean
"more likely a member" for both attacks, and no labels are used to choose the sign.
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-12


def log_loss(losses):
    """[images, levels, draws] losses -> [images, levels] stabilized log-losses."""
    return np.log(np.asarray(losses, dtype=np.float64).mean(axis=-1) + EPSILON)


def tsc(log_losses, left, center, right):
    """Temporal self-calibration: center log-loss minus the mean of its two context points."""
    return log_losses[:, center] - 0.5 * (log_losses[:, left] + log_losses[:, right])


def rmc(target_log_losses, reference_log_losses, level):
    """Reference-model calibration: target minus reference log-loss at one level."""
    return target_log_losses[:, level] - reference_log_losses[:, level]


def metrics(member_scores, nonmember_scores, fpr_limit=0.01):
    """AUC, ASR (best balanced accuracy over thresholds), and TPR at FPR <= 1%.

    Members are predicted for score <= threshold. Tied scores share one
    threshold, so ties never separate the classes.
    """
    member = np.asarray(member_scores, dtype=np.float64)
    nonmember = np.asarray(nonmember_scores, dtype=np.float64)
    if not (np.isfinite(member).all() and np.isfinite(nonmember).all()):
        raise ValueError("Scores must be finite")
    scores = np.concatenate([member, nonmember])
    labels = np.concatenate([np.ones(len(member), bool), np.zeros(len(nonmember), bool)])
    order = np.argsort(scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    last_of_tie = np.r_[scores[1:] != scores[:-1], True]
    tpr = np.r_[0.0, np.cumsum(labels)[last_of_tie] / len(member)]
    fpr = np.r_[0.0, np.cumsum(~labels)[last_of_tie] / len(nonmember)]
    area = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return {"auc": float(area(tpr, fpr)),
            "asr": float((1 + (tpr - fpr).max()) / 2),
            "tpr_at_1pct_fpr": float(tpr[fpr <= fpr_limit][-1])}
