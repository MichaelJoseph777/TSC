# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Helpers for flow-matching prediction/loss spaces.

def _normalize_fm_space(space):
    if space == 'eps':
        space = 'e'
    if space not in ['x', 'e', 'v']:
        raise ValueError(f'Invalid flow-matching space "{space}"')
    return space

def _safe_t(t, t_eps):
    return t.clamp_min(t_eps)

def _safe_one_minus_t(t, t_eps):
    return (1 - t).clamp_min(t_eps)

def _fm_target(space, x, e):
    space = _normalize_fm_space(space)
    if space == 'x':
        return x
    if space == 'e':
        return e
    assert space == 'v'
    return x - e

def _fm_convert_prediction(pred, from_space, to_space, z_t, t, t_eps):
    from_space = _normalize_fm_space(from_space)
    to_space = _normalize_fm_space(to_space)
    if from_space == to_space:
        return pred

    safe_t = _safe_t(t, t_eps)
    safe_one_minus_t = _safe_one_minus_t(t, t_eps)
    one_minus_t = 1 - t

    if to_space == 'x':
        if from_space == 'e':
            return (z_t - one_minus_t * pred) / safe_t
        if from_space == 'v':
            return one_minus_t * pred + z_t
    if to_space == 'e':
        if from_space == 'x':
            return (z_t - t * pred) / safe_one_minus_t
        if from_space == 'v':
            return z_t - t * pred
    if to_space == 'v':
        if from_space == 'x':
            return (pred - z_t) / safe_one_minus_t
        if from_space == 'e':
            return (z_t - pred) / safe_t

    raise ValueError(f'Cannot convert flow-matching prediction from "{from_space}" to "{to_space}"')

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Unweighted linear flow-matching loss.

@persistence.persistent_class
class FMLoss:
    def __init__(self, prediction='v', loss_target='v', t_eps=5e-2):
        self.prediction = _normalize_fm_space(prediction)
        self.loss_target = _normalize_fm_space(loss_target)
        self.t_eps = t_eps

    def __call__(self, net, images, labels=None, augment_pipe=None):
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        t = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        e = torch.randn_like(y)
        z_t = t * y + (1 - t) * e

        pred = net(z_t, t, labels, augment_labels=augment_labels)
        pred_for_loss = _fm_convert_prediction(pred, self.prediction, self.loss_target, z_t, t, self.t_eps)
        target = _fm_target(self.loss_target, y, e)
        loss = (pred_for_loss - target) ** 2
        return loss

#----------------------------------------------------------------------------
