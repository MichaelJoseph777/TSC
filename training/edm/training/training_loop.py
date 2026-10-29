# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

# mia-diffusion integration changes: explicit trusted checkpoint loading for
# PyTorch 2.8, and validation of resume counters and unfinished duration.

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc


#----------------------------------------------------------------------------

# Lightweight helpers for epoch-based checkpoints and W&B loss logging.

def _state_dict_to_cpu(module):
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}

def _load_module_state(dst_module, src, require_all=True):
    if isinstance(src, dict):
        dst_module.load_state_dict(src, strict=require_all)
    else:
        misc.copy_params_and_buffers(src_module=src, dst_module=dst_module, require_all=require_all)

def _load_training_state(path, total_kimg):
    # Legacy EDM states pickle Module objects. Only load trusted checkpoints.
    data = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    missing = {'cur_nimg', 'cur_tick', 'optimizer_state'} - data.keys()
    if missing:
        raise RuntimeError(f'Resume checkpoint lacks required progress/state keys: {sorted(missing)}')
    if int(data['cur_nimg']) < 0 or int(data['cur_tick']) < 0:
        raise RuntimeError('Resume progress counters must be nonnegative')
    if int(data['cur_nimg']) >= total_kimg * 1000:
        raise RuntimeError('Resume checkpoint already reached the requested duration, so increase the duration explicitly')
    return data

def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str))

def _save_weight_snapshot(run_dir, net, ema, epoch, cur_nimg, dataset_kwargs, network_kwargs, loss_kwargs):
    data = dict(
        net=_state_dict_to_cpu(net),
        ema=_state_dict_to_cpu(ema),
        epoch=int(epoch),
        cur_nimg=int(cur_nimg),
        dataset_kwargs=dict(dataset_kwargs),
        network_kwargs=dict(network_kwargs),
        loss_kwargs=dict(loss_kwargs),
    )
    torch.save(data, os.path.join(run_dir, f'weights-epoch-{int(epoch):06d}.pt'))

def _save_checkpoint_last(run_dir, net, ema, optimizer, epoch, cur_nimg, cur_tick, dataset_kwargs, network_kwargs, loss_kwargs):
    data = dict(
        net_state_dict=net.state_dict(),
        ema_state_dict=ema.state_dict(),
        optimizer_state=optimizer.state_dict(),
        epoch=int(epoch),
        cur_nimg=int(cur_nimg),
        cur_tick=int(cur_tick),
        dataset_kwargs=dict(dataset_kwargs),
        network_kwargs=dict(network_kwargs),
        loss_kwargs=dict(loss_kwargs),
    )
    last_path = os.path.join(run_dir, 'checkpoint-last.pt')
    prev_path = os.path.join(run_dir, 'checkpoint-last_prev.pt')
    tmp_path = os.path.join(run_dir, f'checkpoint-last.tmp.{os.getpid()}')
    torch.save(data, tmp_path)
    loaded = torch.load(tmp_path, map_location=torch.device('cpu'), weights_only=True)
    required = ['net_state_dict', 'ema_state_dict', 'optimizer_state', 'cur_nimg', 'cur_tick']
    if not all(key in loaded for key in required):
        raise RuntimeError(f'Checkpoint verification failed for {tmp_path}')
    del loaded
    if os.path.exists(last_path):
        os.replace(last_path, prev_path)
    os.replace(tmp_path, last_path)

def _init_wandb(wandb_kwargs, run_dir, config):
    if wandb_kwargs is None or not wandb_kwargs.get('enabled', False):
        return None, None
    try:
        import wandb
    except ImportError as err:
        raise RuntimeError('Weights & Biases logging was requested, but wandb is not installed') from err
    init_kwargs = dict(
        project=wandb_kwargs.get('project'),
        entity=wandb_kwargs.get('entity'),
        name=wandb_kwargs.get('name') or os.path.basename(run_dir),
        mode=wandb_kwargs.get('mode', 'online'),
        dir=run_dir,
        config=_jsonable(config),
    )
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    return wandb, wandb.init(**init_kwargs)

def _evaluate_validation_batch(loss_fn, net, batch, device, seed):
    images, labels = batch
    images = images.to(device).to(torch.float32) / 127.5 - 1
    labels = labels.to(device)
    was_training = net.training
    net.eval()
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        loss = loss_fn(net=net, images=images, labels=labels, augment_pipe=None)
        value = loss.mean().detach().cpu().item()
    net.train(was_training)
    return value

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    validation_kwargs   = None,     # Options for one-batch validation loss, None = disable.
    wandb_kwargs        = None,     # Options for Weights & Biases logging, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    weight_snapshot_epochs = 500,   # Save net+EMA weights every N epochs, None = disable.
    checkpoint_last_epochs = 10,    # Overwrite checkpoint-last.pt every N epochs, None = disable.
    wandb_train_interval = 100,     # Log mean train loss every N optimizer steps.
    validation_epochs   = 1,        # Log one validation batch every N epochs, None = disable.
    validation_batch_size = 128,    # Number of validation samples in the fixed validation batch.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    epoch_nimg = len(dataset_obj)
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Load a fixed validation batch on rank 0.
    validation_batch = None
    if validation_kwargs is not None and validation_epochs is not None and dist.get_rank() == 0:
        dist.print0('Loading validation batch...')
        validation_obj = dnnlib.util.construct_class_by_name(**validation_kwargs)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_obj, batch_size=validation_batch_size, shuffle=False, **data_loader_kwargs)
        validation_batch = next(iter(validation_loader))
        validation_obj.close()
        del validation_loader, validation_obj

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().to(device)
    net.requires_grad_(True)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM|FM)Loss
    optimizer_params = [param for param in net.parameters() if param.requires_grad]
    optimizer = dnnlib.util.construct_class_by_name(params=optimizer_params, **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Initialize W&B after run_dir exists and before training starts.
    wandb_module = None
    wandb_run = None
    if dist.get_rank() == 0:
        wandb_config = dict(
            dataset_kwargs=dict(dataset_kwargs),
            validation_kwargs=None if validation_kwargs is None else dict(validation_kwargs),
            network_kwargs=dict(network_kwargs),
            loss_kwargs=dict(loss_kwargs),
            optimizer_kwargs=dict(optimizer_kwargs),
            batch_size=batch_size,
            total_kimg=total_kimg,
            epoch_nimg=epoch_nimg,
            seed=seed,
        )
        wandb_module, wandb_run = _init_wandb(wandb_kwargs, run_dir, wandb_config)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    resume_state_nimg = None
    resume_state_tick = None
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = _load_training_state(resume_state_dump, total_kimg)
        if 'net_state_dict' in data:
            _load_module_state(net, data['net_state_dict'], require_all=True)
        else:
            _load_module_state(net, data['net'], require_all=True)
        if 'ema_state_dict' in data:
            _load_module_state(ema, data['ema_state_dict'], require_all=True)
        elif 'ema' in data:
            _load_module_state(ema, data['ema'], require_all=False)
        optimizer.load_state_dict(data['optimizer_state'])
        resume_state_nimg = data.get('cur_nimg', None)
        resume_state_tick = data.get('cur_tick', None)
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = int(resume_state_nimg) if resume_state_nimg is not None else resume_kimg * 1000
    cur_tick = int(resume_state_tick) if resume_state_tick is not None else 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    cur_step = cur_nimg // batch_size
    completed_epochs = cur_nimg // epoch_nimg
    next_weight_snapshot_epoch = None if weight_snapshot_epochs is None else ((completed_epochs // weight_snapshot_epochs) + 1) * weight_snapshot_epochs
    next_checkpoint_last_epoch = None if checkpoint_last_epochs is None else ((completed_epochs // checkpoint_last_epochs) + 1) * checkpoint_last_epochs
    next_validation_epoch = None if validation_epochs is None else completed_epochs + validation_epochs
    wandb_train_loss_sum = 0.0
    wandb_train_loss_count = 0
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        step_loss_sum = torch.zeros([], device=device)
        step_loss_count = torch.zeros([], device=device)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                step_loss_sum += loss.detach().sum()
                step_loss_count += loss.detach().numel()
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Per-step loss bookkeeping, W&B logging, and epoch-based checkpoints.
        loss_stats = torch.stack([step_loss_sum, step_loss_count])
        if dist.get_world_size() > 1:
            torch.distributed.all_reduce(loss_stats)
        step_loss = (loss_stats[0] / loss_stats[1].clamp_min(1)).item()
        cur_nimg += batch_size
        cur_step += 1
        completed_epochs = cur_nimg // epoch_nimg
        epoch_float = cur_nimg / epoch_nimg

        if wandb_run is not None:
            wandb_train_loss_sum += step_loss
            wandb_train_loss_count += 1
            if cur_step % wandb_train_interval == 0:
                wandb_run.log(dict(
                    **{'train/loss': wandb_train_loss_sum / max(wandb_train_loss_count, 1),
                       'train/step_loss': step_loss},
                    epoch=epoch_float,
                    kimg=cur_nimg / 1e3,
                ), step=cur_step)
                wandb_train_loss_sum = 0.0
                wandb_train_loss_count = 0

        if wandb_run is not None and validation_batch is not None and next_validation_epoch is not None and completed_epochs >= next_validation_epoch:
            while next_validation_epoch is not None and completed_epochs >= next_validation_epoch:
                val_loss = _evaluate_validation_batch(loss_fn, net, validation_batch, device, seed + int(next_validation_epoch))
                wandb_run.log(dict(**{'val/loss': val_loss}, epoch=float(next_validation_epoch), kimg=cur_nimg / 1e3), step=cur_step)
                next_validation_epoch += validation_epochs

        if dist.get_rank() == 0 and next_weight_snapshot_epoch is not None and completed_epochs >= next_weight_snapshot_epoch:
            while next_weight_snapshot_epoch is not None and completed_epochs >= next_weight_snapshot_epoch:
                _save_weight_snapshot(run_dir, net, ema, next_weight_snapshot_epoch, cur_nimg, dataset_kwargs, network_kwargs, loss_kwargs)
                next_weight_snapshot_epoch += weight_snapshot_epochs

        if dist.get_rank() == 0 and next_checkpoint_last_epoch is not None and completed_epochs >= next_checkpoint_last_epoch:
            _save_checkpoint_last(run_dir, net, ema, optimizer, completed_epochs, cur_nimg, cur_tick, dataset_kwargs, network_kwargs, loss_kwargs)
            while next_checkpoint_last_epoch is not None and completed_epochs >= next_checkpoint_last_epoch:
                next_checkpoint_last_epoch += checkpoint_last_epochs

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, ema=ema, optimizer_state=optimizer.state_dict(), cur_nimg=int(cur_nimg), cur_tick=int(cur_tick)), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if wandb_run is not None:
        wandb_run.finish()

    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
