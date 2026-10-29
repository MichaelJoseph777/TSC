import copy
import glob
import json
import os
import re
import random
from pathlib import Path
import sys

import numpy as np
import warnings
from absl import app, flags
import torch
from tensorboardX import SummaryWriter
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, CelebA, SVHN
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
from mia_evals.dataset_utils import MIACelebA, MIACIFAR10, MIACIFAR100, MIASVHN, MIASTL10, MIAImageFolder
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet

# from opacus import PrivacyEngine


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_enum('beta_schedule', 'linear', ['linear', 'cosine'], help='noise beta schedule')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800001, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', True, help='multi gpu training')
flags.DEFINE_bool('conditional', False, help='class-conditional training using dataset labels')
flags.DEFINE_integer('num_classes', 0, help='number of class labels for conditional training (0 infers it for supported datasets)')

flags.DEFINE_string('dataset', 'CIFAR10', help='data set')
flags.DEFINE_string('dataset_root', './datasets', help='data set')
flags.DEFINE_string('split_root', str(Path(__file__).resolve().parent / 'splits'),
                    help='directory containing exact saved member/eval index manifests')
flags.DEFINE_integer('seed', None, help='optional new-run seed (historical training did not record one)')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 100000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 100000, help='frequency of rolling checkpoints, 0 to disable during training')
flags.DEFINE_integer('archive_step', 0, help='frequency of permanent milestone checkpoints, 0 to disable')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 128, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_string('defense', None, help='Defense tricks, e.g., DP-SGD or L2 Regularizationgg')
flags.DEFINE_bool('only_member', True, help='Training only on member split')
flags.DEFINE_enum('train_split', 'auto', ['auto', 'member', 'eval', 'all'],
                  help='which MIA split to train/evaluate on (auto preserves only_member semantics)')
flags.DEFINE_enum('hflip', 'auto', ['auto', 'true', 'false'],
                  help='override RandomHorizontalFlip usage in training transforms')
flags.DEFINE_bool('resume', True, help='resume from the newest checkpoint in logdir')

device = torch.device('cuda:0')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader, return_labels=False):
    while True:
        for x, y in iter(dataloader):
            if return_labels:
                yield x, y
            else:
                yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def resolve_train_split(only_member):
    if FLAGS.train_split != 'auto':
        return FLAGS.train_split
    return 'member' if only_member else 'all'


def split_file_name(dataset):
    return {
        'STL10_U': 'STL10_Unlabeled_train_ratio0.5.npz',
    }.get(dataset.upper(), f'{dataset.upper()}_train_ratio0.5.npz')


def selected_split_indices(dataset, only_member):
    split = resolve_train_split(only_member)
    if dataset.upper() == 'CIFAR10-SYNTHETIC':
        if split != 'all':
            raise ValueError('CIFAR10-SYNTHETIC only supports train_split=all')
        return None, split
    if split == 'all':
        return None, split

    splits = np.load(os.path.join(FLAGS.split_root, split_file_name(dataset)), allow_pickle=False)
    key = {
        'member': 'mia_train_idxs',
        'eval': 'mia_eval_idxs',
    }[split]
    return splits[key], split


def use_random_hflip(default):
    if FLAGS.hflip == 'auto':
        return default
    return FLAGS.hflip == 'true'


def hflip_transform(default):
    if use_random_hflip(default):
        return [torchvision.transforms.RandomHorizontalFlip()]
    return []


def infer_num_classes(dataset):
    if FLAGS.num_classes > 0:
        return FLAGS.num_classes
    dataset = dataset.upper()
    if dataset == 'CIFAR10':
        return 10
    if dataset == 'CIFAR100':
        return 100
    if dataset == 'TINY-IN':
        return 200
    raise ValueError(f'Cannot infer num_classes for conditional dataset={FLAGS.dataset}, so pass --num_classes')


def checkpoint_candidates(logdir):
    paths = glob.glob(os.path.join(logdir, 'ckpt-step*.pt'))
    paths.extend([
        os.path.join(logdir, 'ckpt-last.pt'),
        os.path.join(logdir, 'ckpt-prev-last.pt'),
    ])
    paths = [path for path in paths if os.path.exists(path)]
    return sorted(paths, key=os.path.getmtime, reverse=True)


def save_checkpoint(path, net_model, ema_model, sched, optim, step, x_T):
    ckpt = {
        'net_model': net_model.state_dict(),
        'ema_model': ema_model.state_dict(),
        'sched': sched.state_dict(),
        'optim': optim.state_dict(),
        'step': step,
        'x_T': x_T,
    }
    tmp_path = path + '.tmp'
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def save_rolling_checkpoint(logdir, net_model, ema_model, sched, optim, step, x_T):
    last_path = os.path.join(logdir, 'ckpt-last.pt')
    prev_path = os.path.join(logdir, 'ckpt-prev-last.pt')
    next_path = os.path.join(logdir, 'ckpt-next.pt')
    save_checkpoint(next_path, net_model, ema_model, sched, optim, step, x_T)
    if os.path.exists(last_path):
        os.replace(last_path, prev_path)
    os.replace(next_path, last_path)


def evaluate(sampler, model, save_path, ckpt_name):
    # model.eval()
    # with torch.no_grad():
    #     images = []
    #     desc = "generating images"
    #     for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
    #         batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
    #         x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
    #         batch_images = sampler(x_T.to(device)).cpu()
    #         images.append((batch_images + 1) / 2)
    #     images = torch.cat(images, dim=0).numpy()
    # # save sampled images
    # torch.save({'samples': images}, os.path.join(save_path, f'{ckpt_name}_samples.pt'))
    # model.train()
    # (IS, IS_std), FID = get_inception_and_fid_score(
    #     images, FLAGS.fid_cache, num_images=FLAGS.num_images,
    #     use_torch=FLAGS.fid_use_torch, verbose=True)
    # return (IS, IS_std), FID, images


    dataloader = get_dataset(FLAGS, only_member=FLAGS.only_member)
    datalooper = infiniteloop(dataloader, return_labels=FLAGS.conditional)
    x_0 = next(datalooper)

    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))

            alpha = 0.05
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
            sqrt_alpha = torch.sqrt(alpha_tensor)
            sqrt_1_alpha = torch.sqrt(torch.tensor(1 - alpha, dtype=torch.float32))
            x_T = sqrt_alpha * x_0[0:batch_size] + sqrt_1_alpha * x_T

            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    # save sampled images
    torch.save({'samples': images}, os.path.join(save_path, f'{ckpt_name}_samples.pt'))

    mean = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
    std = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
    x_0 = x_0 * std + mean
    x_0 = torch.clamp(x_0, 0, 1)
    torch.save({'samples': x_0}, os.path.join(save_path, f'{ckpt_name}_origins.pt'))
    return x_0, images


def get_dataset(FLAGS, only_member=False):
    dataset_root = FLAGS.dataset_root
    selected_idxs, selected_split = selected_split_indices(FLAGS.dataset, only_member)

    if FLAGS.dataset.upper() == 'CIFAR10':
        transforms = torchvision.transforms.Compose(hflip_transform(True) + [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))])
        if selected_idxs is not None:
            dataset = MIACIFAR10(selected_idxs, root=os.path.join(dataset_root, 'cifar10'), train=True,
                                 transform=transforms, download=True)
        else:
            dataset = CIFAR10(root=os.path.join(dataset_root, 'cifar10'), train=True,
                              transform=transforms)
    elif FLAGS.dataset.upper() == 'CIFAR100':
        transforms = torchvision.transforms.Compose(hflip_transform(True) + [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))])
        if selected_idxs is not None:
            dataset = MIACIFAR100(selected_idxs, root=os.path.join(dataset_root, 'cifar100'), train=True,
                                  transform=transforms)
        else:
            dataset = CIFAR100(root=os.path.join(dataset_root, 'cifar100'), train=True,
                               transform=transforms)
    elif FLAGS.dataset.upper() == 'CELEBA':
        # for CelebA, first center crop 140 and then resize to 32 (by default)
        transforms = torchvision.transforms.Compose([
            # torchvision.transforms.CenterCrop(140),
            torchvision.transforms.Resize(FLAGS.img_size),
            *hflip_transform(True),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])
        if selected_idxs is not None:
            dataset = MIACelebA(selected_idxs, root=os.path.join(dataset_root, 'celeba'), split='train',
                                transform=transforms, download=True)
        else:
            dataset = CelebA(root=os.path.join(dataset_root, 'celeba'), split='train',
                             transform=transforms, download=True)
    elif FLAGS.dataset.upper() == 'SVHN':
        transforms = torchvision.transforms.Compose(hflip_transform(True) + [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))])
        if selected_idxs is not None:
            dataset = MIASVHN(selected_idxs, root=os.path.join(dataset_root, 'svhn'), split='train',
                              transform=transforms, download=True)
        else:
            dataset = SVHN(root=os.path.join(dataset_root, 'svhn'), split='train',
                           transform=transforms, download=True)
    elif FLAGS.dataset.upper() == 'STL10_U':
        transforms = torchvision.transforms.Compose(hflip_transform(True) + [
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])
        if selected_idxs is not None:
            dataset = MIASTL10(selected_idxs, root=os.path.join(dataset_root, 'stl10'), split='unlabeled',
                               download=True, transform=transforms)
        else:
            dataset = torchvision.datasets.STL10(root=os.path.join(dataset_root, 'stl10'), split='unlabeled',
                                                 download=True, transform=transforms)
    elif FLAGS.dataset.upper() == 'TINY-IN':
        transforms = torchvision.transforms.Compose(hflip_transform(True) + [
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])
        if selected_idxs is not None:
            dataset = MIAImageFolder(selected_idxs, root=os.path.join(dataset_root, 'tiny-imagenet-200/train'),
                                     transform=transforms)
        else:
            dataset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_root, 'tiny-imagenet-200/train'),
                                                       transform=transforms)
    else:
        raise NotImplemented

    print(f'Using train_split={selected_split}, hflip={FLAGS.hflip}, dataset_size={len(dataset)}')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True,
                                              num_workers=FLAGS.num_workers)

    return data_loader


def train():
    dataloader = get_dataset(FLAGS, only_member=FLAGS.only_member)
    num_classes = infer_num_classes(FLAGS.dataset) if FLAGS.conditional else None
    print(f'conditional={FLAGS.conditional}, num_classes={num_classes}')
    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        num_classes=num_classes)
    if FLAGS.defense == 'L2':
        print('Applying L2 Regularization')
        optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr, weight_decay=5e-4)
    else:
        optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)

    ema_model = copy.deepcopy(net_model)

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T,
        beta_schedule=FLAGS.beta_schedule).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type, num_classes=num_classes,
        beta_schedule=FLAGS.beta_schedule).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type, num_classes=num_classes,
        beta_schedule=FLAGS.beta_schedule).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    datalooper = infiniteloop(dataloader, return_labels=FLAGS.conditional)

    # log setup
    if not os.path.exists(os.path.join(FLAGS.logdir, 'sample')):
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    start_step = 0
    if FLAGS.resume:
        for resume_path in checkpoint_candidates(FLAGS.logdir):
            try:
                ckpt = torch.load(resume_path, map_location=device)
                net_model.load_state_dict(ckpt['net_model'])
                ema_model.load_state_dict(ckpt['ema_model'])
                optim.load_state_dict(ckpt['optim'])
                sched.load_state_dict(ckpt['sched'])
                x_T = ckpt['x_T'].to(device)
                start_step = int(ckpt['step']) + 1
                print(f'Resuming from {resume_path} at step {start_step}')
                break
            except Exception as error:
                print(f'Could not load checkpoint {resume_path}: {error}')

    writer = SummaryWriter(FLAGS.logdir, purge_step=start_step)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(start_step, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            batch = next(datalooper)
            if FLAGS.conditional:
                x_0, labels = batch
                x_0 = x_0.to(device)
                labels = labels.to(device).long()
                loss = trainer(x_0, labels).mean()
            else:
                x_0 = batch.to(device)
                loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            loss_value = loss.item()
            writer.add_scalar('loss', loss_value, step)
            pbar.set_postfix(loss='%.3f' % loss_value)

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            completed_steps = step + 1
            if FLAGS.save_step > 0 and completed_steps % FLAGS.save_step == 0:
                save_rolling_checkpoint(
                    FLAGS.logdir, net_model, ema_model, sched, optim, step, x_T)

            if FLAGS.archive_step > 0 and completed_steps % FLAGS.archive_step == 0:
                save_checkpoint(
                    os.path.join(FLAGS.logdir, f'ckpt-step{completed_steps}.pt'),
                    net_model, ema_model, sched, optim, step, x_T)

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    if FLAGS.total_steps > 0 and start_step < FLAGS.total_steps:
        final_step = FLAGS.total_steps - 1
        save_checkpoint(
            os.path.join(FLAGS.logdir, f'ckpt-step{FLAGS.total_steps}.pt'),
            net_model, ema_model, sched, optim, final_step, x_T)
    writer.close()


def eval():
    num_classes = infer_num_classes(FLAGS.dataset) if FLAGS.conditional else None
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        num_classes=num_classes)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type,
        num_classes=num_classes, beta_schedule=FLAGS.beta_schedule).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)


    # load model and evaluate
    ckpt_name = 'checkpoint.pt'
    ckpt = torch.load(os.path.join(FLAGS.logdir, ckpt_name))
    model.load_state_dict(ckpt['net_model'])
    origins, samples = evaluate(sampler, model, FLAGS.logdir, ckpt_name.split('.')[0])
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples.png'),
        nrow=16)

    save_image(
        torch.tensor(origins[:256]),
        os.path.join(FLAGS.logdir, 'origins.png'),
        nrow=16)

    # # load model and evaluate
    # ckpt_name = 'checkpoint.pt'
    # ckpt = torch.load(os.path.join(FLAGS.logdir, ckpt_name))
    # model.load_state_dict(ckpt['net_model'])
    # (IS, IS_std), FID, samples = evaluate(sampler, model, FLAGS.logdir, ckpt_name.split('.')[0])
    # print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    # save_image(
    #     torch.tensor(samples[:256]),
    #     os.path.join(FLAGS.logdir, 'samples.png'),
    #     nrow=16)
    #
    # model.load_state_dict(ckpt['ema_model'])
    # (IS, IS_std), FID, samples = evaluate(sampler, model, FLAGS.logdir, ckpt_name.split('.')[0])
    # print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    # save_image(
    #     torch.tensor(samples[:256]),
    #     os.path.join(FLAGS.logdir, f'{ckpt_name}samples_ema.png'),
    #     nrow=16)


def main(argv):
    global device
    if FLAGS.eval or FLAGS.eval_step > 0:
        raise ValueError('This import supports training only. Use a separate evaluation entry point.')
    if FLAGS.mean_type != 'epsilon':
        raise ValueError('The imported training loss supports epsilon prediction only.')
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
