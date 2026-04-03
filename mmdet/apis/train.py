# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_runner,
                         get_dist_info)

from mmdet.core import DistEvalHook, EvalHook, build_optimizer
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                         find_latest_checkpoint, get_root_logger)


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.train_dataloader.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s) with {samples_per_gpu} '
                f'samples per GPU. The total batch size is {batch_size}.')

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]
    # data_loaders[batch_idx]: dict(['img_metas', 'img', 'gt_bboxes', 'gt_labels'])
    #   data_loaders[batch_idx]['img_metas'].data: list(4 x dict{
    #       'filename',
    #       'ori_filename',
    #       'ori_shape': (256, 256, 172),
    #       'img_shape': (1200, 1200, 172),
    #       'pad_shape': (1216, 1216, 172),
    #       'scale_factor': array([4.6875, 4.6875, 4.6875, 4.6875], dtype=float32),
    #       'flip': True,
    #       'flip_direction': 'horizontal',
    #       'img_norm_cfg': {'mean','std'}
    #       'to_rgb': False}
    #   }]
    #   data_loaders[batch_idx]['img'].data: torch.Size([4, 172, 1216, 1216])
    #   data_loaders[batch_idx]['gt_bboxes'].data: list of [4 x tensor(1, 4)]
    #   data_loaders[batch_idx]['gt_labels'].data: list of [4 x tensor(1, 4)]
    #check(data_loaders)
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids) # cfg.gpu_ids = [--gpu_id]

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)

import json
from PIL import Image
import cv2
def find_bbox_for_file(filename: str) -> tuple:
    with open("/ssd2/TzuYu/metadata.jsonl", "r", encoding="utf-8") as f:
        meta_data = [json.loads(line) for line in f]
    for entry in meta_data:
    # If your metadata uses the base name without extension:
        if entry.get("file_prefix") == os.path.splitext(filename)[0]:
            bbox = entry.get("bbox")
            bbox[1] = 255 if bbox[1] < bbox[0] else bbox[1]
            bbox[3] = 255 if bbox[3] < bbox[2] else bbox[3]
            return tuple(bbox)

def to_uint8(channel):
    c_min, c_max = channel.min(), channel.max()
    scaled = (channel - c_min) / (c_max - c_min)  # [0,1]
    return (255 * scaled).astype(np.uint8)
def false_color_image(hsi: np.ndarray, nir_band=5, red_band=15, green_band=25) -> Image:
    blue  = hsi[:, :, nir_band]
    green = hsi[:, :, red_band]
    red   = hsi[:, :, green_band]
    rgb8 = np.dstack([to_uint8(red), to_uint8(green), to_uint8(blue)])

    return rgb8
def draw_bounding_box(image: np.ndarray, bbox: tuple, color: tuple=tuple([255, 255, 0])) -> np.ndarray:
    # Draw a rectangle on the image using the bounding box coordinates
    if bbox is not None:
        #print(bbox)
        x1, y1, x2, y2 = bbox[0:4]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    return image
from mmcv import imshow_bboxes, tensor2imgs
def check(data_loaders):
    with open('out.log', 'w') as f:
        print(f"Prinitng data_loaders {len(data_loaders[0])}")
        for batch_inputs in data_loaders[0]:
            img_metas = batch_inputs['img_metas'].data[0]
            gt_bboxes = batch_inputs['gt_bboxes'].data[0]
            img = batch_inputs['img'].data[0]
            #print(f"Batch Inputs['img']: {batch_inputs['img'].data[0].shape}")
            # print(f"Batch Inputs['gt_bboxes']: {gt_bboxes}\n")
            for idx in [0, 1, 2, 3]:
                meta_bbox = find_bbox_for_file(img_metas[idx]['ori_filename'].replace('_inpaint_result(0)', ''))[0:4]
                meta_bbox = [meta_bbox[2], meta_bbox[0], meta_bbox[3], meta_bbox[1]]
                if img_metas[idx]['flip']:
                    meta_bbox = [255 - meta_bbox[2], meta_bbox[1], 255 - meta_bbox[0], meta_bbox[3]]
                meta_bbox = [x * 4.6875 for x in meta_bbox ]

                f.write(
                    f"{img_metas[idx]['ori_filename']}:\n"
                    f"\timg_max[{img[idx].max()}], img_min[{img[idx].min()}], img_mean[{img[idx].mean()}]\n"
                    f"\tflip: {img_metas[idx]['flip']}, flip_direction: {img_metas[idx]['flip_direction']}\n"
                    f"\tgt_bbox({gt_bboxes[idx]}),\n"
                    f"\tmeta_bbox      ({meta_bbox})\n"
                )

                img_input = np.array(img[idx]).transpose(1, 2, 0)
                false_img = false_color_image(np.array(img_input))
                false_img = imshow_bboxes(false_img, np.array(gt_bboxes[idx]), show=False, colors='red')
                imshow_bboxes(false_img, np.array([meta_bbox]), colors='blue', show=False, 
                             out_file=f'./img_log/{img_metas[idx]["ori_filename"]}.png')
        print("end printing dataloaders")
    exit(0)