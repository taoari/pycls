#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

import os
import time
import json

import numpy as np
import torch.nn as nn
import pycls.core.benchmark as benchmark
import pycls.core.builders as builders
import pycls.core.checkpoint as checkpoint
import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.meters as meters
import pycls.core.net as net
import pycls.core.optimizer as optim
import pycls.datasets.loader as loader
import torch
from pycls.core.config import cfg

from torch.utils.tensorboard import SummaryWriter
from taowei.torch2.utils import _unwrap_model

# # NOTE: it is not necessary to import print, as initialize_logger redirect both stdout and stderr
# #     but it can be faster for direct logging without reformatting
# from taowei.torch2.utils.logging import print
# from taowei.torch2.utils import viz
# from taowei.torch2.utils import classif
# viz.print = print
# classif.print = print

logger = logging.get_logger(__name__)
writer = None


def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure that the output dir exists
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    global writer
    if dist.is_master_proc():
        writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, 'runs'))
    from taowei.torch2.utils.classif import print_torch_info
    print_torch_info()


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.build_model()
    # Print summary and plot network
    if cfg.PRINT_SUMMARY:
        try:
            from taowei.torch2.utils.viz import print_summary, plot_network
            model.eval() # NOTE: avoid batch_norm buffer being changed
            data_shape = (1, 3, cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE)
            print_summary(model, data_shape=data_shape)
            if cfg.NUM_GPUS == 1: # not args.distributed: # TODO: support for distributed
                plot_network(model, data_shape=data_shape).save(os.path.join(cfg.OUT_DIR, cfg.MODEL.ARCH if cfg.MODEL.ARCH else 'network') + '.gv')
        except Exception as e:
            print(e)
    # Log Model Info
    model_strs = str(model).split('\n')
    model_strs = model_strs[:25] + ['... ...'] + model_strs[-25:] if len(model_strs) > 50 else model_strs
    logger.info("Model:\n{}".format('\n'.join(model_strs)))
    if hasattr(_unwrap_model(model), 'genotype'):
        print('Genotype:\n{}'.format(_unwrap_model(model).genotype))
    # Log model complexity
    logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            # NOTE: find_unused_parameters=True for DARTS models with auxiliary branch, otherwise will raise RuntimeError: your module has parameters that were not used in producing loss
            module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
        )
        # Set complexity function to be module's complexity function
        if hasattr(model.module, 'complexity'):
            model.complexity = model.module.complexity
    return model


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
    """Performs one epoch of training."""
    from taowei.torch2.utils.classif import ProgressMeter
    progress = ProgressMeter(iters_per_epoch=len(train_loader),
        epoch=cur_epoch, epochs=cfg.OPTIM.MAX_EPOCH, split='train', writer=writer, batch_size=cfg.TRAIN.BATCH_SIZE)
    # Shuffle the data
    loader.shuffle(train_loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    # train_meter.iter_tic()
    # end = time.time()
    from taowei.timer import Timer
    timer = Timer()
    timer.tic()
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # measure data loading time
        # progress.update('data_time', time.time() - end)
        progress.update('data_time', timer.toc(from_last_toc=True))

        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        if cfg.MODEL.AUXILIARY_WEIGHT > 0.0:
            # Perform the forward pass
            preds, preds_aux = model(inputs)
            # Compute the loss
            loss = loss_fun(preds, labels)
            loss += cfg.MODEL.AUXILIARY_WEIGHT * loss_fun(preds_aux, labels)
        else:
            # Perform the forward pass
            preds = model(inputs)
            # Compute the loss
            loss = loss_fun(preds, labels)
        progress.update('forward_time', timer.toc(from_last_toc=True))
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        if cfg.OPTIM.GRAD_CLIP > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIM.GRAD_CLIP)
        progress.update('backward_time', timer.toc(from_last_toc=True))
        # Update the parameters
        optimizer.step()
        progress.update('update_time', timer.toc(from_last_toc=True))
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        # train_meter.iter_toc()
        # Update and log stats
        mb_size = inputs.size(0) * cfg.NUM_GPUS

        progress.update('loss', loss, mb_size)
        progress.update('top1_err', top1_err, mb_size)
        progress.update('top5_err', top5_err, mb_size)

        # measure elapsed time
        # progress.update('batch_time', time.time() - end)
        # end = time.time()
        progress.update('batch_time', timer.toctic())

        if cur_iter % cfg.LOG_PERIOD == 0:
            progress.log_iter_stats(iter=cur_iter, batch_size=mb_size,
                lr=optimizer.param_groups[0]['lr'])

    progress.log_epoch_stats(lr=optimizer.param_groups[0]['lr'])
    #     train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
    #     train_meter.log_iter_stats(cur_epoch, cur_iter, writer)
    #     train_meter.iter_tic()
    # # Log epoch stats
    # train_meter.log_epoch_stats(cur_epoch, writer)
    # train_meter.reset()


@torch.no_grad()
def test_epoch(test_loader, model, loss_fun, test_meter, cur_epoch):
    """Evaluates the model on the test set."""
    from taowei.torch2.utils.classif import ProgressMeter
    progress = ProgressMeter(iters_per_epoch=len(test_loader),
        epoch=cur_epoch, split='val', writer=writer, batch_size=cfg.TEST.BATCH_SIZE)
    # Enable eval mode
    model.eval()
    # test_meter.iter_tic()
    # end = time.time()
    from taowei.timer import Timer
    timer = Timer()
    timer.tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # measure data loading time
        # progress.update('data_time', time.time() - end)
        progress.update('data_time', timer.toc(from_last_toc=True))

        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # NOTE: distributed training requires loss to use all the model parameters, total loss is always calculated
        #       Here preds_aux is None in testing mode
        # Perform the forward pass
        if cfg.MODEL.AUXILIARY_WEIGHT > 0.0:
            preds, _ = model(inputs)
        else:
            preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        progress.update('forward_time', timer.toc(from_last_toc=True))
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()

        mb_size = inputs.size(0) * cfg.NUM_GPUS
        progress.update('loss', loss.item(), mb_size)
        progress.update('top1_err', top1_err, mb_size)
        progress.update('top5_err', top5_err, mb_size)

        # measure elapsed time
        # progress.update('batch_time', time.time() - end)
        # end = time.time()
        progress.update('batch_time', timer.toctic())

        if cur_iter % cfg.LOG_PERIOD == 0:
            progress.log_iter_stats(iter=cur_iter, batch_size=mb_size)

    progress.log_epoch_stats()

    #     test_meter.iter_toc()
    #     # Update and log stats
    #     test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
    #     test_meter.log_iter_stats(cur_epoch, cur_iter)
    #     test_meter.iter_tic()
    # # Log epoch stats
    # test_meter.log_epoch_stats(cur_epoch, writer)
    # test_meter.reset()


def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model, loss_fun, and optimizer
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    print('Criterion:\n{}'.format(loss_fun))
    optimizer = optim.construct_optimizer(model)
    print('Optimizer:\n{}'.format(optimizer))
    print('Scheduler Kwargs:\n{}'.format(json.dumps(optim._get_scheduler_kwargs(), indent=4)))
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = checkpoint.load_checkpoint(last_checkpoint, model, optimizer)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = loader.construct_train_loader()
    if hasattr(train_loader, 'dataset'):
        print('Train Dataset:\n{}'.format(train_loader.dataset))
    test_loader = loader.construct_test_loader()
    if hasattr(test_loader, 'dataset'):
        print('Test Dataset:\n{}'.format(test_loader.dataset))
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    # Compute model and loader timings
    if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
        benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch))
    # Evaluate the model first
    if cfg.EVAL_FIRST:
        test_epoch(test_loader, model, loss_fun, test_meter, start_epoch - 1)
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        if cfg.MODEL.DROP_PATH_PROB > 0.0:
            # NOTE: drop path prob for DARTS-like models, not affect training
            assert hasattr(_unwrap_model(model), 'drop_path_prob')
            _unwrap_model(model).drop_path_prob = cfg.MODEL.DROP_PATH_PROB * cur_epoch / cfg.OPTIM.MAX_EPOCH
        # Train for one epoch
        train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(model, optimizer, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.TRAIN.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            test_epoch(test_loader, model, loss_fun, test_meter, cur_epoch)
        # Clear the memory if necessary
        if cfg.CLEAR_MEMORY:
            from taowei.torch2.utils import clear_memory
            clear_memory(verbose=True)


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Load model weights
    checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, loss_fun, test_meter, 0)


def time_model():
    """Times model and data loader."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Create data loaders
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    # Compute model and loader timings
    benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
