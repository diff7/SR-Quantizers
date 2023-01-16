""" Search cell """

import os
import torch
import torch.nn as nn
import logging
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import utils

from omegaconf import OmegaConf as omg


from QSB.qconfig import QConfig
from QSB.tools import get_flops_and_memory


def train_setup(cfg, mode="train"):

    # INIT FOLDERS & cfg

    cfg.env.save_path = utils.get_run_path(
        cfg.env.log_dir, f"{mode}" + cfg.env.run_name
    )
    utils.save_scripts(cfg.env.save_path)
    log_handler = utils.LogHandler(cfg.env.save_path + "/log.txt")
    logger = log_handler.create()

    # FIX SEED
    np.random.seed(cfg.env.seed)
    torch.cuda.set_device(cfg.env.gpu)
    np.random.seed(cfg.env.seed)
    torch.manual_seed(cfg.env.seed)
    torch.cuda.manual_seed_all(cfg.env.seed)
    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(log_dir=os.path.join(cfg.env.save_path, "board"))

    writer.add_hparams(
        hparam_dict={str(k): str(cfg[k]) for k in cfg},
        metric_dict={f"{mode}/train/loss": 0},
    )

    omg.save(cfg, os.path.join(cfg.env.save_path, "config.yaml"))

    return cfg, writer, logger, log_handler


def run_train(
    model,
    main_params,
    cfg,
    writer,
    logger,
    log_handler,
    alphas=None,
    alphas_names=None,
    search=True,
):
    # cfg, writer, logger, log_handler = train_setup(cfg)
    logger.info("Logger is set - training start")

    if search:
        mode = "search"
        assert (
            len(alphas_names) > 0
        ), "A list of searchable quantization (alphas) bits should be greater than zero during bilevel search"

        assert len(alphas) == len(alphas_names)
    else:
        mode = "train"

    # set default gpu device id
    device = cfg.env.gpu
    torch.cuda.set_device(device)

    model.to(cfg.env.gpu)

    train_loader, val_loader = utils.get_data_loaders(cfg)

    if cfg.train.load_path is not None:
        model.load_state_dict(torch.load(cfg.train.load_path))
        model.eval()
        print(f"loaded a model from: {cfg.train.load_path}")

    criterion = nn.L1Loss().to(device)

    model = model.to(device)
    wh = cfg.dataset.crop_size
    input_size = [1, 3, wh, wh]
    flops, _ = get_flops_and_memory(
        model, input_size=input_size, device=cfg.env.gpu
    )
    flops_loss = utils.FlopsLoss(flops, cfg.train.penalty)

    # weights optimizer
    if cfg.train.optimizer == "sgd":
        w_optim = torch.optim.SGD(
            main_params,
            cfg.train.w_lr,
            momentum=cfg.train.w_momentum,
            weight_decay=cfg.train.w_weight_decay,
        )
        print("USING SGD")
    else:
        w_optim = torch.optim.Adam(
            main_params,
            cfg.train.w_lr,
            weight_decay=cfg.train.w_weight_decay,
        )
        print("USING ADAM")

    if search:
        alpha_optim = torch.optim.Adam(
            alphas,
            cfg.train.alpha_lr,
            betas=(0.5, 0.999),
            weight_decay=cfg.train.alpha_weight_decay,
        )
    else:
        alpha_optim = None

    scheduler = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(
            w_optim, cfg.train.epochs
        ),
        "linear": torch.optim.lr_scheduler.StepLR(
            w_optim, step_size=3, gamma=0.8
        ),
    }

    lr_scheduler = scheduler[cfg.train.lr_scheduler]

    # training loop
    best_score = 1e3
    cur_step = 0
    for epoch in range(cfg.train.epochs):
        lr = lr_scheduler.get_last_lr()[0]
        print("LR: ", lr)

        if search:
            alpha_string = []
            for a, n in zip(alphas, alphas_names):
                alpha_string.append(f"{n} alpha: {a}")

            logger.info("\n".join(alpha_string))

        # training
        score_train, cur_step, best_current_flops = train(
            train_loader,
            model,
            criterion,
            w_optim,
            alpha_optim,
            lr,
            epoch,
            writer,
            logger,
            cfg,
            device,
            cur_step,
            flops_loss,
            search,
            alphas,
            alphas_names,
        )

        lr_scheduler.step()
        # validation
        score_val = validate(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            writer,
            cfg,
            device,
        )

        # save
        if best_score > score_val:
            best_score = score_val
            best_flops = best_current_flops
            writer.add_scalar(f"{mode}/best_val", best_score, epoch)
            writer.add_scalar(f"{mode}/best_flops", best_flops, epoch)

            is_best = True

            utils.save_checkpoint(model, cfg.env.save_path, is_best)
            print("")

        print("best current", best_current_flops)
        writer.add_scalars(
            "loss/search", {"val": best_score, "train": score_train}, epoch
        )
        logger.info("Final best LOSS = {:.3f}".format(best_score))

    # FINISH TRAINING
    log_handler.close()
    logging.shutdown()


def train(
    train_loader,
    model,
    criterion,
    w_optim,
    alpha_optim,
    lr,
    epoch,
    writer,
    logger,
    cfg,
    device,
    cur_step,
    flops_loss,
    search,
    alphas=None,
    alphas_names=None,
):
    loss_meter = utils.AverageMeter()

    if search:
        mode = "search"
    else:
        mode = "train"

    writer.add_scalar("search/train/lr", lr, cur_step)
    model.train()

    for step, (trn_X, trn_y, _, _) in enumerate(train_loader):
        trn_X, trn_y = (
            trn_X.to(device, non_blocking=True),
            trn_y.to(device, non_blocking=True),
        )

        # True & 1 False
        # True & 2 True
        if search and step % 2 == 0:
            # optimize alpha on every second step

            preds = model(trn_X)
            flops, mem = get_flops_and_memory(model, use_cached=True)
            f_loss = flops_loss(flops)
            loss = f_loss + criterion(preds, trn_y)
            loss.backward()
            alpha_optim.step()
            alpha_optim.zero_grad()

        # False & 1 -> False
        # False & 2 -> True
        else:
            if step == len(train_loader) - 1:
                utils.log_weigths_hist(model, writer, epoch)

            preds = model(trn_X)
            flops, mem = get_flops_and_memory(model, use_cached=True)
            loss_w = criterion(preds, trn_y)
            loss_w.backward()

            if step == len(train_loader) - 1:
                utils.log_weigths_hist(model, writer, epoch)

            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.w_grad_clip)
            w_optim.step()
            w_optim.zero_grad()

        N = trn_X.size(0)

        if step % 2 != 0:
            loss_meter.update(loss_w.item(), N)

        if step % cfg.env.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss: {losses.avg:.3f} BitOps: {flops:.4e}, BitOps Loss:{f_loss:.3e}".format(
                    epoch + 1,
                    cfg.train.epochs,
                    step,
                    len(train_loader) - 1,
                    losses=loss_meter,
                    flops=flops,
                    f_loss=f_loss,
                )
            )

        if step % 2 != 0:
            writer.add_scalar("{mode}/train/loss", loss_w, cur_step)

        writer.add_scalar(f"{mode}/train/flops_loss", flops, cur_step)
        writer.add_scalar(f"{mode}/train/weighted_flops", flops, cur_step)
        writer.add_scalar(f"{mode}/train/weighted_memory", mem, cur_step)

        cur_step += 1

    logger.info(
        "Train: [{:2d}/{}] Final LOSS {:.3f}".format(
            epoch + 1, cfg.train.epochs, loss_meter.avg
        )
    )

    return loss_meter.avg, cur_step, flops


def validate(
    valid_loader, model, criterion, epoch, logger, writer, cfg, device
):

    loss_meter = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y, x_path, y_path) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(
                device, non_blocking=True
            )

            N = X.size(0)
            preds = model(X)

            loss = criterion(preds, y)

            loss_meter.update(loss.item(), N)
            if step % cfg.env.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss: {losses.avg:.3f} ".format(
                        epoch + 1,
                        cfg.train.epochs,
                        step,
                        len(valid_loader) - 1,
                        losses=loss_meter,
                    )
                )

    logger.info(
        "Valid: [{:2d}/{}] Final LOSS {:.3f}".format(
            epoch + 1, cfg.train.epochs, loss_meter.avg
        )
    )

    utils.save_images(
        cfg.env.save_path, x_path[0], y_path[0], preds[0], epoch, writer
    )
    return loss_meter.avg
