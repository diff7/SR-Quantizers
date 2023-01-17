import os
import torch
import random
import utils
import pandas as pd
from datasets import ValidationSet
from QSB.tools import get_flops_and_memory


def run_val(model, cfg_val, save_dir, device):
    # set default gpu device id
    torch.cuda.set_device(device)
    val_data = ValidationSet(cfg_val)

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        # sampler=sampler_val,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    ssim, score_val = validate(val_loader, model, device, save_dir)

    flops_32, mem = get_flops_and_memory(
        model, input_size=[1, 3, 32, 32], device=device
    )
    flops_256, mem = get_flops_and_memory(
        model, input_size=[1, 3, 256, 256], device=device
    )

    mb_params = utils.param_size(model)

    # during search pahse flops are multipleid by alpha which is differentiable tensor

    if not isinstance(flops_32, float):
        flops_32 = flops_32.detach().cpu()

    if not isinstance(flops_256, float):
        flops_256 = flops_256.detach().cpu()

    return (
        ssim,
        score_val,
        flops_32,
        flops_256,
        mb_params,
    )


def validate(valid_loader, model, device, save_dir):
    psnr_meter = utils.AverageMeter()
    ssim_meter = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for (
            X,
            y,
            x_path,
            y_path,
        ) in valid_loader:
            X, y = X.to(device, non_blocking=True), y.to(device)
            N = X.size(0)

            preds = model(X).clamp(0.0, 1.0)

            psnr = utils.compute_psnr(preds, y)
            ssim = utils.compute_ssim(preds, y)
            psnr_meter.update(psnr, N)
            ssim_meter.update(ssim, N)

    indx = random.randint(0, len(x_path) - 1)
    utils.save_images(
        save_dir,
        x_path[indx],
        y_path[indx],
        preds[indx],
        cur_iter=0,
        logger=None,
    )

    return ssim_meter.avg, psnr_meter.avg


def dataset_loop(valid_cfg, model, logger, save_dir, device):
    model.to(device)

    df = pd.DataFrame(
        columns=[
            "Model size",
            "BitOps(32x32)",
            "BitOps(256x256)",
            "PSNR",
            "SSIM",
        ]
    )
    for dataset in valid_cfg:
        os.makedirs(os.path.join(save_dir, str(dataset)), exist_ok=True)
        ssim, score_val, flops_32, flops_256, mb_params = run_val(
            model,
            valid_cfg[dataset],
            os.path.join(save_dir, str(dataset)),
            device,
        )
        logger.info("\n{}:".format(str(dataset)))
        logger.info("Model size = {:.3f} MB".format(mb_params))
        logger.info("BitOps = {:.2e} operations 32x32".format(flops_32))
        logger.info("BitOps = {:.2e} operations 256x256".format(flops_256))
        logger.info("PSNR = {:.3f}%".format(score_val))
        logger.info("SSIM = {:.3f}%".format(ssim))

        df.loc[str(dataset)] = [mb_params, flops_32, flops_256, score_val, ssim]
    df.to_csv(os.path.join(save_dir, "..", "validation_df.csv"))


if __name__ == "__main__":

    # RUN TESTS HERE
    pass
