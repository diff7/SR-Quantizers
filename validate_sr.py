import os
import torch
import random
from omegaconf import OmegaConf as omg
import utils
from datasets import ValidationSet
import pandas as pd
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
    return ssim, score_val, flops_32.detach().cpu(), flops_256.detach().cpu(), mb_params


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
    df = pd.DataFrame(
        columns=["Model size", "BitOps(32x32)", "BitOps(256x256)", "PSNR", "SSIM"]
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

    # ! IMPORTANT NON WORKING PART - REQUIERS refactoring

    CFG_PATH = "./sr_models/valsets4x.yaml"
    valid_cfg = omg.load(CFG_PATH)
    run_name = "TEST_2"
    genotype_path = "/home/dev/2021_09/QuanToaster/genotype_example_sr.gen"
    weights_path = None  # "/home/dev/data_main/LOGS/SR/11_2022/TUNE/Basic_With_ESA-2022-11-22-13/best.pth.tar"
    log_dir = "/home/dev/data_main/LOGS/SR/11_2022/TUNE/"
    save_dir = os.path.join(log_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    channels = 3
    repeat_factor = 16
    device = 1

    with open(genotype_path, "r") as f:
        genotype = from_str(f.read())

    logger = utils.get_logger(save_dir + "/validation_log.txt")
    logger.info(genotype)

    # model = RFDN()
    # model.to(device)

    model = get_model(
        weights_path,
        device,
        genotype,
        c_fixed=36,
        channels=3,
        scale=4,
        body_cells=3,
        skip_mode=True,
    )
    # print(count_Flops(model))
    dataset_loop(valid_cfg, model, logger, save_dir, device)
