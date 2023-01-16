import os
import torch
import argparse
from omegaconf import OmegaConf as omg

from QSB.qconfig import QConfig
from QSB.tools import (
    replace_modules,
    set_signle,
    get_flops_and_memory,
    prepare_and_get_params,
)


from sr_trainer import run_train, train_setup
from validate_sr import dataset_loop
from models.IMDN.architecture import IMDN
from models.ESCPCN.model import espcn_x4 as Net  


# (1) PROBLEM activation functions quantization / double usage !!

parser = argparse.ArgumentParser()
parser.add_argument(
    "-g",
    "--gpu",
    type=int,
    default=1,
    help="gpu number",
)


parser.add_argument(
    "-n",
    "--name",
    default='debug',
    help="experiment name",
)

parser.add_argument(
    "-m",
    "--mode",
    default='search',
    help="modes: search or train",
)


args = parser.parse_args()

MODE = args.mode  # 'train'

if MODE == "search":
    search = True
else:
    search = False

qconfig = QConfig(
    act_quantizer="HWGQ",
    weight_quantizer="LSQ",
    noise_search=False,
    bits=[8,4,2],
)

# qconfig = QConfig()


#model = IMDN()
model = Net()
print("Initial params:", len([p for p in model.parameters()]))

CFG_PATH = "./sr_config.yaml"

cfg = omg.load(CFG_PATH)
cfg.env.run_name = args.name

wh = cfg.dataset.crop_size
input_x = torch.randn(10, 3, wh, wh)
model(input_x)

model, main_params, alpha_params, alpha_names = prepare_and_get_params(
    model, qconfig
)
print("MAIN PARAMS", len(main_params), "ALPHAS", len(alpha_names))

print("FLOPS:",get_flops_and_memory(model, input_size=(1, 3, 28, 28)))


cfg, writer, logger, log_handler = train_setup(cfg, mode=MODE)

# set_signle(model)

run_train(
    model,
    main_params,
    cfg,
    writer,
    logger,
    log_handler,
    alphas=alpha_params,
    alphas_names=alpha_names,
    search=search,
)

save_dir = os.path.join(cfg.env.save_path, "FINAL_VAL")
os.makedirs(save_dir, exist_ok=True)

valid_cfg = omg.load("./validation_conf.yaml")

cfg, writer, logger, log_handler = train_setup(cfg, mode=MODE)

dataset_loop(valid_cfg, model, logger, save_dir, cfg.env.gpu)

print('SETTING SINGLE:')
set_signle(model, device=cfg.env.gpu)
model.to(cfg.env.gpu)
print("FLOPS:",get_flops_and_memory(model, input_size=(1, 3, 28, 28),device=cfg.env.gpu))
dataset_loop(valid_cfg, model, logger, save_dir, cfg.env.gpu)