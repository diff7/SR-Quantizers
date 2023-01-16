import torch
from QSB.qconfig import QConfig
from QSB.tools import (
    replace_modules,
    set_signle,
    get_flops_and_memory,
    prepare_and_get_params,
)

from models.IMDN.architecture import IMDN

qconfig = QConfig()
model = IMDN()

input_x = torch.randn(10, 3, 28, 28)
model(input_x)

replace_modules(model, qconfig, verbose=True)
print(model)
model(input_x)
model, main_params, alpha_params, alpha_names = prepare_and_get_params(model, qconfig)

print("FLOPS:")
print(get_flops_and_memory(model))
set_signle(model)
print(model)
model(input_x)
print("FLOPS:")
print(get_flops_and_memory(model))
