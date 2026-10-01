import torch
import common

SCALE = 10.0

m = common.build_model()
with torch.no_grad():
    m.lm_head.weight.mul_(SCALE)
    # bias は初期値のまま(0 にしない)

common.report_init(m)          # report_init を作っていなければ measure_init で
common.run(m, tag=f"scale{SCALE}")