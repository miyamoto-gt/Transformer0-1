import common
from broken import BlockNoResidual

m=common.build_model()
for i in range(len(m.blocks)):
    m.blocks[i].__class__ =BlockNoResidual

common.report_init(m)
common.run(m,tag="no_residual")