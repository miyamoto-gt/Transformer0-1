import common
LR=1e-1
TAG="lr1e-1"
m=common.build_model()
common.report_init(m)
common.run(m,tag=TAG,lr_override=LR)