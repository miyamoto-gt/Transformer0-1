import common
m=common.build_model()
common.report_init(m)
common.run(m,tag="shuffle_y",shuffle_y=True)
