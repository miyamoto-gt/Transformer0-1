import numpy as np
MAX_STEP=0.02
GOAL_DISTANSCE=0.02
TRY_STEP=128
HURDLE_RANGE=0.1
center=np.array([0.5,0.5])
goal=np.array([1.0,1.0])
pos=np.array([0.5,0.55]) #位置
contact=False #衝突したか
if np.linalg.norm(pos - center) <= HURDLE_RANGE:
    contact = True
print(contact)