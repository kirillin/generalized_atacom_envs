
from mushroom_rl.environments.segway import Segway
import numpy as np
import time

mdp = Segway()

mdp.reset()

while True:

    mdp.step(np.array([0]))
    mdp.render()
    print(mdp._state)
    time.sleep(0.5)
