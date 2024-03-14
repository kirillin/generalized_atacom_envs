from onedof import OneDof
mdp = OneDof()

from mushroom_rl.environments.inverted_pendulum import InvertedPendulum
mdp = InvertedPendulum(max_u=20.)


mdp.reset()

import time

while True:
    q = mdp._state[0]
    dq = mdp._state[1]
    u = 100. * (-1. - q) - 20. * dq
    mdp.step([u])
    mdp.render()
    time.sleep(0.1)