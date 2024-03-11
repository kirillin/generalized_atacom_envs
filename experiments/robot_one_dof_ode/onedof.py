import numpy as np
from scipy.integrate import odeint

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces
from mushroom_rl.utils.angles import normalize_angle
from mushroom_rl.utils.viewer import Viewer


class OneDof(Environment):
    """
        tau = m ddq + c dq + m g r cos(q)
    """
    
    def __init__(self, random_start=False, random_target=False, is_closedloop=False):
        self.is_closedloop = is_closedloop

        # MDP parameters
        gamma = 0.97

        self._m = 1.0
        self._c = 1.0
        self._r = 0.5   # center mass position
        self._g = 9.81
        self._max_u = 2

        self._random = random_start
        self._random_target = random_target

        high = np.array([np.pi, np.pi])   # limits for state q, dq

        # MDP properties
        dt = 1e-2
        observation_space = spaces.Box(low=-high, high=high)
        action_space = spaces.Box(low=np.array([-self._max_u]),
                                  high=np.array([self._max_u]))
        horizon = 300
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        # target
        self.target_theta = 0
        self.target_point = np.array([1, 0])
        self._state_prev = np.array([0, 0])
        self.rotations = 0

        # Visualization
        self._viewer = Viewer(5 * self._r, 5 * self._r)
        self._last_x = 0

        super().__init__(mdp_info)

    def reset(self, state=None):
        angle = -np.pi/8
        self._state = np.array([angle, 0.])

        self.target_theta = normalize_angle(np.random.uniform(-3.14, 3.14))
        self.target_point = self.fk(self.target_theta)

        return self._state, {}
        
    def step(self, action):
        u = self._bound(action[0], -self._max_u, self._max_u)
        new_state = odeint(self._dynamics, self._state, [0, self.info.dt], (u,))

        self._state = np.array(new_state[-1])
        self._state[0] = normalize_angle(self._state[0])

        self.error = abs(self.target_theta - self._state[0])

        reward = - self.error * 5
        # if self.error < 0.05:
        #     reward += 5

        return self._state, reward, False, {}

    def _f(self, x):
        theta, dtheta = x[0], x[1]
        return np.array([
            dtheta,
            -self._c / self._m * dtheta - self._g * self._r * np.cos(theta)
        ])

    def _G(self, x):
        return np.array([
            0, 
            1 / self._m
        ])

    def _dynamics(self, state, t, u):
        dxdt = self._f(state) + self._G(state) * u
        return dxdt.tolist()

    def get_state(self):
        return self._state

    def get_target(self):
        return self.target_theta
    
    def render(self, record=False):
        start = 2.5 * self._r * np.ones(2)
        end = 2.5 * self._r * np.ones(2)
        end_target = 2.5 * self._r * np.ones(2)

        end[0] += 2 * self._r * np.cos(self._state[0])
        end[1] += 2 * self._r * np.sin(self._state[0])
        # scale = self.error / 2.
        scale = 1.0
        self._viewer.line(start, end, color=(255*scale, 255-255*scale, 0), width=2)

        end_target[0] += 2 * self._r * np.cos(self.target_theta)
        end_target[1] += 2 * self._r * np.sin(self.target_theta)
        self._viewer.line(start, end_target, color=(0, 255, 0), width=1)

        frame = self._viewer.get_frame() if record else None

        self._viewer.display(self.info.dt)

        return frame

    def stop(self):
        self._viewer.close()

    def fk(self, angle):
        """ Returns x-y link position"""
        return np.array([
            2. * self._r * np.cos(angle),
            2. * self._r * np.sin(angle)
        ])

    def ik(self, xy):
        """ Returns link angle"""
        return np.arctan2(xy[1], xy[0])


def main():
    import time
    
    onedof = OneDof()
    onedof.reset(np.array([np.pi/2, 0]))

    while True:

        # pd control
        state = onedof.get_state()
        u = 100. * (onedof.get_target() - state[0]) - 20. * state[1]

        # step intergator
        onedof.step(np.array([u]))

        onedof.render()

        xy = onedof.fk(onedof._state[0])
        q = onedof.ik(xy)
                       
        # print(f"IK: {q}, FK: {xy}")


if __name__ == '__main__':
    main()
    