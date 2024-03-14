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

        # Plant parameters
        self._m = 1.0
        self._c = 1.0
        self._r = 0.5   # center mass position
        self._g = 9.81
        self._max_u = 20.   # value hardcoded in actor network

        # memory vars.
        self.x = np.zeros(2) # plant state
        self.target_theta = 0

        # MDP parameters
        gamma = 0.97
        horizon = 300
        dt = 1e-2

        # MDP properties
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
        action_space = spaces.Box(low=np.array([-self._max_u]),
                                  high=np.array([self._max_u]))
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        # Visualization
        self._viewer = Viewer(5 * self._r, 5 * self._r)
        self._last_x = 0

        super().__init__(mdp_info)

    def reset(self, state=None):
        self.x = np.array([-np.pi/8, 0]) # reset initial state of plant
        self.target_theta = np.random.uniform(-np.pi, np.pi) # generate new target
        return self._get_obs(), {}
        
    def step(self, action):
        # do simulation
        u = self._bound(action[0], -self._max_u, self._max_u)
        new_x = odeint(self._dynamics, self.x, [0, self.info.dt], (u,))
        self.x = np.array(new_x[-1]) # NON NORMALIZED
        
        # compute reward
        observation = self._get_obs()
        reward = self._get_reward(action)
        # self._state = observation
        return observation, reward, False, {}

    def _get_reward(self, action):
        vec = self.fk(self.x[0]) - self.fk(self.target_theta)
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + reward_ctrl
        return reward

    def _f(self, x):
        theta, dtheta = x[0], x[1]
        return np.array([
            dtheta,
            0#-self._c / self._m * dtheta - self._g * self._r * np.cos(theta)
        ])

    def _G(self, x):
        return np.array([
            0, 
            1 / self._m
        ])

    def _dynamics(self, state, t, u):
        dxdt = self._f(state) + self._G(state) * u
        return dxdt.tolist()

    def _get_obs(self):
        """
            Observation space
                0 cos(theta)
                1 sin(theta)
                2 x_target
                3 y_target
                4 dot_theta
                5 x_tcp - x_target
                6 y_tcp - y_target
        """
        theta, dot_theta = self.x.flatten()
        return np.concatenate(
            [
                [np.cos(theta)],
                [np.sin(theta)],
                self.fk(self.target_theta).flatten(),
                [dot_theta],
                (self.fk(theta) - self.fk(self.target_theta)).flatten(),
            ]
        )       

    def render(self, record=False):
        _target = 2.5 * self._r * np.ones(2)
        _state = 2.5 * self._r * np.ones(2)

        start = 2.5 * self._r * np.ones(2)
        end = 2.5 * self._r * np.ones(2)
        end_target = 2.5 * self._r * np.ones(2)

        end[0] += 2 * self._r * np.cos(self.x[0])
        end[1] += 2 * self._r * np.sin(self.x[0])
        # scale = self.error / 2.
        scale = 1.0
        self._viewer.line(start, end, color=(255*scale, 255-255*scale, 0), width=2)

        end_target[0] += 2 * self._r * np.cos(self.target_theta)
        end_target[1] += 2 * self._r * np.sin(self.target_theta)
        self._viewer.line(start, end_target, color=(0, 255, 0), width=1)

        self._viewer.circle(end_target, 0.1, color=(0, 255, 0), width=1)
        self._viewer.circle(end, 0.1, color=(255, 0, 0), width=1)

        # just to check fk and vis
        _target += self.fk(self.target_theta)
        _state += self.fk(self.x[0])
        self._viewer.circle(_target, 0.05, color=(255, 255, 0), width=1)
        self._viewer.circle(_state, 0.05, color=(255, 0, 255), width=1)

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
        # step intergator
        onedof.step(np.array([1]))

        onedof.render()

        xy = onedof.fk(onedof._state[0])
        q = onedof.ik(xy)

        # print(f"IK: {q}, FK: {xy}")


if __name__ == '__main__':
    main()
    