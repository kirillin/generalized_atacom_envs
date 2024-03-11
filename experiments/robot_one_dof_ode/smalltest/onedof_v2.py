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
    
    def __init__(self):
        
        self._max_u = 20
        self._r = 0.5
        
        dt = 1e-2
        gamma = 0.97
        horizon = 300
        high = np.array([1,1]) # cos(theta), sin(theta)

        observation_space = spaces.Box(low=-high, high=high)
        action_space = spaces.Box(low=np.array([-self._max_u]),
                                  high=np.array([self._max_u]))
        
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        # Visualization
        self._viewer = Viewer(5 * self._r, 5 * self._r)
        self._last_x = 0

        self._goal = 0
        self._memory_of_dot_theta = 0

        super().__init__(mdp_info)

    def theta_to_obs(self, theta):
        return np.array([np.cos(theta), np.sin(theta)])

    def obs_to_theta(self, obs):
        return np.arctan2(obs[1], obs[0])

    def reset(self, state=None):
        # initial state
        theta_0 = -np.pi/8
        self._state = self.theta_to_obs(theta_0)

        # new target
        self.target_theta = normalize_angle(np.random.uniform(-3.14, 3.14))
        self.target_point = self.fk(self.target_theta)

        return self._state, {}

    def step(self, action):
        u = self._bound(action[0], -self._max_u, self._max_u)
        
        # unpack from observation state
        theta = normalize_angle(self.obs_to_theta(self._state))
        dot_theta = self._memory_of_dot_theta
        x = np.array([theta, dot_theta])

        # integrate for dt
        new_x = odeint(self._dynamics, x, [0, self.info.dt], (u,))

        # pack to observation state
        theta, self._memory_of_dot_theta = np.array(new_x[-1])
        self._state = self.theta_to_obs(normalize_angle(theta))

        self.goal = np.linalg.norm(self.fk(self.target_theta) - self.fk(self.obs_to_theta(self._state)))

        reward = - self.goal - 0.01 * u**2

        if self.goal < 0.05:
            reward = 5

        return self._state, reward, False, {}

    def _f(self, x):
        theta, dtheta = x[0], x[1]
        return np.array([
            dtheta,
            0 # -self._c / self._m * dtheta - self._g * self._r * np.cos(theta)
        ])

    def _G(self, x):
        return np.array([
            0, 
            1
        ])

    def _dynamics(self, state, t, u):
        x = np.array([state[0], state[1]])
        dxdt = self._f(x) + self._G(x) * u
        return dxdt.tolist()

    def render(self, record=False):
        theta = self.obs_to_theta(self._state)
        start = 2.5 * self._r * np.ones(2)
        end = 2.5 * self._r * np.ones(2)
        end_target = 2.5 * self._r * np.ones(2)

        end[0] += 2 * self._r * np.cos(theta)
        end[1] += 2 * self._r * np.sin(theta)
        scale = self.goal / 2.
        self._viewer.line(start, end, color=(255*scale, 255-255*scale, 0), width=1)

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
    onedof.reset()

    while True:
        onedof.step(np.array([1]))
        onedof.render()

        xy = onedof.fk(onedof._state[0])
        q = onedof.ik(xy)
                       
        # print(f"IK: {q}, FK: {xy}")


if __name__ == '__main__':
    main()
    