import numpy as np
from scipy.integrate import odeint

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces
from mushroom_rl.utils.angles import normalize_angle
from mushroom_rl.utils.viewer import Viewer

from gym.utils import seeding


class OneDof(Environment):
    """
        tau = m ddq + c dq + m g r cos(q)
    """
    
    def __init__(self,):

        self._m = 1.
        self._r = 0.5
        self._max_dq = 1.0
        self._max_u = 1.0
        self.goal = 0.0

        high = np.array([np.pi, self._max_dq])
        action_space = spaces.Box(low=-self._max_u, high=self._max_u, shape=(1,))
        observation_space = spaces.Box(low=-high, high=high)

        # mdp params and props
        gamma = 0.97
        horizon = 300
        dt = 1e-2
       
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        # Visualization
        self._viewer = Viewer(5 * self._r, 5 * self._r)
        self._last_x = 0

        # reset
        self.np_random = None
        self.seed()
        self.reset()

        super().__init__(mdp_info)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, state=None):
        high = np.array([np.pi, self._max_dq])
        # self._state = self.np_random.uniform(low=-high, high=high)
        self._state = np.zeros(2)
        self.target_theta = self.np_random.uniform(low=-high, high=high)[0]
        return self._state, {}

    # def setup(self, state=None):
    #     super(OneDof, self).setup(state)

    def step(self, action):
        u = self._bound(action[0], -self._max_u, self._max_u)
        new_state = odeint(self._dynamics, self._state, [0, self.info.dt], (u,))

        self._state = np.array(new_state[-1])
        self._state[0] = normalize_angle(self._state[0])

        # reward = - ((self.target_theta - self._state[0]) ** 2 + 0.1 * self._state[1] ** 2)
        error = np.abs(self.target_theta - self._state[0])
        absorbing = False
        if error < 0.01:
            reward = 100
            absorbing = True
        else:
            reward = -error**2
        
        
        return self._state, reward, absorbing, {}

    def _f(self, x):
        theta, dtheta = x[0], x[1]
        return np.array([
            dtheta,
            0 # -self._c / self._m * dtheta - self._g * self._r * np.cos(theta)
        ])

    def _G(self, x):
        return np.array([
            0, 
            1 / self._m
        ])

    def _dynamics(self, state, t, u):
        x = np.array([state[0], state[1]])
        dxdt = self._f(x) + self._G(x) * u
        return dxdt.tolist()

    def _pd_control(self, state, t, u):
        """desired angle control"""
        x = np.array([state[0], state[1]])
        k = 10.
        b = 2  * np.sqrt(self._m * k)
        return k * (u - x[0]) - b * x[1]

    def render(self, record=False):
        start = 2.5 * self._r * np.ones(2)
        end = 2.5 * self._r * np.ones(2)
        end_target = 2.5 * self._r * np.ones(2)

        end[0] += 2 * self._r * np.cos(self._state[0])
        end[1] += 2 * self._r * np.sin(self._state[0])
        self._viewer.line(start, end, color=(255, 255, 255), width=1)

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
        onedof.step(np.array([0.01]))
        onedof.render()

        xy = onedof.fk(onedof._state[0])
        q = onedof.ik(xy)
                       
        print(f"IK: {q}, FK: {xy}")


if __name__ == '__main__':
    main()
    