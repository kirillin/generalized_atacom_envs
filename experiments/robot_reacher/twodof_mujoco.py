import mujoco
import numpy as np
from dm_control import mjcf
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.utils.mujoco import *
from mushroom_rl.utils.viewer import Viewer


class TwoDofMujoco(Environment):

    def __init__(self, xml_file, gamma=0.99, horizon=300, dt=1e-2):
        # Create the simulation
        self._model = mujoco.MjModel.from_xml_path(xml_file)
        self._data = mujoco.MjData(self._model)

        self.x = np.concatenate([self._data.qpos.flat, self._data.qvel.flat])

        # Observation space and Action space
        observation_space = Box(-np.inf, np.inf, shape=(10,))
        action_space =  Box(-1, 1, shape=(2,))

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)
        super().__init__(mdp_info)

        # Visualization
        self._viewer = Viewer(4, 4)

    def seed(self, seed):
        np.random.seed(seed)

    def set_target(self, q):
        # TODO check for a base frame?
        x, y = self.fk(q)[0][:2]

        # set target marker in simulation
        self._data.qpos[[2,3]] = [x,y]
        mujoco.mj_forward(self._model, self._data)
        # self._data.joint('target_x').qpos = x
        # self._data.joint('target_y').qpos = y
    
    def _get_observation(self):
        # x \in R^{8}:
        #   0  1  2 3 4   5    6  7
        #   q1 q2 x y dq1 dq2  dx dy
        q = self.x[[0,1]]
        dq = self.x[[4,5]]
        target_position = self.x[[2,3]]
        tcp_position = self.fk(q)[0][:2]

        observation = np.concatenate([
            np.cos(q),
            np.sin(q),
            target_position,
            dq,
            (tcp_position - target_position)
        ])
        return observation

    def reset(self, state=None):
        # Reset simulation
        mujoco.mj_resetData(self._model, self._data)

        # add noise to state
        self.x[0] = self.x[0] + np.random.uniform(low=-0.1, high=0.1) # position noise
        self.x[1] = self.x[1] + np.random.uniform(low=-0.005, high=0.005) # velocity noise

        # New target
        self.target_q = np.random.uniform(-np.pi, np.pi, size=2) # generate new target
        self.set_target(self.target_q)

        # Reset observation
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action):
        # do simulation
        action = self._bound(action, -1, 1)         # bound action
        self._data.ctrl[[0,1]] = action               # apply action
        mujoco.mj_step(self._model, self._data, 1)  # sim. step

        # x \in R^{8} = [q1, q2, x_target, y_target, 
        #                dq1, dq2, dx_target, dy_target]
        self.x = np.concatenate([self._data.qpos.flat, self._data.qvel.flat])
        # print(self.x[[2,3]])

        # compute reward
        reward = self._get_reward(action)
        
        observation = self._get_observation()
        return observation, reward, False, {}

    def _get_reward(self, action):
        target_position = self.x[[2,3]]
        tcp_position = self.fk(self.x[[0,1]])[0][:2]

        vec = tcp_position - target_position
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + reward_ctrl
        return reward

    def render(self, record=False):
        _target = 2. * np.ones(2)
        
        start = 2. * np.ones(2)
        end = 1 * np.ones(2)
        end2 = 1 * np.ones(2)

        end[0] = 1 * np.cos(self.x[0])
        end[1] = 1 * np.sin(self.x[0])
        self._viewer.circle(start, 1, color=(100, 100, 100), width=1)

        end2[0] = 1 * np.cos(self.x[1])
        end2[1] = 1 * np.sin(self.x[1])
        self._viewer.circle(end + start, 1, color=(100, 100, 100), width=1)

        self._viewer.line(start, end + start, color=(255, 255, 0), width=4)
        self._viewer.line(end + start, end + start + end2, color=(255, 255, 0), width=2)

        self._viewer.circle(end + start + end2, 0.1, color=(255, 255, 0), width=2)

        _target += self.x[[2,3]] * 10
        self._viewer.circle(_target, 0.1, color=(255, 0, 0), width=2)
        
        frame = self._viewer.get_frame() if record else None
        self._viewer.display(self.info.dt)
        return frame


    def stop(self):
        self._viewer.close()

    def fk(self, q, body_name='fingertip'):
        self._data.qpos[:len(q)] = q
        mujoco.mj_fwdPosition(self._model, self._data)
        return self._data.body(body_name).xpos.copy(), self._data.body(body_name).xmat.reshape(3, 3).copy()


def test():
    xml_file = '/home/kika/path/iros2024/generalized_atacom_envs/experiments/robot_reacher/twodof.xml'
    towdof = TwoDofMujoco(xml_file)
    
    q = np.array(towdof.x[[0,1]])
    dq = np.array(towdof.x[[4,5]])

    q_des = np.array([np.pi/2, np.pi/2])
    towdof.reset()
    while True:
        u = 60. * (q_des - q) - 20. * dq
        towdof.step(u)
        # print(towdof.x)
        q = np.array(towdof.x[[0,1]])
        dq = np.array(towdof.x[[4,5]])
        # print(towdof.fk(q)[0])
        towdof.render()
        q_des[0] += 0.002
        q_des[1] += 0.01
        # if q_des[0] >= np.pi/2:
        #     towdof.reset()
        #     q_des[0] = 0


if __name__ == '__main__':
    test()
        