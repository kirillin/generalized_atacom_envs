import mujoco
import numpy as np
from dm_control import mjcf
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils.spaces import Box
# from mushroom_rl.utils.mujoco import *
from mushroom_rl.utils.mujoco import MujocoViewer


class TwoDofMujoco(Environment):

    def __init__(self, xml_file, gamma=0.99, horizon=300, dt=1e-2, timestep=None, n_substeps=1, n_intermediate_steps=1):
        # Create the simulation
        self._model = mujoco.MjModel.from_xml_path(xml_file)
        self._data = mujoco.MjData(self._model)

        self.x = np.concatenate([self._data.qpos.flat, self._data.qvel.flat])

        # time stuff
        if timestep is not None:
            self._model.opt.timestep = timestep
            self._timestep = timestep
        else:
            self._timestep = self._model.opt.timestep
        self._n_intermediate_steps = n_intermediate_steps
        self._n_substeps = n_substeps

        # Observation space and Action space
        observation_space = Box(-np.inf, np.inf, shape=(10,))
        action_space =  Box(-1, 1, shape=(2,))

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)
        super().__init__(mdp_info)

        # Visualization
        self._viewer_params = dict(default_camera_mode="top_static", camera_params=dict(top_static=dict(distance=1.0, elevation=-90.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0]))))
        self._viewer = None

    def seed(self, seed):
        np.random.seed(seed)

    def set_target(self, q, min_radius=0.03):
        x, y = self.fk(q)[0][:2]
        if np.linalg.norm([x,y]) > min_radius:
            # set target marker in simulation
            self._data.qpos[[2,3]] = [x,y]
            self._data.qvel[[2,3]] = [0,0]
            mujoco.mj_forward(self._model, self._data)
            return True
        return False
    
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
        
        # set initial state
        self.x = np.zeros(8)

        # Add noise to state
        self.x[0] = self.x[0] + np.random.uniform(low=-0.1, high=0.1) # position noise
        self.x[1] = self.x[1] + np.random.uniform(low=-0.1, high=0.1)
        self.x[4] = self.x[4] + np.random.uniform(low=-0.005, high=0.005) # velocity noise
        self.x[5] = self.x[5] + np.random.uniform(low=-0.005, high=0.005)

        # Reset simulation
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[:] = np.copy(self.x[:4])
        self._data.qvel[:] = np.copy(self.x[4:])

        # New target
        suc = False
        while not suc:  # generate target non close to a robot base
            self.target_q = np.random.uniform(-np.pi, np.pi, size=2) # generate a new target
            suc = self.set_target(self.target_q)

        # apply initial for simulator
        mujoco.mj_forward(self._model, self._data)
        
        if self._viewer is not None:
            self._viewer.load_new_model(self._model)

        # Reset observation
        observation = self._get_observation()

        return observation, {}

    def step(self, action):
        # do simulation
        action = self._bound(action, -1, 1)             # bound action
        self._data.ctrl[[0,1]] = action                 # apply action
        mujoco.mj_step(self._model, self._data, 1)      # sim. step

        # x \in R^{8} = [q1, q2, x_target, y_target, dq1, dq2, dx_target, dy_target]
        self.x = np.concatenate([self._data.qpos.flat, self._data.qvel.flat])

        # compute reward
        reward = self._get_reward(action)
        
        absorbing = False

        # check if it's close to a joint limit
        for i in range(2):
            if np.pi - np.abs(self.x[i]) < 0.01 * np.pi:
                absorbing = True

        observation = self._get_observation()
        return observation, reward, absorbing, {}

    def _get_reward(self, action):
        target_position = self.x[[2,3]]
        tcp_position = self.fk(self.x[[0,1]])[0][:2]

        vec = tcp_position - target_position
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + reward_ctrl
        return reward

    def render(self, record=False):
        if self._viewer is None:
            self._viewer = MujocoViewer(self._model, self.dt, record=record, **self._viewer_params)
        return self._viewer.render(self._data, record)

    def stop(self):
        if self._viewer is not None:
            self._viewer.stop()
            del self._viewer
            self._viewer = None

    def fk(self, q, body_name='fingertip'):
        self._data.qpos[:len(q)] = q
        mujoco.mj_fwdPosition(self._model, self._data)
        return self._data.body(body_name).xpos.copy(), self._data.body(body_name).xmat.reshape(3, 3).copy()

    @property
    def dt(self):
        return self._timestep * self._n_intermediate_steps * self._n_substeps

def test():
    import time
    xml_file = '/home/kika/path/iros2024/generalized_atacom_envs/experiments/robot_reacher/twodof.xml'

    twodof = TwoDofMujoco(xml_file, )
    
    q = np.array(twodof.x[[0,1]])
    dq = np.array(twodof.x[[4,5]])

    twodof.reset()

    q_des = np.array([0., 0.])
    twodof.set_target(q_des, min_radius=0.03)

    t0 = time.time()
    while True:
        t = time.time() - t0

        q_des = np.array([
            np.pi * np.sin(t), 
            0.9 * np.pi * np.sin(t)]
        )
        # twodof.set_target(q_des, min_radius=0.03)

        q = np.array(twodof.x[[0,1]])
        dq = np.array(twodof.x[[4,5]])
        u = 1. * (q_des - q) - 0.2 * dq
        
        twodof.step(u)
        twodof.render()

        if t > 2.:
            t0 = time.time()
            twodof.reset()


if __name__ == '__main__':
    test()
