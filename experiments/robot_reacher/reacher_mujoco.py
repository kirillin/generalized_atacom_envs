import copy
import mujoco
import numpy as np
from dm_control import mjcf
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils.spaces import Box
# from mushroom_rl.utils.mujoco import *
from mushroom_rl.utils.mujoco import MujocoViewer


class TwoDofMujoco(Environment):
    """
        x = state \in R^{2*nq+4} = [q1, q2, x_target, y_target dq1, dq2, dx_target, dy_target]
            where q -- robot, x_target,y_target -- target 

        observation \in R^{2*nq+nq+4} = [q1, q2, x_tcp, y_tcp, dq1, dq2, (x_tcp - x_target), (y_tcp - y_target)]

    """
    def __init__(self, xml_file='/home/kika/path/iros2024/generalized_atacom_envs/experiments/robot_reacher/assests/onedof.xml',
                  gamma=0.99, horizon=300, dt=1e-2, timestep=None, n_substeps=1, n_intermediate_steps=1, debug_gui=True):

        print("vec_env: ", self)
        # Create the simulation
        self._model = mujoco.MjModel.from_xml_path(xml_file)
        self._data = mujoco.MjData(self._model)

        self.nq = self._model.nq - 2    # minus target size
        self.nu = self._model.nu

        self.x = np.concatenate([self._data.qpos.flat, self._data.qvel.flat])
        self.nx = self.x.size

        # time stuff
        if timestep is not None:
            self._model.opt.timestep = timestep
            self._timestep = timestep
        else:
            self._timestep = self._model.opt.timestep
        self._n_intermediate_steps = n_intermediate_steps
        self._n_substeps = n_substeps

        # Observation space and Action space
        observation_shape = (4 + self.nq + self.nq * 2, )
        observation_space = Box(-np.inf, np.inf, shape=observation_shape)
        action_space =  Box(-1, 1, shape=(1,))

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)
        super().__init__(mdp_info)

        # Visualization
        self.debug_gui = debug_gui
        self._viewer_params = dict(default_camera_mode="top_static", camera_params=dict(top_static=dict(distance=1.0, elevation=-90.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0]))))
        self._viewer = None

        # memory.
        self.sum = 0
        self.count = 0        

    def get_avg_dist(self):
        if self.count > 0:
            return self.sum / self.count
        return np.inf

    def seed(self, seed):
        np.random.seed(seed)

    def get_state_q(self):
        return self.x[:self.nq]

    def get_state_dq(self):
        return self.x[self.nq + 2 : 2 * self.nq + 2].copy()

    def get_state_target(self):
        return self.x[self.nq : self.nq + 2].copy()

    def get_state_dtarget(self):
        return self.x[-2:].copy()

    def get_jnt_limmits(self):
        low_limits = self._model.jnt_range[: self.nq, 0]
        high_limits = self._model.jnt_range[: self.nq, 1]
        low_limits[ low_limits == 0.] = -np.inf
        high_limits[ high_limits == 0.] = np.inf
        return low_limits, high_limits

    def set_target(self, q, min_radius=0.0):
        x, y = self.fk(q)[0][:2]
        if np.linalg.norm([x,y]) > min_radius:
            # set target marker in simulation
            self._data.qpos[-2:] = [x,y]
            self._data.qvel[-2:] = [0,0]
            mujoco.mj_forward(self._model, self._data)
            return True
        return False

    def _get_observation(self):
        # x \in R^{8}:
        #   0  1  2 3 4   5    6  7
        #   q1 q2 x y dq1 dq2  dx dy
        q = self.get_state_q()
        dq = self.get_state_dq()
        target_position = self.get_state_target()
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
        if state == None:
            self.x = np.zeros(self.nx)
            self.x[:self.nq] = [0.5] * self.nq
        else:
            self.x = state

        # Add noise to state
        for i in range(self.nq):
            self.x[i] = self.x[i] + np.random.uniform(low=-0.1, high=0.1) # position noise

            offset = i + self.nq + 2
            self.x[offset] = self.x[offset] + np.random.uniform(low=-0.005, high=0.005) # velocity noise

        # Reset simulation
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[:] = np.copy(self.x[: self.nx // 2])
        self._data.qvel[:] = np.copy(self.x[self.nx // 2 :])

        # New target
        suc = False
        # while not suc:  # generate target non close to a robot base
        # extract limits form model
        low_limits, high_limits = self.get_jnt_limmits()

        # generate a new target
        self.target_q = np.random.uniform(low_limits, high_limits)
        suc = self.set_target(self.target_q)
        
        # apply initial for simulator
        mujoco.mj_forward(self._model, self._data)

        if self._viewer is not None:
            self._viewer.load_new_model(self._model)

        # Reset observation
        observation = self._get_observation()

        # avg dist stuff
        self.sum = 0
        self.count = 0

        return observation, {}

    def step(self, action):
        # do simulation
        #  extract action limmits
        low_action = self._model.actuator_ctrlrange[:,0]
        high_action = self._model.actuator_ctrlrange[:,1]
        low_action[ low_action == 0.] = -np.inf
        high_action[ high_action == 0.] = np.inf

        action = self._bound(action, low_action, high_action) # bound action
        self._data.ctrl[:self.nu] = action.copy()                 # apply action
        mujoco.mj_step(self._model, self._data, 1)      # sim. step

        # x \in R^{8} = [q1, q2, x_target, y_target, dq1, dq2, dx_target, dy_target]
        self.x = np.concatenate([self._data.qpos.flat.copy(), self._data.qvel.flat.copy()])

        # compute reward
        reward = self._get_reward(action)

        absorbing = False

        # check if it's close to a joint limit
        for i in range(self.nq):
            low_limit, high_limit = self.get_jnt_limmits()
            if not (low_limit[i] <= self.x[i] <=  high_limit[i]):
                absorbing = True
                reward = -100
                break

        observation = self._get_observation()
        return observation, reward, absorbing, {}

    def _get_reward(self, action):
        target_position = self.get_state_target()
        tcp_position = self.fk(self.x[:self.nq])[0][:2]

        vec = tcp_position - target_position
        reward_dist = -np.linalg.norm(vec)      * 1.0
        reward_ctrl = -np.square(action).sum()  * 0.1 
        reward = reward_dist + reward_ctrl

        # target color
        scale = 1 / (0.1 * self.nq + 0.01)
        self._model.geom('target').rgba = np.array([ 
                abs(np.sin(reward_dist * scale)) * 2,
                abs(np.cos(reward_dist * scale)) * 2,
                0, 
                0.9 ], dtype=np.float32)

        # avg dist stuff
        self.sum += -reward_dist
        self.count += 1

        return reward

    def render(self, record=False):
        if self.debug_gui:
            if self._viewer is None:
                self._viewer = MujocoViewer(self._model, self.dt, record=record, **self._viewer_params)
            return self._viewer.render(self._data, record)
        return {}

    def stop(self):
        if self._viewer is not None:
            self._viewer.stop()
            del self._viewer
            self._viewer = None

    def fk(self, q, body_name='fingertip'):
        # print(q)
        self._data.qpos[:self.nq] = q
        mujoco.mj_fwdPosition(self._model, self._data)
        return self._data.body(body_name).xpos.copy(), self._data.body(body_name).xmat.reshape(3, 3).copy()

    @property
    def dt(self):
        return self._timestep * self._n_intermediate_steps * self._n_substeps

def test():
    import time
    path = '/home/kika/path/iros2024/generalized_atacom_envs/experiments/robot_reacher/assests'
    xml_file = f'{path}/onedof.xml'
    # xml_file = f'{path}/twodof.xml'
    # xml_file = f'{path}/threedof.xml'
    # xml_file = f'{path}/fourdof.xml'

    plant = TwoDofMujoco(xml_file)
    
    q = plant.get_state_q()
    dq = plant.get_state_dq()

    plant.reset()

    q_des = np.zeros(plant.nq)
    # plant.set_target(q_des.copy(), min_radius=0.03)

    t0 = time.time()
    while True:
        t = time.time() - t0

        q_des = np.array([
            np.pi * np.sin(t), 
            0.9 * np.pi * np.sin(t),
            0.9 * np.pi * np.sin(0.5 * t),
            0.9 * np.pi * np.sin(0.25 * t),
        ])
        # twodof.set_target(q_des, min_radius=0.03)

        q = plant.get_state_q()
        dq = plant.get_state_dq()
        u = 1. * (q_des - q) - 0.2 * dq
        
        plant.step(u[:plant.nu])
        plant.render()

        if t > 2.:
            t0 = time.time()
            plant.reset()


if __name__ == '__main__':
    test()
