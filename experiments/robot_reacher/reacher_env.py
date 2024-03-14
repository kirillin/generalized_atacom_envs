import os

import numpy as np
import mujoco

from mushroom_rl.environments.mujoco import MuJoCo
from mushroom_rl.environments.mujoco import ObservationType

from mushroom_rl.environments.mujoco_envs import __file__ as path_robots


class ReacherEnv(MuJoCo):

    def __init__(self, gamma=0.99, horizon=500,
                 timestep=1 / 240., n_substeps=1, n_intermediate_steps=1, default_camera_mode="top_static",
                 **viewer_params):
        
        # initialize mujoco robot
        mj_model_path = 'model.xml'
        self.robot_model = mujoco.MjModel.from_xml_path(mj_model_path)
        self.robot_data = mujoco.MjData(self.robot_model)

        self.init_qpos = self.robot_data.qpos.ravel().copy()
        self.init_qvel = self.robot_data.qvel.ravel().copy()

        # actions and observations
        bounds = self.robot_model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)