import numpy as np

import pybullet
import pybullet_data

from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType
from mushroom_rl.utils.angles import normalize_angle
from mushroom_rl.rl_utils.spaces import Box

import os
env_dir = os.path.dirname(__file__)

from robot_kinematics import Kinematics


class OneDof(PyBullet):

    def __init__(self, 
                 robot_file=env_dir + "/models/one_dof.urdf",
                 init_state=None,
                 gamma=0.99, horizon=500, timestep=1/240., n_intermediate_steps=4,
                 debug_gui=True):

        np.random.seed()
        self.debug_gui = debug_gui
        self.init_state = init_state

        # add robot kinematics
        self.kinematics = Kinematics(robot_file)
        self.kinematics_pos = np.zeros(self.kinematics.model.nq)

        # create pybullet env
        model_files = dict()
        model_files[robot_file] = dict(basePosition=[0., 0., 1.5],
                                       baseOrientation=[0., 0., 0., 1.])
        self.pybullet_data_path = pybullet_data.getDataPath()
        plane_file = os.path.join(self.pybullet_data_path, "plane.urdf")
        model_files[plane_file] = dict(useFixedBase=True, basePosition=[0.0, 0.0, 0.0],
                                       baseOrientation=[0, 0, 0, 1])

        # add control and observation
        self.control_flags = {'mode': pybullet.VELOCITY_CONTROL}
        actuation_spec, observation_spec = self.construct_act_obs_spec()

        # initilize mushroom rl pybullet env
        super().__init__(model_files, 
                        actuation_spec, 
                        observation_spec, 
                        gamma,
                        horizon, 
                        timestep, n_intermediate_steps,
                        debug_gui=debug_gui, size=(500, 500), distance=1.8)

        self.one_dof = self._indexer.model_map['one_dof']       

        self.init_post_process()

        self.target_pos = np.zeros(3)
        self.count_collide = 0
        self.count_joint_constraint = 0
        self.step_counter = 0
        self.episode_steps = list()
        self.key_frame_list = list()
        self.step_action_function = None
        self.target_theta = 0
        self.steps_to_target = 0

    def set_target(self, target):
        self.target_theta = target

    def _modify_mdp_info(self, mdp_info):
        observation_high = np.array([3.14, 1.0, 1, 1, 1, 1, 1, 1, 1])
        observation_low = -observation_high
        mdp_info.observation_space = Box(observation_low, observation_high)
        return mdp_info

    def _custom_load_models(self):
        """add target pose marker"""
        vis_shape = self.client.createVisualShape(self.client.GEOM_SPHERE, radius=0.1, rgbaColor=[1., 0., 0., 0.5])
        self.target_pb_id = self.client.createMultiBody(baseVisualShapeIndex=vis_shape,
                                                        basePosition=[0,0,0])
        return {"target": self.target_pb_id}

    def init_post_process(self):
        """joint info """
        self.joint_names = list()
        self.kinematics_update_idx = list()

        for j, joint_name in enumerate(self.kinematics.model.names[1:]):
            self.joint_names.append(joint_name)
            if joint_name in self._indexer.observation_indices_map.keys():
                self.kinematics_update_idx.append(
                    (j, self._indexer.observation_indices_map[joint_name][PyBulletObservationType.JOINT_POS][0])
                )

    def construct_act_obs_spec(self):
        """
            action
                dq \in R^1
            observation
                [q, dq] \in R^2
        """
        actuation_spec = list()
        observation_pos_spec = list()
        observation_vel_spec = list()

        actuation_spec.append(("joint_1", self.control_flags['mode']))
        observation_pos_spec.append(("joint_1", PyBulletObservationType.JOINT_POS))
        observation_vel_spec.append(("joint_1", PyBulletObservationType.JOINT_VEL))
        observation_vel_spec.append(("target", PyBulletObservationType.BODY_POS))

        return actuation_spec, observation_pos_spec + observation_vel_spec

    def reset(self, state=None):
        observation = super().reset(state)
        self.steps_to_target = 0
        self.steps = 0
        return observation

    def setup(self, state=None):
        """ executes the setup code after an environment reset """

        # # generate new target point
        # # y-z plane robot
        # # for theta = 0 robot concine with z-axis
        theta = np.random.uniform(-3.14, 3.14)
        self.target_theta = theta
        # theta = self.target_theta
        self.target_pos = [0, np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)+1.5]

        if self.debug_gui:
            self._client.resetBasePositionAndOrientation(self.target_pb_id, self.target_pos, [0., 0., 0., 1.])

        self.kinematics_pos = np.zeros(self.kinematics.model.nq)

        if state is not None:
            for j, joint_name in enumerate(self.joint_names):
                self.kinematics_pos[j] = state[j]
        elif self.init_state is not None:
            for j, joint_name in enumerate(self.joint_names):
                self.kinematics_pos[j] = self.init_state[j]

        # set initial state in pybullet position_control mode
        for j, joint_name in enumerate(self.joint_names):
            self.client.resetJointState(*self._indexer.joint_map[joint_name], self.kinematics_pos[j])
            self.client.setJointMotorControl2(*self._indexer.joint_map[joint_name],
                                              controlMode=self.client.POSITION_CONTROL,
                                              targetPosition=self.kinematics_pos[j])

        if self.debug_gui:
            # self.client.resetDebugVisualizerCamera(3.1, 90, -91, (0,0,0))
            self.client.resetDebugVisualizerCamera(2, 90, 0, (0,0,1))

        super(OneDof, self).setup(state)

    def reward(self, state, action, next_state, absorbing):
        self.steps += 1

        q, dq = state[:2]
        goal = np.abs(self.target_theta - q)

        reward = - goal**2 - 0.1 * dq**2 - 0.001 * action[0]**2

        if goal < 0.05:
            self.steps_to_target += 1
            reward += 10.0

        # visualize closure
        if self.debug_gui:
            self._client.changeVisualShape(self.target_pb_id, -1, rgbaColor=[np.sin(abs(goal)/2), np.cos(abs(goal)/2), 0, 0.5])

        return reward

    def is_absorbing(self, state):
        """ Check whether the given state is an absorbing state or not """
        if self.steps_to_target >= 1:
            return True
        if self.steps > 500:
            return True

        # if abs(state[0]) >= 3.14:
        #     return True

        return False

    def get_joint_states(self):
        result = list()
        for joint in self.joint_names:
            result.append(self.client.getJointState(self.robot, self._indexer.joint_map[joint][1])[0])
        return result

    def capture_key_frame(self):
        view_mat = self.client.computeViewMatrixFromYawPitchRoll([0.3, -0.3, 0.6], 1.5, 150, -40, 0., 2)
        proj_mat = self.client.computeProjectionMatrixFOV(90, 1, 0.1, 20)
        img = self.client.getCameraImage(800, 800, view_mat, proj_mat)
        self.key_frame_list.append(img[2])

    def reset_log_info(self):
        self.count_collide = 0
        self.count_joint_constraint = 0
        self.episode_steps = list()
        self.key_frame_list = list()

    def get_log_info(self):
        return self.count_collide, self.count_joint_constraint, np.mean(self.episode_steps)


def test_env():
    import time
    mdp = OneDof(debug_gui=True)
    
    mdp.reset()
    
    u = 0
    q_des = 1.5
    t = time.time()
    while True:
        res = mdp.step([u])
        q, dq = res[0][:2]
        # print(f"{time.time() - t}\t{q}\t{dq}\t{u}\n")
        print(mdp._mdp_info.observation_space.low)
        u = 10 * (q_des - q) - 2  * dq

        if time.time() - t > 3.0:
            q_des = -q_des
            t = time.time()


if __name__ == '__main__':
    test_env()
