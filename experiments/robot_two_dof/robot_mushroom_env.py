import numpy as np

import pybullet
import pybullet_data

from mushroom_rl.utils.spaces import Box
from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType

import os
env_dir = os.path.dirname(__file__)

from robot_kinematics import Kinematics


class RobotEnv(PyBullet):
    """ Two dof robot (2R manipulator)"""
    
    def __init__(self, 
        robot_file=env_dir + "/models/two_dof.urdf",
        init_state=None,
        self_collision=False,
        collide_termination=False, control='velocity_position',
        step_action_function = None,
        gamma=0.99, horizon=500, timestep=1/240., n_intermediate_steps=4,
        debug_gui=False
    ):
        self.debug_gui = debug_gui
        self.init_state = init_state
        
        self.control_flags = {'velocity_position': False}
        self.self_collision = self_collision

        if control == 'torque':
            self.control_flags['mode'] = pybullet.TORQUE_CONTROL
        elif control == 'position':
            self.control_flags['mode'] = pybullet.POSITION_CONTROL
        elif control == 'velocity':
            self.control_flags['mode'] = pybullet.VELOCITY_CONTROL
        elif control == 'velocity_position':
            self.control_flags['mode'] = pybullet.POSITION_CONTROL
            self.control_flags['velocity_position'] = True
        else:
            raise NotImplementedError

        self.collide_termination = collide_termination

        self.kinematics = Kinematics(robot_file)
        self.kinematics_pos = np.zeros(self.kinematics.model.nq)

        model_files = dict()
        model_files[robot_file] = dict(basePosition=[0., 0., 1.],
                                       baseOrientation=[0., 0., 0., 1.])
        self.pybullet_data_path = pybullet_data.getDataPath()
        plane_file = os.path.join(self.pybullet_data_path, "plane.urdf")
        model_files[plane_file] = dict(useFixedBase=True, basePosition=[0.0, 0.0, 0.0],
                                       baseOrientation=[0, 0, 0, 1])
        # add additional models
        model_files.update(self.add_models())   # add walls

        if self_collision:
            model_files[robot_file].update(dict(flags=pybullet.URDF_USE_SELF_COLLISION))

        # get the robot actuation and observation info
        actuation_spec, observation_spec = self.construct_act_obs_spec()

        # initilize mushroom rl pybullet env
        super().__init__(model_files, 
            actuation_spec, 
            observation_spec, 
            gamma,
            horizon, 
            timestep, n_intermediate_steps,
            debug_gui=debug_gui, size=(500, 500), distance=1.8
        )

        self.robot = self._indexer.model_map['two_dof']       

        self.init_post_process()

        self.target_pos = np.zeros(3)
        self.count_collide = 0
        self.count_joint_constraint = 0
        self.step_counter = 0
        self.episode_steps = list()
        self.key_frame_list = list()
        self.final_distance_list = list()
        self.step_action_function = step_action_function

    def _custom_load_models(self):
        vis_shape = self.client.createVisualShape(self.client.GEOM_SPHERE, radius=0.1, rgbaColor=[1., 0., 0., 0.5])
        self.target_pb_id = self.client.createMultiBody(baseVisualShapeIndex=vis_shape,
                                                        basePosition=[0,0,0])
        return {"target": self.target_pb_id}

    def add_models(self):
        model_files = dict()

        # add model of walls
        # wall_file = env_dir + '/models/wall.urdf'
        # model_files[wall_file] = dict(useFixedBase=True, basePosition=[-1.5, -1.5, 0.],baseOrientation=[0., 0., 0., 1.])

        return model_files

    def init_post_process(self):
        self.joint_names = list()
        self.kinematics_update_idx = list()

        for j, joint_name in enumerate(self.kinematics.model.names[1:]):
            self.joint_names.append(joint_name)
            if joint_name in self._indexer.observation_indices_map.keys():
                self.kinematics_update_idx.append(
                    (j, self._indexer.observation_indices_map[joint_name][PyBulletObservationType.JOINT_POS][0])
                )

        if self.self_collision:
            self.collision_mask()

    # Remove z,qx,qy,qz parts of target pose vector
    def _modify_mdp_info(self, mdp_info):
        super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(mdp_info.observation_space.low[:-5], mdp_info.observation_space.high[:-5])
        if self.control_flags['velocity_position']:
            mdp_info.action_space = Box(-2*np.ones(mdp_info.action_space.shape), 2*np.ones(mdp_info.action_space.shape))
        return mdp_info
    
    # Remove z,qx,qy,qz parts of target pose vector
    def _create_observation(self, state):
        return state[:-5]
    

    def collision_mask(self):

        for idx in range(self.client.getNumJoints(self.robot)):
            self.client.setCollisionFilterGroupMask(self.robot, idx, collisionFilterGroup=int('00000001', 2),
                                                    collisionFilterMask=int('11111000', 2))
        for name in ['link_1', 'link_2']:
            self.client.setCollisionFilterGroupMask(*self._indexer.link_map[name],
                                                    collisionFilterGroup=int('00000001', 2),
                                                    collisionFilterMask=int('11111110', 2))
            
    def construct_act_obs_spec(self):
        actuation_spec = list()
        observation_pos_spec = list()
        observation_vel_spec = list()

        actuation_spec.append(("joint_1", self.control_flags['mode']))
        actuation_spec.append(("joint_2", self.control_flags['mode']))
        observation_pos_spec.append(("joint_1", PyBulletObservationType.JOINT_POS))
        observation_pos_spec.append(("joint_2", PyBulletObservationType.JOINT_POS))

        observation_vel_spec.append(("target", PyBulletObservationType.BODY_POS))

        return actuation_spec, observation_pos_spec + observation_vel_spec

    def reset(self, state=None):
        observation = super().reset(state)
        return observation

    def setup(self, state=None):
        """ executes the setup code after an environment reset """
        x, y, z = np.random.uniform([-2,-2, 1], [2,2,1])
        radius = 2.0 # hardcoded, the robot length

        distance_to_point = np.sqrt(x**2 + y**2)
        
        if distance_to_point > radius:
            k = radius / distance_to_point
            projected_x = x * k
            projected_y = y * k
            x, y = projected_x, projected_y
        
        self.target_pos = [x, y, z]


        if self.debug_gui:
            self._client.resetBasePositionAndOrientation(self.target_pb_id, self.target_pos, [0., 0., 0., 1.])

        self.kinematics_pos = np.zeros(self.kinematics.model.nq)

        if state is not None:
            for j, joint_name in enumerate(self.joint_names):
                self.kinematics_pos[j] = state[j]
        elif self.init_state is not None:
            for j, joint_name in enumerate(self.joint_names):
                self.kinematics_pos[j] = self.init_state[j]

        for j, joint_name in enumerate(self.joint_names):
            self.client.resetJointState(*self._indexer.joint_map[joint_name], self.kinematics_pos[j])
            self.client.setJointMotorControl2(*self._indexer.joint_map[joint_name],
                                              controlMode=self.client.POSITION_CONTROL,
                                              targetPosition=self.kinematics_pos[j])
        
        if self.debug_gui:
            self.client.resetDebugVisualizerCamera(3.1, 90, -91, (0,0,0))
        super(RobotEnv, self).setup(state)

    def reward(self, state, action, next_state, absorbing):
        """
            Compute the reward based on the given transition.
            Args:
                state (np.array): the current state of the system;
                action (np.array): the action that is applied in the current state;
                next_state (np.array): the state reached after applying the given action;
                absorbing (bool): whether next_state is an absorbing state or not.
        """        

        # visualize closure
        if self.debug_gui:
            self._client.changeVisualShape(self.target_pb_id, -1, rgbaColor=[np.sin(abs(self.goal_distance)/2), np.cos(abs(self.goal_distance)/2), 0, 0.5])

        reward = -self.goal_distance*5 + 2
        if self.goal_distance < 0.05:
            reward += 5

        reward -= 0.01 * np.linalg.norm(action)

        if self._in_collision():
            if self.collide_termination:
                reward = -30
            else:
                reward -= 1

        return reward

    def is_absorbing(self, state):
        """ Check whether the given state is an absorbing state or not """
        for index in self.kinematics_update_idx:
            self.kinematics_pos[index[0]] = state[index[1]]
        self.kinematics.forward(self.kinematics_pos)
        tcp_frame_id = self.kinematics.model.getFrameId("joint_tcp")
        self.tcp_pose = self.kinematics.get_frame(tcp_frame_id)
        self.goal_distance = np.linalg.norm(self.tcp_pose.translation[:2] - self.target_pos[:2])

        # if self._in_collision():
        #     return True
        self.step_counter += 1

        threshold = (self.joints.limits()[1] - self.joints.limits()[0]) * 0.01
        if (state[0:2] - self.joints.limits()[0] < threshold).any() or \
                (self.joints.limits()[1] - state[0:2] < threshold).any():
            self.count_joint_constraint += 1
            if self.collide_termination:
                self.episode_steps.append(self.step_counter)
                self.final_distance_list.append(self.goal_distance)
                return True

        if self.step_counter >= self.info.horizon:
            self.episode_steps.append(self.step_counter)
            self.final_distance_list.append(self.goal_distance)
        return False

    def get_joint_states(self):
        result = list()
        for joint in self.joint_names:
            result.append(self.client.getJointState(self.robot, self._indexer.joint_map[joint][1])[0])
        return result

    def _compute_action(self, state, action):
        if self.step_action_function is None:
            ctrl_action = action.copy()
        else:
            ctrl_action = self.step_action_function(state, action)

        if self.control_flags['velocity_position']:
            ctrl_action = self.joints.positions(self._state) + \
                          action * self._timestep * self._n_intermediate_steps

            ctrl_action = np.clip(ctrl_action, self.joints.limits()[0] + 0.05, self.joints.limits()[1] - 0.05)
        return ctrl_action

    def _in_collision(self):
        # robot_id = self._indexer.model_map['two_dof']
        # wall_id = self._indexer.model_map['wall']
        # if len(self.client.getContactPoints(robot_id, wall_id)):
        #     return True
        # else:
        #     return False
        return False

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
        self.final_distance_list = list()

    def get_log_info(self):
        return self.count_collide, self.count_joint_constraint, np.mean(self.episode_steps)


def test_env():
    import time
    env = RobotEnv(self_collision=True, debug_gui=True)
    
    np.random.seed(1)
    
    for i in range(100):
        env.reset()
        J = 0
        R = 0
        while True:
            action = np.random.uniform(env.info.action_space.low, env.info.action_space.high)
            # action = np.zeros(3)
            observation, reward, absorbing, _ = env.step(action)
            J += reward * (env.info.gamma ** i)
            R += reward
            if absorbing:
                print("collision, absorb!")
                break
            time.sleep(1 / 240. * env._n_intermediate_steps)
        print("Total Return: ", R, "Discounted Return:", J)


if __name__ == '__main__':
    test_env()
    