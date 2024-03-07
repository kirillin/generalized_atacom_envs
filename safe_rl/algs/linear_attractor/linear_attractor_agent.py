import numpy as np
from mushroom_rl.core import Agent
from mushroom_rl.policy import Policy
from mushroom_rl.utils.angles import shortest_angular_distance

from safe_rl.atacom_wrappers.atacom_tiago_table import TiagoTableAtacom, TiagoTableEnv, env_dir
from safe_rl.atacom_wrappers.atacom_tiago_shelf import TiagoShelfRealAtacom, TiagoShelfRealEnv
from safe_rl.atacom_wrappers.atacom_tiago_navigation import TiagoFetchNavEnv, TiagoNavAtacom
from safe_rl.environments.tiago_base_env import TiagoKinematics


class LinearAttractorPolicy(Policy):
    def __init__(self, env_name, kinematic_file=env_dir + "/models/tiago_urdf/tiago_no_wheel_with_screen.urdf"):
        self.env_name = env_name
        self.kinematics = TiagoKinematics(kinematic_file)
        if "table" in env_name or "shelf" in env_name:
            self.joint_idx = [0, 1, 2, 3, 4, 5, 6]
            self.target_idx = [7, 8, 9]
            self.right_hand_frame_id = self.kinematics.pino_model.getFrameId("gripper_right_grasping_frame")
            self.kinematics_update_idx = [10, 11, 12, 13, 14, 15, 16]
            self.kinematics_init_pos = np.array([0.1,
                                                 -1., 1.5, 2.8, 1.57, 1.57, -1.57, 0., 0., 0.,
                                                 -0.28307335, -0.5956663, -0.32651209, 0.1543221, -0.57897781,
                                                 -0.21732101, 0.12036629, 0., 0.,
                                                 -0.9, -0.5])
        if "nav" in env_name:
            self.target_idx = [0, 1]
            self.tiago_pos_idx = [2, 3, 4, 5]
            self.tiago_vel_idx = [6, 7, 8]
            self.fetch_pos_idx = [9, 10, 11, 12]
            self.fetch_vel_idx = [13, 14, 15]
            self.prev_action = [16, 17]

        self.K = np.diag(np.ones(3))

        self._add_save_attr()

    def __call__(self, *args, **kwargs):
        pass

    def draw_action(self, state):
        if "table" in self.env_name or "shelf" in self.env_name:
            return self.draw_action_manip(state)
        if "nav" in self.env_name:
            return self.draw_action_nav(state)

    def draw_action_manip(self, state):
        self.kinematics_init_pos[self.kinematics_update_idx] = state[self.joint_idx]
        self.kinematics.forward_kinematics(self.kinematics_init_pos)
        ee_pos = self.kinematics.get_frame(self.right_hand_frame_id).translation
        jacobian = self.kinematics.get_jacobian(self.right_hand_frame_id)[:3, self.kinematics_update_idx]

        joint_vel = np.linalg.pinv(jacobian) @ self.K @ (state[self.target_idx] - ee_pos)

        vel_limit_low = -np.maximum(self.kinematics.pino_model.velocityLimit[self.kinematics_update_idx],
                                5 * (self.kinematics.pino_model.lowerPositionLimit[self.kinematics_update_idx] - state[self.joint_idx]))
        vel_limit_high = np.minimum(self.kinematics.pino_model.velocityLimit[self.kinematics_update_idx],
                                5 * (self.kinematics.pino_model.upperPositionLimit[self.kinematics_update_idx] - state[self.joint_idx]))
        joint_vel = np.clip(joint_vel, vel_limit_low, vel_limit_high)
        return joint_vel

    def draw_action_nav(self, state):
        tiago_pos = state[self.tiago_pos_idx]
        tiago_lin_vel = state[self.tiago_vel_idx][:2]
        tiago_yaw = np.arctan2(tiago_pos[3], tiago_pos[2])
        target_pos = state[self.target_idx]

        pos_err = target_pos - tiago_pos[:2]
        target_angle = np.arctan2(*pos_err[::-1])
        ang_err = shortest_angular_distance(tiago_yaw, target_angle)
        if ang_err > np.pi:
            ang_err -= np.pi * 2
        if ang_err < -np.pi:
            ang_err += np.pi * 2

        v = np.clip(1 * np.linalg.norm(pos_err), -1, 1)
        omega = np.clip(1 * ang_err, -1.57, 1.57)

        return np.array([v, omega])

    def reset(self):
        pass


class LinearAttractorAgent(Agent):
    def __init__(self, mdp_info, policy, features=None):
        super().__init__(mdp_info, policy, None)

    def fit(self, dataset, **info):
        pass


if __name__ == '__main__':
    env_name = "nav_atacom"
    eval_params = {}

    if env_name == "table":
        env = TiagoTableEnv(debug_gui=True, random_init=True, control='velocity_position')
    elif env_name == "shelf":
        env = TiagoShelfRealEnv(debug_gui=True, random_init=True, control='velocity_position')
    elif env_name == "table_atacom":
        env = TiagoTableAtacom(debug_gui=True, random_init=True, control='velocity_position')
    elif env_name == "shelf_atacom":
        env = TiagoShelfRealAtacom(debug_gui=True, random_init=True, control='velocity_position')
    elif env_name == "nav":
        env = TiagoFetchNavEnv(debug_gui=True)
    elif env_name == "nav_atacom":
        env = TiagoNavAtacom(debug_gui=True)

    policy = LinearAttractorPolicy(env_name)
    agent = LinearAttractorAgent(env.info, policy)

    for j in range(10):
        observation = env.reset()
        J = 0
        R = 0
        for i in range(env.info.horizon):
            action = agent.draw_action(observation)
            observation, reward, absorbing, _ = env.step(action)
            J += reward * (env.info.gamma ** i)
            R += reward
            if absorbing:
                print("collision, absorb!")
                break
            # time.sleep(1 / 30)
        print("Total Return: ", R, "Discounted Return:", J)
