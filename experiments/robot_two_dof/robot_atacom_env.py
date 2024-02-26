import numpy as np
from safe_rl.algs.atacom import AtacomEnvWrapper, StateConstraint, ConstraintsSet
from robot_mushroom_env import RobotEnv, env_dir


class RobotAtacomEnv(AtacomEnvWrapper):

    def __init__(self, gamma=0.99, horizon=500, 
                    step_action_function=None, timestep=1 / 30., n_intermediate_steps=1,
                    debug_gui=False, 
                    init_state=None, terminate_on_collision=True, 
                    save_key_frame=False,
                    slack_type='softcorner', update_with_agent_freq=True, is_opponent_moving=True,
                    slack_beta_wall_constraint=2., slack_beta_fetch_constraint=2.,
                    slack_threshold_wall_constraint=0.01, slack_threshold_fetch_constraint=0.01,
                    robot_file=env_dir + "/models/two_dof.urdf"):

        base_env = RobotEnv(gamma=gamma, horizon=horizon, 
                        step_action_function=step_action_function,
                        timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                        debug_gui=debug_gui, 
                        init_state=init_state,
                        robot_file=robot_file)

        dim_q = base_env.info.action_space.shape[0]
        dim_x = 0

        constraints = ConstraintsSet(dim_q, dim_x=dim_x)
        
        c_wall = StateConstraint(dim_q=dim_q, dim_out=4, fun=self.wall_f, jac_q=self.wall_J,
                                dim_x=dim_x, jac_x=self.wall_J_x,
                                slack_type=slack_type, slack_beta=slack_beta_wall_constraint,
                                threshold=slack_threshold_wall_constraint)

        constraints.add_constraint(c_wall)

        atacom_step_size = timestep
        if update_with_agent_freq:
            atacom_step_size = timestep * n_intermediate_steps

        super().__init__(base_env, 
                            dq_max=base_env.info.action_space.high, 
                            constraints=constraints,
                            step_size=atacom_step_size, 
                            update_with_agent_freq=update_with_agent_freq)
    
    def _get_q(self, state):
        q = state[:2]
        return np.array(q)

    def _get_x(self, state):
        return None

    def _get_dx(self, state):
        return np.atleast_1d(0.)

    def compute_ctrl_action(self, action):
        return action

    def reset(self, state=None):
        state = super().reset(state)
        self.alpha_prev = np.zeros_like(self.env.info.action_space.low)
        success = np.all(self.constraints.c(self.q, self.x, origin_constr=True) < 0)
        while not success:
            state = super().reset()
            success = np.all(self.constraints.c(self.q, self.x, origin_constr=True) < 0)
        return state

    def reset_log_info(self):
        self.env.reset_log_info()

    def get_log_info(self):
        return self.env.get_log_info()

    def wall_f(self, q, x=None):
        return np.array([0])

    def wall_J(self, q, x=None):
        J_cart = np.array([[1., 0., 0.], [0., 1., 0.], [-1, 0, 0.], [0, -1, 0.]])
        return J_cart

    def wall_J_x(self, q, x=None):
        return np.zeros((4, self.constraints.dim_x))

    def reward(self, state, action, next_state, absorbing):
        return 0

def test_atacom_env():
    mdp = RobotAtacomEnv(debug_gui=True, slack_type='softcorner')
    while True:
        mdp.reset()
        for i in range(mdp.info.horizon):
            # action = np.random.uniform(mdp.info.action_space.low, mdp.info.action_space.high)
            action = np.zeros_like(mdp.info.action_space.low)
            # action[0] = 1.
            state, reward, absorbing, _ = mdp.step(action)
            if absorbing:
                print("############ Collisions! ####################")
                break
        print(mdp.get_constraints_logs())

if __name__ == '__main__':
    test_atacom_env()
    