import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from mushroom_rl.core import Core, Agent
from mushroom_rl.policy import Policy
from mushroom_rl.algorithms.actor_critic import SAC, PPO, TD3, DDPG, TRPO
from mushroom_rl.utils.preprocessors import MinMaxPreprocessor
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from safe_rl.utils.network import *
from robot_atacom_env import RobotAtacomEnv


class SACMeanPolicy(Policy):
    def __init__(self, mu_network):
        self.mu_network = mu_network

    def draw_action(self, state):
        return torch.tanh(self.mu_network.predict(state, output_tensor=True)).detach().cpu().numpy()


class SACMeanAgent(Agent):
    def __init__(self, mdp_info, mu_network, features=None):
        policy = SACMeanPolicy(mu_network)
        super().__init__(mdp_info, policy, features)


if __name__ == '__main__':
    agent_dir = "./logs/0"
    deterministic_policy = True

    # get Env Args
    for subdir, dirs, files in os.walk(os.path.abspath(agent_dir)):
        for filename in files:
            if filename == "args.json":
                args = json.load(open(os.path.join(subdir, filename)))

    device = 'cuda' if args['use_cuda'] else 'cpu'

    mdp = RobotAtacomEnv(gamma=args['gamma'], horizon=args['horizon'], control=args['control'],
        timestep=args['timestep'], n_intermediate_steps=args['n_intermediate_steps'],
        debug_gui=True,
        update_with_agent_freq=args['atacom_update_with_agent_freq'], 
        atacom_slack_type=args['atacom_slack_type'],
        device=device
    )
    eval_params = dict(n_episodes=10,
                       render=False,
                       quiet=False)
    for subdir, dirs, files in os.walk(os.path.abspath(agent_dir)):
        for filename in files:
            if 'agent-' in filename and 'best' not in filename and filename.endswith('.msh'):
                mdp.reset_log_info()
                if args['alg'] == 'sac':
                    sac_agent = eval(args['alg'].upper()).load(os.path.join(subdir, filename))

                # Adapt to the changes of mushroom-rl 1.9.1
                if os.path.isfile(os.path.join(subdir, 'state_normalization' + filename[len('agent'):])):
                    prepro = eval(args['preprocessor']).load(os.path.join(subdir, 'state_normalization' + filename[len('agent'):]))
                    sac_agent.add_preprocessor(prepro)

                core = Core(sac_agent, mdp)
                dataset = core.evaluate(**eval_params)
                num_collision, num_joint_constraint, episode_steps = mdp.get_log_info()
                print("Num Collision:", num_collision)
                print("Num Joint Constraint:", num_joint_constraint)
                print("Episode Steps:", episode_steps)

                parsed_dataset = parse_dataset(dataset)
                J = np.mean(compute_J(dataset, core.mdp.info.gamma))
                R = np.mean(compute_J(dataset))
                print("J:", J, " R:", R)

                action_track = list()
                state_track = list()
                for sample in dataset:
                    state_track.append(sample[0])
                    action_track.append(sample[1])
                action_track = np.array(action_track)
                state_track = np.array(state_track)

                fig, axes = plt.subplots(4, 2)
                for k in range(action_track.shape[1]):
                    axes[int(k / 2), k % 2].plot(action_track[:, k])
                plt.show()
