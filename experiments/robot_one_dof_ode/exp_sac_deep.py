import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.utils import TorchUtils

from onedof import OneDof
# from mushroom_rl.environments.inverted_pendulum import InvertedPendulum

# from onedof_full_observation import OneDof

from tqdm import trange

# from safe_rl.utils.network import *

import torch
import torch.nn as nn


class SACCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.topology = [n_input] + n_features + [n_output]
        layers = []
        for i in range(len(self.topology) - 2):
            layers.append(nn.Linear(self.topology[i], self.topology[i + 1]))
            nn.init.xavier_uniform_(layers[-1].weight, gain=nn.init.calculate_gain('relu'))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.topology[-2], self.topology[-1]))
        nn.init.xavier_uniform_(layers[-1].weight, gain=nn.init.calculate_gain('relu'))

        self._layers = layers
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        q = self.mlp(state_action)
        return torch.squeeze(q)


class SACActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(SACActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.topology = [n_input] + n_features + [n_output]
        layers = []
        for i in range(len(self.topology) - 2):
            layers.append(nn.Linear(self.topology[i], self.topology[i + 1]))
            nn.init.xavier_uniform_(layers[-1].weight, gain=nn.init.calculate_gain('relu'))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.topology[-2], self.topology[-1]))
        nn.init.xavier_uniform_(layers[-1].weight, gain=nn.init.calculate_gain('linear'))

        self._layers = layers
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, state):
        a = self.mlp(state.float())
        return a



def experiment(alg, n_epochs, n_steps, n_steps_test, save, load):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir='./logs' if save else None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    mdp = OneDof()
    # mdp = InvertedPendulum(max_u=100.)


    # Settings
    initial_replay_size = 256
    max_replay_size = 100000
    batch_size = 256
    n_features = [256,256,256]
    warmup_transitions = 1000
    tau = 0.001
    lr_alpha = 3e-4

    if load:
        agent = SAC.load('logs/SAC/agent-best.msh')
    else:
        # Approximator
        actor_input_shape = mdp.info.observation_space.shape
        actor_mu_params = dict(network=SACActorNetwork,
                               n_features=n_features,
                               input_shape=actor_input_shape,
                               output_shape=mdp.info.action_space.shape)
        actor_sigma_params = dict(network=SACActorNetwork,
                                  n_features=n_features,
                                  input_shape=actor_input_shape,
                                  output_shape=mdp.info.action_space.shape)

        actor_optimizer = {'class': optim.Adam,
                           'params': {'lr': 3e-4}}

        critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
        critic_params = dict(network=SACCriticNetwork,
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': 1e-4}},
                             loss=F.mse_loss,
                             n_features=n_features,
                             input_shape=critic_input_shape,
                             output_shape=(1,))

        # Agent
        agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                    actor_optimizer, critic_params, batch_size, initial_replay_size,
                    max_replay_size, warmup_transitions, tau, lr_alpha,
                    critic_fit_params=None)

    # Algorithm
    core = Core(agent, mdp)

    if load:
        logger.info('Press a button to visualize pendulum')
        input()
        core.evaluate(n_episodes=5, render=True)        
    else:
        # RUN
        dataset = core.evaluate(n_steps=n_steps_test, render=True)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy(dataset.state)

        logger.epoch_info(0, J=J, R=R, entropy=E)

        core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

        for n in trange(n_epochs, leave=False):
            core.learn(n_steps=n_steps, n_steps_per_fit=100)
            dataset = core.evaluate(n_steps=n_steps_test, render=False)
            # dataset = core.evaluate(n_episodes=5, render=False)

            J = np.mean(dataset.discounted_return)
            R = np.mean(dataset.undiscounted_return)
            E = agent.policy.entropy(dataset.state)

            logger.epoch_info(n+1, J=J, R=R, entropy=E)

            if save:
                logger.log_best_agent(agent, J)

        logger.info('Press a button to visualize pendulum')
        input()
        core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learn', action='store_true')
    parser.add_argument('-e', '--eval', action='store_true')
    args = parser.parse_args()

    TorchUtils.set_default_device('cpu')

    if args.learn:
        save = True
        load = False

    if args.eval:
        save = False
        load = True

    experiment(alg=SAC, n_epochs=50, n_steps=1000, n_steps_test=500, save=save, load=load)
