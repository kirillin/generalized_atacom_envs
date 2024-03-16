import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.utils import TorchUtils

# from onedof import OneDof
from onedof_with_limits import OneDof

from tqdm import trange

import time
from torch.utils.tensorboard import SummaryWriter

from nn import SACActorNetwork as ActorNetwork
from nn import SACCriticNetwork as CriticNetwork


def experiment(alg, n_epochs, n_steps, n_steps_test, save, load):
    run_name = f"onedof__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "onedof_test",
        ""
    )

    np.random.seed()
    

    logger = Logger(alg.__name__, results_dir='./logs' if save else None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    mdp = OneDof()

    # Settings
    initial_replay_size = 64
    max_replay_size = 50000
    batch_size = 64
    actor_n_features = [64]
    critic_n_features = [64]
    warmup_transitions = 100
    tau = 0.005
    lr_alpha = 3e-4

    if load:
        agent = SAC.load('logs/SAC/agent-best.msh')
    else:
        # Approximator
        actor_input_shape = mdp.info.observation_space.shape
        actor_mu_params = dict(network=ActorNetwork,
                               n_features=actor_n_features,
                               input_shape=actor_input_shape,
                               output_shape=mdp.info.action_space.shape)
        actor_sigma_params = dict(network=ActorNetwork,
                                  n_features=actor_n_features,
                                  input_shape=actor_input_shape,
                                  output_shape=mdp.info.action_space.shape)

        actor_optimizer = {'class': optim.Adam,
                           'params': {'lr': 3e-4}}

        critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
        critic_params = dict(network=CriticNetwork,
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': 1e-4}},
                             loss=F.mse_loss,
                             n_features=critic_n_features,
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
            core.learn(n_steps=n_steps, n_steps_per_fit=1)
            dataset = core.evaluate(n_steps=n_steps_test, render=True)

            J = np.mean(dataset.discounted_return)
            R = np.mean(dataset.undiscounted_return)
            E = agent.policy.entropy(dataset.state)

            logger.epoch_info(n+1, J=J, R=R, entropy=E)
            print('epoch!')
            writer.add_scalar("charts/discounted_return", J, n)
            writer.add_scalar("charts/undiscounted_return", R, n)
            writer.add_scalar("charts/agent_policy_entropy", E, n)
            
            if save:
                logger.log_best_agent(agent, J)

        logger.info('Press a button to visualize pendulum')
        input()
        core.evaluate(n_episodes=5, render=True)
    
    writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learn', action='store_true')
    parser.add_argument('-e', '--eval', action='store_true')
    args = parser.parse_args()

    TorchUtils.set_default_device('cuda')

    if args.learn:
        save = True
        load = False
        experiment(alg=SAC, n_epochs=100, n_steps=1000, n_steps_test=500, save=save, load=load)

    if args.eval:
        save = False
        load = True
        experiment(alg=SAC, n_epochs=100, n_steps=1000, n_steps_test=500, save=save, load=load)
