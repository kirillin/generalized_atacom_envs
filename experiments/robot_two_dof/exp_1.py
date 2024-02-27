import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.gymnasium_env import Gymnasium
from mushroom_rl.utils import TorchUtils
# from mushroom_rl.utils.preprocessors import MinMaxPreprocessor

from tqdm import trange

from safe_rl.utils.network import *
from robot_mushroom_env import RobotEnv, env_dir


def experiment(alg, n_epochs, n_steps, n_steps_test, save, load):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir='./logs' if save else None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    horizon = 500
    gamma = 0.995
    mdp = RobotEnv(gamma=gamma, horizon=horizon, 
                   timestep=1/240., n_intermediate_steps=8, 
                   debug_gui=True)

    # Settings
    initial_replay_size = 1000 #5000
    max_replay_size = 50000 #200000
    batch_size = 64
    n_features = [64]*3
    warmup_transitions = 1000 #5000
    tau = 1e-3
    lr_alpha = 1e-3 #1e-6
    actor_lr= 3e-4
    critic_lr = 3e-4
    target_entropy = -10

    # Approximator
    agent_params = dict()
    network_params = dict(actor_lr=actor_lr, critic_lr=critic_lr, n_features=n_features, batch_size=batch_size)
    sac_params = dict(initial_replay_size=initial_replay_size, max_replay_size=max_replay_size, tau=tau,
                      warmup_transitions=warmup_transitions, lr_alpha=lr_alpha, target_entropy=target_entropy)
    agent_params.update(network_params)
    agent_params.update(sac_params)
    agent_params.update({"env_name": "exp_1"})

    # Agent
    agent, build_params = build_agent("SAC", mdp_info=mdp.info, **agent_params)
    print("Agent:", agent.__class__)

    # # State Normalization
    # prepro = MinMaxPreprocessor(mdp_info=mdp.info)
    # agent.add_preprocessor(prepro)

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    E = agent.policy.entropy(dataset.state)

    logger.epoch_info(0, J=J, R=R, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy(dataset.state)

        logger.epoch_info(n+1, J=J, R=R, entropy=E)

        if save:
            logger.log_best_agent(agent, J)

    logger.info('Press a button to visualize pendulum')
    input()
    core.evaluate(n_episodes=5, render=True)



def build_agent(alg, mdp_info, **kwargs):
    alg = alg.upper()
    if alg == 'SAC':
        agent, build_params = build_agent_SAC(mdp_info, **kwargs)
    else:
        raise NotImplementedError
    return agent, build_params


def build_agent_SAC(mdp_info, actor_lr, critic_lr, n_features, batch_size,
                    initial_replay_size, max_replay_size, tau,
                    warmup_transitions, lr_alpha, target_entropy,
                    **kwargs):
    actor_mu_params = dict(network=SACActorNetwork,
                           input_shape=mdp_info.observation_space.shape,
                           output_shape=mdp_info.action_space.shape,
                           n_features=n_features,
                           use_cuda=torch.cuda.is_available())
    actor_sigma_params = dict(network=SACActorNetwork,
                              input_shape=mdp_info.observation_space.shape,
                              output_shape=mdp_info.action_space.shape,
                              n_features=n_features,
                              use_cuda=torch.cuda.is_available())

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': actor_lr}}
    critic_params = dict(network=SACCriticNetwork,
                         input_shape=(mdp_info.observation_space.shape[0] + mdp_info.action_space.shape[0],),
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': critic_lr}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         output_shape=(1,),
                         use_cuda=torch.cuda.is_available())

    alg_params = dict(initial_replay_size=initial_replay_size,
                      max_replay_size=max_replay_size,
                      batch_size=batch_size,
                      warmup_transitions=warmup_transitions,
                      tau=tau,
                      lr_alpha=lr_alpha,
                      critic_fit_params=None,
                      target_entropy=target_entropy)

    build_params = dict(compute_entropy_with_states=True,
                        compute_policy_entropy=True)

    return SAC(mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params,
               **alg_params), build_params


if __name__ == '__main__':
    save = False
    load = False
    TorchUtils.set_default_device('cpu')
    experiment(alg=SAC, n_epochs=40, n_steps=1000, n_steps_test=100, save=save, load=load)
