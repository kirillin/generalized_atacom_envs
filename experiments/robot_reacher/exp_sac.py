import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.utils import TorchUtils

from reacher_mujoco import TwoDofMujoco

from tqdm import trange

import time
from torch.utils.tensorboard import SummaryWriter

# from nn import SimpleActorNetwork as ActorNetwork
# from nn import SimpleCriticNetwork as CriticNetwork

from atacom.nn import SACActorNetwork as ActorNetwork
from atacom.nn import SACCriticNetwork as CriticNetwork


def experiment(alg, n_epochs, n_steps, n_steps_test, save, load, params):
    xml_file = params['xml_file']
    al, cl, af, cf = map(int, params['network'])

    run_name = f"reacher__a{al}_{af}_c{cl}_{cf}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "reacher_test",
        ""
    )
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=f'./logs/reacher__a{al}_{af}_c{cl}_{cf}' if save else None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    mdp = TwoDofMujoco(xml_file)

    # Settings
    initial_replay_size = af
    max_replay_size = 100000
    batch_size = af
    actor_n_features = [af] * al
    critic_n_features = [cf] * cl
    warmup_transitions = 1000
    tau = 0.002
    lr_alpha = 3e-4

    if load:
        try:
            agent = SAC.load(f'logs/reacher__a{al}_{af}_c{cl}_{cf}/SAC/agent-best.msh')
        except:
            print(f"no logs for 'logs/reacher__a{al}_{af}_c{cl}_{cf}/SAC/agent-best.msh'")
            exit(0)
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
            avg_dist = core.env.get_avg_dist()

            logger.epoch_info(n+1, J=J, R=R, entropy=E, avg_dist=avg_dist)

            writer.add_scalar("charts/discounted_return", J, n)
            writer.add_scalar("charts/undiscounted_return", R, n)
            writer.add_scalar("charts/agent_policy_entropy", E, n)
            writer.add_scalar("charts/avg_dist", avg_dist, n)

            if save:
                logger.log_best_agent(agent, J)

        # logger.info('Press a button to visualize pendulum')
        # input()
        # core.evaluate(n_episodes=5, render=True)
    
    writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learn', action='store_true')
    parser.add_argument('-e', '--eval', action='store_true')
    parser.add_argument('-n', '--network', default='1,1,256,256')

    args = parser.parse_args()

    TorchUtils.set_default_device('cuda')

    robotfile = 'onedof.xml'

    # path = '/home/kika/path/iros2024/generalized_atacom_envs/experiments/robot_reacher/assests'
    path = '/home/human/artemov/generalized_atacom_envs/experiments/robot_reacher/assests'

    xml_file = f'{path}/{robotfile}'


    if args.learn:
        # actor critic
        networks = [
            [1,1,256,256],
            [2,2,256,256],
            [3,3,256,256],
            [3,6,256,256],
            [1,1,512,512],
            [2,2,512,512],
            [3,3,512,512],
            [3,6,512,512],
        ]

        save = True
        load = False



        for network in networks:
            experiment(alg=SAC, n_epochs=200, n_steps=1000, n_steps_test=1000, 
                    save=save, load=load,
                    params=dict(xml_file=xml_file, network=network))

    if args.eval:
        save = False
        load = True
        network = list(map(int,args.network.split()))

        experiment(alg=SAC, n_epochs=200, n_steps=1000, n_steps_test=1000, 
                save=save, load=load,
                params=dict(xml_file=xml_file, network=network))        