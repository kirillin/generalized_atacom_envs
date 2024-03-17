import numpy as np

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.utils import TorchUtils

from reacher_mujoco import TwoDofMujoco

from tqdm import trange

import time
from torch.utils.tensorboard import SummaryWriter


from atacom.builders import mdp_builder, agent_builder

def experiment(alg, n_epochs, n_steps, n_steps_test, save, load, params):
    np.random.seed()

    training_config = {
        "id": 'reacher',
        "number_epochs": 100,
        "number_learn_iterations": 1000,
        "number_evaluation_iterations": 1000,
        "agent_config": {
            # SAC
            "initial_replay_size": 256,
            "max_replay_size": 100000,
            "warmup_transitions": 1000,
            "learning_rate_actor": 3e-4,
            "learning_rate_critic": 1e-4,
            "learning_rate_alpha": 0.005,
            "batch_size": 256,
            "tau": 0.0025,
            "actor_structure": [256] * 3,
            "critic_structure": [256] * 3,
        },
        "env_config": {
            # mujoco
            "xmlfile" : xml_file
        }
    }

    run_name = f"{training_config['id']}__a{'_'.join(list(map(str,training_config['agent_config']['actor_structure'])))}__c{'_'.join(list(map(str,training_config['agent_config']['critic_structure'])))}"

    # Tensorboard logger
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("reacher_test", "")

    # Terminal logger
    logger = Logger(alg.__name__, results_dir=f"./logs/{run_name}" if save else None)
    logger.strong_line()
    logger.info("Experiment Algorithm: " + alg.__name__)
    logger.info("Press a button to visualize pendulum")

    # Setup env and agent
    env = mdp_builder(TwoDofMujoco, training_config["env_config"])
    agent = agent_builder(SAC, env.info, training_config["agent_config"])

    # Algorithm
    core = Core(agent, env)

    # Run learn or evaluate
    number_epochs = training_config['number_epochs']
    number_learn_iterations = training_config['number_learn_iterations']
    number_evaluation_iterations = training_config['number_evaluation_iterations']
    initial_replay_size = training_config['agent_config']['initial_replay_size']

    if load:
        try:
            agent = SAC.load(f'logs/{run_name}/SAC/agent-best.msh')
        except:
            print(f"no logs for 'logs/{run_name}/SAC/agent-best.msh'")
            exit(0)
    else:
        dataset = core.evaluate(n_steps=number_evaluation_iterations)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy(dataset.state)

        logger.epoch_info(0, J=J, R=R, entropy=E)

        core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

        for n in trange(number_epochs, leave=False):
            core.learn(n_steps=number_learn_iterations, n_steps_per_fit=1)
            dataset = core.evaluate(n_steps=number_evaluation_iterations, render=False)

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

    writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learn', action='store_true')
    parser.add_argument('-e', '--eval', action='store_true')
    parser.add_argument('-n', '--network', default='1,1,256,256')

    args = parser.parse_args()

    TorchUtils.set_default_device('cpu')

    robotfile = 'onedof.xml'

    path = '/home/kika/path/iros2024/generalized_atacom_envs/experiments/robot_reacher/assests'
    # path = '/home/human/artemov/generalized_atacom_envs/experiments/robot_reacher/assests'

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
