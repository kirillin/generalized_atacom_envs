import numpy as np
from datetime import datetime

import torch.optim as optim
import torch.nn.functional as F
import distutils.version

from torch.utils.tensorboard import SummaryWriter

from tqdm import trange

from safe_rl.utils.network import *
from robot_mushroom_env import RobotEnv

from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.utils.preprocessors import MinMaxPreprocessor
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.utils.frames import LazyFrames

from experiment_launcher import run_experiment
from experiment_launcher.decorators import single_experiment


@single_experiment
def experiment(
        env: str = 'robot_two_dof',
        alg: str = 'sac',
        preprocessor: str = 'MinMaxPreprocessor',
        n_epochs: int = 15,
        n_steps: int = 1000,
        n_steps_per_fit: int = 1,
        n_episodes_test: int = 10,
        quiet: bool = False,
        use_cuda: bool = False,
        n_features: str = "256-256-256",
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        batch_size: int = 64,
        initial_replay_size: int = 5000,
        max_replay_size: int = 200000,
        tau: float = 1e-3,
        warmup_transitions: int = 5000,
        lr_alpha: float = 1e-6,
        target_entropy: int = -10,
        gamma: float = 0.995,
        horizon: int = 500,
        control: str = 'velocity_position',
        timestep: float = 1 / 240.,
        n_intermediate_steps: int = 8,
        debug_gui: bool = False,
        random_init: bool = True,
        save_key_frame: bool = False,
        collide_termination: bool = True,
        atacom_slack_type: str = 'softcorner',
        atacom_update_with_agent_freq: bool = True,
        seed: int = 0,
        results_dir: str = './logs'
):
    device = 'cuda' if use_cuda else 'cpu'
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
    mdp = RobotEnv(gamma=gamma, horizon=horizon, timestep=timestep,
                    n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui)
    
    logger = Logger(results_dir=results_dir, seed=seed, use_timestamp=True)
    writer = SummaryWriter(log_dir=logger.path)

    # Create network
    n_features = list(map(int, n_features.split('-')))

    agent_params = dict()
    network_params = dict(actor_lr=actor_lr, critic_lr=critic_lr, n_features=n_features, batch_size=batch_size)
    sac_params = dict(initial_replay_size=initial_replay_size, max_replay_size=max_replay_size, tau=tau,
                      warmup_transitions=warmup_transitions, lr_alpha=lr_alpha, target_entropy=target_entropy)
    agent_params.update(network_params)
    agent_params.update(sac_params)
    agent_params.update({"env_name": env})

    agent, build_params = build_agent(alg, mdp_info=mdp.info, **agent_params)
    print("Agent:", agent.__class__)

    # State Normalization
    if preprocessor == "MinMaxPreprocessor":
        prepro = MinMaxPreprocessor(mdp_info=mdp.info)
        agent.add_preprocessor(prepro)

    core = Core(agent, mdp)

    # Set the parameters for evaluation
    eval_params = dict(n_episodes=n_episodes_test, render=False, quiet=quiet)
    log_params = dict(env_name=env, best_J=-np.inf, save_key_frame=save_key_frame, it=0)

    logger.log_agent(agent)

    for it in trange(n_epochs, leave=False, disable=quiet):
        # mdp.reset_log_info()
        log_params['it'] = it

        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=quiet)
        dataset = core.evaluate(**eval_params)

        log_data(core, dataset, logger, writer, log_params)


def build_agent(alg, mdp_info, **kwargs):
    alg = alg.upper()
    if alg == 'SAC':
        agent, build_params = build_agent_SAC(mdp_info, **kwargs)
    elif 'LA' in alg:
        agent, build_params = build_agent_LA(mdp_info, **kwargs)
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


def build_agent_LA(mdp_info, env_name, **kwargs):
    # policy = LinearAttractorPolicy(env_name)
    # agent = LinearAttractorAgent(mdp_info, policy)
    # return agent, {}
    return {}, {}


def log_data(core, dataset, logger, writer, log_params):
    mdp = core.mdp
    agent = core.agent
    parsed_dataset = parse_dataset(dataset)

    num_col, num_constrain, average_steps = core.mdp.get_log_info()
    J = np.mean(compute_J(dataset, core.mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    it = log_params['it']

    if log_params['env_name'].endswith("atacom"):
        env_base = mdp.env
    else:
        env_base = mdp

    log_dict = {}
    logger.log_numpy(J=J, R=R)
    writer.add_scalar("reward/discounted_reward", J, it)
    writer.add_scalar("reward/cumulative_reward", R, it)
    log_dict.update({"J": J, "R": R})
    if hasattr(agent.policy, "compute_action_and_log_prob"):
        _, log_prob_pi = core.agent.policy.compute_action_and_log_prob(parsed_dataset[0])
        E = -log_prob_pi.mean()
        logger.log_numpy(E=E)
        writer.add_scalar("parameters/entropy", E, it)
        log_dict.update({"E": E})
    elif hasattr(agent.policy, "entropy"):
        E = core.agent.policy.entropy(parsed_dataset[0])
        logger.log_numpy(E=E)
        writer.add_scalar("parameters/entropy", E, it)
        log_dict.update({"E": E})
    if hasattr(agent, "_alpha_np"):
        alpha_SAC = core.agent._alpha_np
        logger.log_numpy(alpha=alpha_SAC)
        writer.add_scalar("parameters/alpha", alpha_SAC, it)
        log_dict.update({"alpha": alpha_SAC})
    if hasattr(agent, "_critic_approximator"):
        V = compute_V(agent, get_init_states(dataset))
        writer.add_scalar("parameters/value_function", V, it)
        logger.log_numpy(V=V)
        log_dict.update({"V": V})
    writer.add_scalar("others/collisions", num_col, it)
    if hasattr(env_base, "final_distance_list"):
        final_dist = np.mean(env_base.final_distance_list)
        writer.add_scalar("others/final_distance", final_dist, it)
        logger.log_numpy(final_distance=final_dist)
        log_dict.update({"final_distance": final_dist})
    if hasattr(env_base, "ave_distance_list"):
        ave_dist = np.mean(env_base.ave_distance_list)
        writer.add_scalar("others/average_distance", ave_dist, it)
        logger.log_numpy(average_distance=ave_dist)
        log_dict.update({"average_distance": ave_dist})
    if hasattr(env_base, "min_distance_human"):
        min_distance_human = np.mean(env_base.min_distance_human)
        writer.add_scalar("others/average_min_distance_human", min_distance_human, it)
        logger.log_numpy(average_min_distance_human=min_distance_human)
        log_dict.update({"average_min_distance_human": min_distance_human})
    if hasattr(env_base, "count_sprink"):
        count_sprink = np.min(env_base.count_sprink)
        writer.add_scalar("others/count_sprink", count_sprink, it)
        logger.log_numpy(count_sprink=count_sprink)
        log_dict.update({"count_sprink": count_sprink})

    logger.log_agent(agent)

    if J > log_params['best_J']:
        best_J = J
        logger.log_best_agent(agent, J)

    writer.add_scalar("others/joint_constraint", num_constrain, it)
    writer.add_scalar("others/average_steps", average_steps, it)
    log_dict.update({"collisions": num_col, "joint_constraint": num_constrain, "average_steps": average_steps})
    logger.log_numpy(collisions=num_col, joint_constraints=num_constrain, average_steps=average_steps)
    if log_params['env_name'].endswith("atacom") or log_params['env_name'].endswith("se"):
        g_avg, g_max = mdp.get_constraints_logs()
        writer.add_scalar("others/max_constraint", g_max, it)
        writer.add_scalar("others/average_constraint", g_avg, it)
        logger.log_numpy(max_constraint=g_max, average_constraint=g_avg)
        log_dict.update({"max_c": g_max, "avg_c": g_avg})

        if log_params['save_key_frame']:
            for j, key_frame in enumerate(env_base.key_frame_list):
                writer.add_image("key_frame_" + str(j), key_frame, it, dataformats='HWC')
    logger.epoch_info(it, **log_dict)


def compute_V(agent, states):
    Q = list()
    for state in states:
        s = np.array([state for i in range(100)])
        a = np.array([agent.policy.draw_action(state) for i in range(100)])
        Q.append(agent._critic_approximator(s, a).mean())
    return np.array(Q).mean()


def get_init_states(dataset):
    pick = True
    x_0 = list()
    for d in dataset:
        if pick:
            if isinstance(d[0], LazyFrames):
                x_0.append(np.array(d[0]))
            else:
                x_0.append(d[0])
        pick = d[-1]
    return np.array(x_0)


if __name__ == '__main__':
    run_experiment(experiment)
