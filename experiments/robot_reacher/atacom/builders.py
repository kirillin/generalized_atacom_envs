import torch.optim as optim
import torch.nn.functional as F

from atacom.nn import SACActorNetwork as ActorNetwork
from atacom.nn import SACCriticNetwork as CriticNetwork

def mdp_builder(envobj, config):
    xmlfile = config['xmlfile']
    return envobj(xmlfile)

def agent_builder(alg, envinfo, config):
    actor_n_features = config['actor_structure']
    critic_n_features = config['critic_structure']
    lr_actor = config['learning_rate_actor']
    lr_critic = config['learning_rate_critic']
    batch_size = config['batch_size']
    tau = config['tau']
    lr_alpha = config['learning_rate_alpha']

    initial_replay_size = config['initial_replay_size']
    max_replay_size = config['max_replay_size']
    warmup_transitions = config['warmup_transitions']

    # Approximator
    actor_input_shape = envinfo.observation_space.shape
    actor_mu_params = dict(network=ActorNetwork,
                            n_features=actor_n_features,
                            input_shape=actor_input_shape,
                            output_shape=envinfo.action_space.shape)
    actor_sigma_params = dict(network=ActorNetwork,
                                n_features=actor_n_features,
                                input_shape=actor_input_shape,
                                output_shape=envinfo.action_space.shape)

    actor_optimizer = {'class': optim.Adam,
                        'params': {'lr': lr_actor}}

    critic_input_shape = (actor_input_shape[0] + envinfo.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                            optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic}},
                            loss=F.mse_loss,
                            n_features=critic_n_features,
                            input_shape=critic_input_shape,
                            output_shape=(1,))

    # Agent
    agent = alg(envinfo, 
                actor_mu_params, actor_sigma_params, actor_optimizer, 
                critic_params, 
                batch_size, 
                initial_replay_size,
                max_replay_size, 
                warmup_transitions, 
                tau, 
                lr_alpha,
                critic_fit_params=None)
    return agent