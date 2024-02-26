from itertools import product

from experiment_launcher import Launcher
# from experiment_launcher.utils import bool_local_cluster

LOCAL = True
TEST = False 
USE_CUDA = False

PARTITION = 'deflt*'  # 'amd', 'rtx', 'deflt*' for lichtenberg
GRES = 'gpu:rtx3080:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1

CONDA_ENV = 'base'  # None

N_SEEDS = 25

if LOCAL:
    JOBLIB_PARALLEL_JOBS = 5  # os.cpu_count()
else:
    JOBLIB_PARALLEL_JOBS = 5


N_CORES_JOB = 1
MEMORY_SINGLE_JOB = 2000


launcher = Launcher(exp_name='experiments', python_file='experiment', project_name='project01616',  # 'project01616',
                    n_exps=N_SEEDS,
                    joblib_n_jobs=JOBLIB_PARALLEL_JOBS,
                    n_cores=JOBLIB_PARALLEL_JOBS * N_CORES_JOB,
                    memory_per_core=MEMORY_SINGLE_JOB,
                    days=1,
                    hours=12,
                    minutes=0,
                    seconds=0,
                    partition=PARTITION,
                    conda_env=CONDA_ENV,
                    gres=GRES,
                    use_timestamp=False
                    )

envs = {
        # 'table': {'lr_alpha__': 1e-6, 'actor_lr__': 1e-5, 'critic_lr__': 1e-5},
        # 'shelf': {'lr_alpha__': 1e-6, 'actor_lr__': 1e-5, 'critic_lr__': 1e-5},
        # 'table_se': {'lr_alpha__': 1e-6, 'actor_lr__': 1e-5, 'critic_lr__': 1e-5},
        # 'shelf_se': {'lr_alpha__': 1e-6, 'actor_lr__': 1e-5, 'critic_lr__': 1e-5},
        # 'table_atacom': {'lr_alpha__': 1e-6, 'actor_lr__': 3e-4, 'critic_lr__': 3e-4},
        # 'shelf_atacom': {'lr_alpha__': 1e-6, 'actor_lr__': 3e-4, 'critic_lr__': 3e-4},
        # 'hri_atacom': {'n_features': '512-512-256'}
        # 'shelf_real_atacom': {},
        # 'nav_atacom': {'timestep': 1 / 30., 'n_intermediate_steps': 1, 'lr_alpha__': 1e-6, 'actor_lr__': 3e-4, 'critic_lr__': 3e-4},
        # 'fetch_nav_atacom': {'timestep': 1 / 30., 'n_intermediate_steps': 1, 'lr_alpha__': 5e-6, 'actor_lr__': 5e-5, 'critic_lr__': 5e-5}
        # 'table_se': {'lr_alpha__': 1e-5, 'actor_lr__': 3e-4, 'critic_lr__': 3e-4},
        # 'shelf_real_se': {'lr_alpha__': 1e-5, 'actor_lr__': 3e-4, 'critic_lr__': 3e-4},
        # 'nav_se': {'timestep': 1 / 30., 'n_intermediate_steps': 1}

        # Linear Attractor
        'two_dof': {},
        # 'table_atacom': {},
        # 'shelf': {},
        # 'shelf_atacom': {},
        # 'nav': {'timestep': 1 / 30., 'n_intermediate_steps': 1},
        # 'nav_atacom': {'timestep': 1 / 30., 'n_intermediate_steps': 1}
        }

algs = {
    'sac': {'n_steps_per_fit': 1},
    # 'trpo': {'n_steps_per_fit': 600},
    # 'ppo': {'n_steps_per_fit': 600},
    # 'td3': {'n_steps_per_fit': 1},
    # 'ddpg': {'n_steps_per_fit': 1}

    # Linear Attractor
    # 'la': {'n_steps_per_fit': 0}
}

params_list = [
    {'preprocessor': None}
    # {'lr_alpha__': 1e-5, 'actor_lr__': 3e-4, 'critic_lr__': 3e-4},
    # {'lr_alpha__': 1e-6, 'actor_lr__': 3e-4, 'critic_lr__': 3e-4},
    # {'lr_alpha__': 1e-5, 'actor_lr__': 1e-5, 'critic_lr__': 1e-5},
    # {'lr_alpha__': 1e-6, 'actor_lr__': 1e-5, 'critic_lr__': 1e-5},
    # {'lr_alpha__': 5e-6, 'actor_lr__': 5e-5, 'critic_lr__': 5e-5},
]

for env in envs.keys():
    for key in algs.keys():
        for params_dict in params_list:
            launcher.add_experiment(
                use_cuda=USE_CUDA,
                quiet=True,
                debug_gui=False,
                n_steps_per_fit=algs[key]['n_steps_per_fit'],
                n_epochs=200,
                collide_termination=True,
                save_key_frame=False,
                # A subdirectory will be created for parameters with a trailing double underscore.
                env__=env,
                alg__=key,
                **envs[env],
                **params_dict,
            )

launcher.run(LOCAL, TEST)

