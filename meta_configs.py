"""
Meta-Q Learning configurations for SIGMA
Based on meta-q-learning/main.py and run_script.py hyperparameters
"""

import configs  # Import base SIGMA configs

############################################################
####################  Meta-Learning Settings  ##############
############################################################

# Enable meta-learning
enable_meta_learning = True

# Context Encoder Settings
context_hidden_size = 128  # Hidden size for context GRU (half of SIGMA's hidden_dim)
history_length = 16  # Length of trajectory history for context encoding (matches SIGMA's seq_len)

# Covariate Shift Correction Settings
# lam_csc: regularization parameter for logistic regression (C = lam_csc)
# Smaller values mean stronger regularization
# Values from meta-q-learning: 0.05-0.5 depending on environment
lam_csc = 0.1  # Default, can be tuned per environment

# beta_clip: clip importance weights to prevent extreme values
# Values from meta-q-learning: 1.0-2.0 depending on environment
beta_clip = 1.2  # Default

# use_normalized_beta: whether to normalize beta scores using logistic regression probabilities
use_normalized_beta = True

# max_iter_logistic: maximum iterations for logistic regression training
max_iter_logistic = 2000

# use_ess_clipping: whether to use effective sample size for clipping
use_ess_clipping = False

# enable_beta_obs_cxt: whether to concatenate observation with context for discriminator
# If True: discriminator uses [obs, context]
# If False: discriminator uses only context
enable_beta_obs_cxt = False

# Task Sampling Settings
num_tasks_sample = 5  # Number of tasks to sample per training iteration
num_train_tasks = 40  # Total number of training tasks
num_eval_tasks = 10   # Number of evaluation tasks

# Meta-Learning Training Settings
num_train_steps = 500  # Number of gradient steps per task during meta-training
snap_iter_nums = 5     # Number of adaptation steps during evaluation (test-time adaptation)
main_snap_iter_nums = 400  # Number of adaptation steps on training tasks with CSC
sample_mult = 5  # Multiplier for batch size during adaptation

# Adaptation Settings
enable_adaptation = True  # Whether to enable test-time adaptation
num_initial_steps = 1500  # Number of initial steps to collect data before adaptation

# Proximal Point Optimization
# prox_coef: coefficient for proximal term in loss function
# Prevents adapted model from deviating too much from meta-learned initialization
prox_coef = 0.1

# reset_optims: whether to reset optimizers at the start of adaptation
reset_optims = False

# Buffer Settings for Meta-Learning
snapshot_size = 2000  # Size of snapshot buffer for each task
min_buffer_size = 100000  # Minimum buffer size before using num_train_steps

# Evaluation Settings
unbounded_eval_hist = False  # If True, use max_path_length as history length during eval
use_epi_len_steps = True     # Adjust number of train steps based on episode length

# Meta-Batch Settings
meta_batch_size = 10  # Number of tasks per meta-update

# Learning Rate Schedule
lr_milestone = -1  # Reduce learning rate after this many iterations (-1 to disable)
lr_gamma = 0.8     # Learning rate decay factor

# Sampling Style
# 'replay': sample from replay buffer
# other options can be added as needed
sampling_style = 'replay'

############################################################
####################  Integration with SIGMA  ##############
############################################################

# Use SIGMA's base hyperparameters and override as needed
# DQN settings from SIGMA configs
gamma = configs.gamma
batch_size = configs.batch_size
learning_rate = 1e-4  # Standard learning rate for meta-learning

# Network architecture (from SIGMA)
hidden_dim = configs.hidden_dim
cnn_channel = configs.cnn_channel
obs_shape = configs.obs_shape
action_dim = configs.action_dim

# Training settings (from SIGMA)
seq_len = configs.seq_len
forward_steps = configs.forward_steps
learning_starts = configs.learning_starts
target_network_update_freq = configs.target_network_update_freq

# Communication settings (from SIGMA)
max_comm_agents = configs.max_comm_agents
num_comm_layers = configs.num_comm_layers
num_comm_heads = configs.num_comm_heads

# Sheaf theory settings (from SIGMA)
Advantage_all = configs.Advantage_all
Sec_cons = configs.Sec_cons
lambdas = configs.lambdas

############################################################
####################  Environment-Specific Settings  #######
############################################################

# These can be tuned based on the environment
# Following meta-q-learning/run_script.py conventions

def get_meta_config_for_env(env_type='random'):
    """
    Get environment-specific meta-learning hyperparameters

    Args:
        env_type: type of environment ('random', 'house', 'maze', etc.)

    Returns:
        dict of hyperparameters
    """
    base_config = {
        'context_hidden_size': context_hidden_size,
        'history_length': history_length,
        'lam_csc': lam_csc,
        'beta_clip': beta_clip,
        'use_normalized_beta': use_normalized_beta,
        'enable_adaptation': enable_adaptation,
        'num_train_steps': num_train_steps,
        'snap_iter_nums': snap_iter_nums,
        'main_snap_iter_nums': main_snap_iter_nums,
        'sample_mult': sample_mult,
        'prox_coef': prox_coef,
    }

    # Environment-specific tuning
    if env_type == 'random':
        # Default settings work well for random maps
        pass
    elif env_type == 'house':
        # House maps may need more adaptation steps
        base_config['main_snap_iter_nums'] = 500
        base_config['lam_csc'] = 0.05
    elif env_type == 'maze':
        # Maze maps may need stronger regularization
        base_config['lam_csc'] = 0.15
        base_config['beta_clip'] = 1.5
    elif env_type == 'warehouse':
        # Warehouse maps
        base_config['main_snap_iter_nums'] = 400
        base_config['lam_csc'] = 0.1
    elif env_type == 'tunnels':
        # Tunnel maps
        base_config['main_snap_iter_nums'] = 400
        base_config['lam_csc'] = 0.1

    return base_config

############################################################
####################  Logging and Evaluation  ##############
############################################################

# CSV headers for adaptation logging
adapt_csv_header = [
    'eps_num', 'iter', 'critic_loss', 'actor_loss',
    'csc_samples_neg', 'csc_samples_pos', 'train_acc',
    'snap_iter', 'beta_score', 'main_critic_loss',
    'main_actor_loss', 'main_beta_score', 'main_prox_critic',
    'main_prox_actor', 'main_avg_prox_coef',
    'tidx', 'avg_rewards', 'one_raw_reward'
]

# Evaluation frequency
eval_freq = 10000  # Evaluate every N timesteps

# Save frequency
save_freq = 250  # Save model every N episodes

############################################################
####################  Weights & Biases (wandb)  ############
############################################################

# If using wandb for logging meta-learning metrics
use_wandb = True
wandb_project = "SIGMA-MQL"
wandb_entity = None  # Set to your wandb username/team

# Metrics to log
wandb_metrics = [
    'meta/train_loss',
    'meta/adapt_loss',
    'meta/beta_score',
    'meta/csc_accuracy',
    'meta/context_norm',
    'meta/importance_weight_mean',
    'meta/importance_weight_std',
]
