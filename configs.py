communication = False



############################################################
####################    environment     ####################
############################################################
map_length = 50
num_agents = 10
obs_radius = 4
reward_fn = dict(move=-0.075,
                stay_on_goal=0,
                stay_off_goal=-0.075,
                collision=-0.5,
                finish=3)

obs_shape = (6, 2*obs_radius+1, 2*obs_radius+1)
action_dim = 5



############################################################
####################         DQN        ####################
############################################################

# basic training setting
map_type = 'house'
num_actors = 20
log_interval = 10
training_times = 50000
save_interval=1000
gamma=0.99
batch_size=192
learning_starts=25000
target_network_update_freq=1000
save_path=f'./models/{map_type}'
max_episode_length = 256
seq_len = 16
load_model = False
load_path = './models/save_model/model_house/84000_house.pth'

Advantage_all = True
Sec_cons = True
lambdas = 0.001

actor_update_steps = 400

# gradient norm clipping
grad_norm_dqn=40

# n-step forward
forward_steps = 2

# global buffer
episode_capacity = 2048

# prioritized replay
prioritized_replay_alpha=0.6
prioritized_replay_beta=0.4

# curriculum learning
init_env_settings = (3, 10)
max_num_agents = 3
max_map_length = 20
pass_rate = 0.9
# dqn network setting
cnn_channel = 128
hidden_dim = 256

# communication
max_comm_agents = 10 # including agent itself, means one can at most communicate with (max_comm_agents-1) agents

# communication block
num_comm_layers = 2
num_comm_heads = 2


test_seed = 0
num_test_cases = 200

test_env_settings = ((15, 3, 0.2), (30, 3, 0.3), (30, 6, 0.3), (30, 12, 0.3), (30, 24, 0.3))

# List of map types to create test sets for and test models on
test_map_types = ['random', 'warehouse', 'house', 'maze']
