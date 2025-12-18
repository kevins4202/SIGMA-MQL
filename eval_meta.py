
import os
import torch
import numpy as np
import random
import csv

from meta_model import MetaNetwork
from environment import Environment
import configs
import meta_configs

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


TRAINING_TASKS = [
    ('random', 15, 3, 0.2),
    ('warehouse', 15, 3, 0.2),
    ('house', 15, 3, 0.2),
    ('maze', 15, 3, 0.2),
]


def evaluate_episode(env, network, device='cuda'):
    network.reset()
    obs, pos = env.observe()

    total_reward = 0
    episode_length = 0
    max_steps = configs.max_episode_length

    with torch.no_grad():
        while episode_length < max_steps:
            # Convert to tensors
            obs_tensor = torch.FloatTensor(obs).to(device)
            pos_tensor = torch.FloatTensor(pos).to(device)

            # Get actions (list of actions for all agents)
            actions, _, _, _ = network.step(obs_tensor, pos_tensor)

            # Take step - env.step returns 4 values
            (next_obs, next_pos), rewards, done, info = env.step(actions)

            total_reward += sum(rewards)
            obs = next_obs
            pos = next_pos
            episode_length += 1

            if done:
                break

    success = done
    return total_reward, success, episode_length


def collect_episode_for_csc(env, meta_network, device='cuda'):
    meta_network.reset()
    obs, pos = env.observe()

    episode = {
        'observations': [],
        'actions': [],
        'rewards': [],
    }

    total_reward = 0
    episode_length = 0
    max_steps = configs.max_episode_length

    with torch.no_grad():
        while episode_length < max_steps:
            obs_tensor = torch.FloatTensor(obs).to(device)
            pos_tensor = torch.FloatTensor(pos).to(device)

            actions, _, _, _ = meta_network.step(obs_tensor, pos_tensor)

            (next_obs, next_pos), rewards, done, info = env.step(actions)

            episode['observations'].append(obs)
            episode['actions'].append(actions)
            episode['rewards'].append(rewards)

            total_reward += sum(rewards)
            obs = next_obs
            pos = next_pos
            episode_length += 1

            if done:
                break

    success = done
    return episode, total_reward, success


def build_trajectory_for_csc(meta_network, episode, device='cuda'):
    obs_seq = torch.FloatTensor(np.array(episode['observations'])).to(device)
    T, num_agents = obs_seq.size(0), obs_seq.size(1)
    obs_flat = obs_seq.view(T * num_agents, *obs_seq.shape[2:])

    with torch.no_grad():
        latents = meta_network.network.obs_encoder(obs_flat)

    latents = latents.view(T, num_agents, -1).mean(dim=1)
    obs_traj = latents.unsqueeze(1)

    actions_arr = np.array(episode['actions'])
    actions_first = actions_arr[:, 0:1]
    actions_traj = torch.LongTensor(actions_first).unsqueeze(-1).to(device)

    rewards_arr = np.array(episode['rewards'])
    rewards_sum = rewards_arr.sum(axis=1, keepdims=True)
    rewards_traj = torch.FloatTensor(rewards_sum).unsqueeze(-1).to(device)

    return obs_traj, actions_traj, rewards_traj


def adapt_on_task(meta_network, task_config, device='cuda'):
    map_type, map_len, num_agents, obstacle_density = task_config

    env_current = Environment(
        num_agents=num_agents,
        map_length=map_len,
        fix_density=obstacle_density
    )
    current_episode, _, _ = collect_episode_for_csc(env_current, meta_network, device)
    current_traj = build_trajectory_for_csc(meta_network, current_episode, device)

    other_trajs = []
    for train_map_type, t_len, t_agents, t_density in TRAINING_TASKS:
        if train_map_type == map_type:
            continue
        env_other = Environment(
            num_agents=t_agents,
            map_length=t_len,
            fix_density=t_density
        )
        other_episode, _, _ = collect_episode_for_csc(env_other, meta_network, device)
        other_trajs.append(build_trajectory_for_csc(meta_network, other_episode, device))

    if not other_trajs:
        return

    obs_current, acts_current, rews_current = current_traj
    obs_other = torch.cat([t[0] for t in other_trajs], dim=0)
    acts_other = torch.cat([t[1] for t in other_trajs], dim=0)
    rews_other = torch.cat([t[2] for t in other_trajs], dim=0)

    current_batch = (obs_current, acts_current, rews_current)
    other_batch = (obs_other, acts_other, rews_other)

    meta_network.train()
    meta_network.train_covariate_shift_correction(current_batch, other_batch)
    _ = meta_network.get_importance_weights(current_batch)
    meta_network.eval()


def evaluate_on_task(meta_network, task_config, num_episodes=20, device='cuda'):
    map_type, map_len, num_agents, obstacle_density = task_config

    # Set the map type in configs
    original_map_type = configs.map_type
    configs.map_type = map_type

    adapt_on_task(meta_network, task_config, device)

    network = meta_network.network

    rewards = []
    successes = []
    episode_lengths = []

    for ep_idx in range(num_episodes):
        # Create new environment
        env = Environment(
            num_agents=num_agents,
            map_length=map_len,
            fix_density=obstacle_density
        )

        # Evaluate
        total_reward, success, ep_len = evaluate_episode(env, network, device)

        rewards.append(total_reward)
        successes.append(float(success))
        episode_lengths.append(ep_len)

    metrics = {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'success_rate': np.mean(successes),
        'avg_episode_length': np.mean(episode_lengths),
    }

    # Restore original map type
    configs.map_type = original_map_type

    return metrics


def test_meta_models(trained_map_types, test_task_configs, num_episodes=20, device='cuda'): # noqa: E501
    # Prepare CSV output
    csv_data = []
    csv_headers = [
        "trained_on",
        "tested_on",
        "map_length",
        "num_agents",
        "density",
        "success_rate",
        "avg_reward",
        "std_reward",
        "avg_episode_length",
    ]

    all_results = {}

    # Load and test each trained model
    for trained_map_type in trained_map_types:
        model_path = f'./models/{trained_map_type}/meta_model_simple.pth'

        if not os.path.exists(model_path):
            print(f"\nWarning: Model not found at {model_path}. Skipping {trained_map_type}.")
            continue

        print(f"\n{'='*80}")
        print(f"Loading model trained on: {trained_map_type}")
        print(f"{'='*80}")

        # Create and load meta network
        meta_network = MetaNetwork(
            input_shape=configs.obs_shape,
            cnn_channel=configs.cnn_channel,
            hidden_dim=configs.hidden_dim,
            max_comm_agents=configs.max_comm_agents,
            context_hidden_size=meta_configs.context_hidden_size,
            lam_csc=meta_configs.lam_csc,
            beta_clip=meta_configs.beta_clip,
            device=device
        ).to(device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        meta_network.network.load_state_dict(checkpoint['network'])
        meta_network.context_encoder.load_state_dict(checkpoint['context_encoder'])
        meta_network.eval()

        print(f"Model loaded successfully from {model_path}")

        all_results[trained_map_type] = {}

        # Test on each task configuration
        for test_config in test_task_configs:
            test_map_type, map_len, num_agents, obstacle_density = test_config

            print(f"\nTesting on: {test_map_type} (length={map_len}, agents={num_agents}, density={obstacle_density})")

            # Evaluate
            metrics = evaluate_on_task(
                meta_network,
                test_config,
                num_episodes=num_episodes,
                device=device
            )

            all_results[trained_map_type][test_map_type] = metrics

            # Print results
            print(f"  Success Rate: {metrics['success_rate']:.2%}")
            print(f"  Avg Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
            print(f"  Avg Episode Length: {metrics['avg_episode_length']:.1f}")

            # Add to CSV data
            csv_data.append([
                trained_map_type,
                test_map_type,
                map_len,
                num_agents,
                obstacle_density,
                f"{metrics['success_rate']*100:.2f}",
                f"{metrics['avg_reward']:.2f}",
                f"{metrics['std_reward']:.2f}",
                f"{metrics['avg_episode_length']:.1f}",
            ])

    # Save CSV file
    csv_filename = "meta_test_results.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        writer.writerows(csv_data)

    print(f"\n{'='*80}")
    print(f"Results saved to {csv_filename}")
    print(f"{'='*80}")

    return all_results

def main():
    """Main evaluation function"""

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Define the map types we trained on
    trained_map_types = ['random', 'warehouse', 'house', 'maze']

    test_task_configs = [
        ('random', 15, 3, 0.2),
        ('warehouse', 15, 3, 0.2),
        ('house', 15, 3, 0.2),
        ('maze', 15, 3, 0.2),
    ]

    print(f"\n{'='*80}")
    print(f"Cross-Task Evaluation of Meta Models")
    print(f"Testing {len(trained_map_types)} trained models on {len(test_task_configs)} tasks")
    print(f"{'='*80}")

    # Run cross-task evaluation
    all_results = test_meta_models(
        trained_map_types=trained_map_types,
        test_task_configs=test_task_configs,
        num_episodes=20,
        device=device
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
