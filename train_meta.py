import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from meta_model import MetaNetwork
from environment import Environment
import configs
import meta_configs

import wandb

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


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


def collect_episode(env, network, device='cuda'):
    network.reset()  
    obs, pos = env.observe()  # env.observe() returns (obs, pos) tuple

    episode = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'dones': []
    }

    total_reward = 0
    step_count = 0
    max_steps = configs.max_episode_length

    with torch.no_grad():
        while step_count < max_steps:
            # Convert to tensors
            obs_tensor = torch.FloatTensor(obs).to(device)
            pos_tensor = torch.FloatTensor(pos).to(device)

            # Get actions from network (returns list of actions for all agents)
            actions, _, _, _ = network.step(obs_tensor, pos_tensor)  

            # Take step in environment - env.step returns 4 values
            (next_obs, next_pos), rewards, done, info = env.step(actions)

            episode['observations'].append(obs)
            episode['actions'].append(actions)
            episode['rewards'].append(rewards)
            episode['next_observations'].append(next_obs)
            episode['dones'].append(done)

            total_reward += sum(rewards)  # Sum rewards from all agents
            obs = next_obs
            pos = next_pos
            step_count += 1

            if done:
                break

    success = done
    return episode, total_reward, success


def train_on_single_task(meta_network, task_config, task_traj_store, num_episodes=100, device='cuda'):
    map_type, map_len, num_agents, obstacle_density = task_config

    # Set the map type in configs
    original_map_type = configs.map_type
    configs.map_type = map_type

    # Optimizer for the underlying SIGMA network
    optimizer = Adam(meta_network.network.parameters(), lr=1e-4)

    # Training metrics
    all_rewards = []
    all_successes = []

    print(f"\nTraining on task: map_type={map_type}, map_len={map_len}, num_agents={num_agents}, density={obstacle_density}")

    for episode_idx in range(num_episodes):
        # Create new environment for this episode
        # Environment takes num_agents, map_length, and fix_density parameters
        env = Environment(
            num_agents=num_agents,
            map_length=map_len,
            fix_density=obstacle_density
        )

        # Collect episode
        episode, total_reward, success = collect_episode(env, meta_network, device)

        all_rewards.append(total_reward)
        all_successes.append(float(success))

        # Store trajectory for meta components (context encoder + CSC)
        traj = build_trajectory_for_csc(meta_network, episode, device)
        task_traj_store.setdefault(map_type, []).append(traj)

        loss = -torch.tensor(total_reward, dtype=torch.float32, device=device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Train CSC occasionally when we have trajectories from at least two tasks
        if len(task_traj_store) > 1 and (episode_idx + 1) % 10 == 0:
            current_traj = traj
            other_trajs = []
            for other_map_type, trajs in task_traj_store.items():
                if other_map_type == map_type:
                    continue
                if trajs:
                    other_trajs.append(trajs[-1])

            if other_trajs:
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

        # Print progress
        if (episode_idx + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            success_rate = np.mean(all_successes[-10:])
            print(f"  Episode {episode_idx + 1}/{num_episodes} - "
                  f"Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2%}")

    metrics = {
        'avg_reward': np.mean(all_rewards),
        'success_rate': np.mean(all_successes),
        'final_10_avg_reward': np.mean(all_rewards[-10:]),
        'final_10_success_rate': np.mean(all_successes[-10:])
    }

    # Restore original map type
    configs.map_type = original_map_type

    return metrics


def main():
    """Main training function"""

    wandb.init(
        entity="sigmamql",
        project="SIGMA-MQL",
        group="meta_simple",
        name=f"meta_simple_{os.getpid()}",
    )

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create meta network
    print("Initializing MetaNetwork...")
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

    training_tasks = [
        ('random', 15, 3, 0.2),     # Random obstacles
        ('warehouse', 15, 3, 0.2),  # Warehouse layout
        ('house', 15, 3, 0.2),      # House layout
        ('maze', 15, 3, 0.2),       # Maze layout
    ]

    # Train on each task
    all_task_metrics = []
    task_traj_store = {}

    for task_idx, task_config in enumerate(training_tasks):
        print(f"\n{'='*70}")
        print(f"Task {task_idx + 1}/{len(training_tasks)}")
        print(f"{'='*70}")

        metrics = train_on_single_task(
            meta_network,
            task_config,
            task_traj_store,
            num_episodes=50,
            device=device,
        )

        all_task_metrics.append(metrics)

        print(f"\nTask {task_idx + 1} Results:")
        print(f"  Average Reward: {metrics['avg_reward']:.2f}")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")
        print(f"  Final 10 Episodes Avg Reward: {metrics['final_10_avg_reward']:.2f}")
        print(f"  Final 10 Episodes Success Rate: {metrics['final_10_success_rate']:.2%}")

        wandb.log({
            f'task_{task_idx}/avg_reward': metrics['avg_reward'],
            f'task_{task_idx}/success_rate': metrics['success_rate'],
            f'task_{task_idx}/final_reward': metrics['final_10_avg_reward'],
            f'task_{task_idx}/final_success': metrics['final_10_success_rate'],
        })
        
    # Save the final meta network
    save_dir = configs.save_path
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'meta_model_simple.pth')

    torch.save({
        'network': meta_network.network.state_dict(),
        'context_encoder': meta_network.context_encoder.state_dict(),
    }, save_path)

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Model saved to: {save_path}")

    # Print summary
    print("\nSummary across all tasks:")
    avg_reward_all = np.mean([m['avg_reward'] for m in all_task_metrics])
    success_rate_all = np.mean([m['success_rate'] for m in all_task_metrics])
    print(f"  Average Reward: {avg_reward_all:.2f}")
    print(f"  Average Success Rate: {success_rate_all:.2%}")

    wandb.finish()


if __name__ == "__main__":
    main()
