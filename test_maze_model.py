"""Test maze/50000.pth model on different map environments"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import Environment
from model import Network
import configs

torch.manual_seed(configs.test_seed)
np.random.seed(configs.test_seed)
device = torch.device("cpu")
torch.set_num_threads(1)

color_map = np.array(
    [
        [255, 255, 255],  # white
        [190, 190, 190],  # gray
        [0, 191, 255],  # blue
        [255, 165, 0],  # orange
        [0, 250, 154],  # green
    ]
)


def test_maze_model_on_map_types(model_path, map_types, test_case):
    """
    Test a maze model on different map types
    Args:
        model_path: path to the model file
        map_types: list of map types to test on
        test_case: tuple of (length, num_agents, density)
    """
    network = Network()
    network.eval()
    network.to(device)
    
    # Load the maze model
    state_dict = torch.load(model_path, map_location=device)
    network.load_state_dict(state_dict)
    
    results = {}
    
    for map_type in map_types:
        print(f"\n========== Testing on {map_type} map ==========")
        
        # Set the map type in configs temporarily
        original_map_type = configs.map_type
        configs.map_type = map_type
        
        try:
            map_length, num_agents, density = test_case
            test_file = f"./test_set/{map_type}/{map_length}length_{num_agents}agents_{density}density.pth"
            
            if not os.path.exists(test_file):
                print(f"  Warning: Test set {test_file} not found. Skipping.")
                results[map_type] = None
                continue
            
            with open(test_file, "rb") as f:
                tests = pickle.load(f)
            
            # Test on first test case
            env = Environment()
            env.load(tests[0][0], tests[0][1], tests[0][2])
            
            obs, pos = env.observe()
            done = False
            network.reset()
            
            steps = 0
            while not done and env.steps < configs.max_episode_length:
                actions, _, _, _ = network.step(
                    torch.from_numpy(obs.astype(np.float32)).to(device),
                    torch.from_numpy(pos.astype(np.float32)).to(device),
                )
                (obs, pos), _, done, _ = env.step(actions)
                steps += 1
            
            success = np.array_equal(env.agents_pos, env.goals_pos)
            results[map_type] = {
                'success': success,
                'steps': steps,
                'test_case': tests[0]
            }
            
            print(f"  Success: {success}, Steps: {steps}")
            
        finally:
            configs.map_type = original_map_type
    
    return results


def make_animation_cross_map(model_path, map_type, test_case, steps=100):
    """
    Create animation for a model running on a specific map type
    Args:
        model_path: path to the model file
        map_type: map type to test on
        test_case: tuple of (length, num_agents, density)
        steps: maximum steps to visualize
    """
    # Set map type temporarily
    original_map_type = configs.map_type
    configs.map_type = map_type
    
    try:
        network = Network()
        network.eval()
        network.to(device)
        state_dict = torch.load(model_path, map_location=device)
        network.load_state_dict(state_dict)
        
        map_length, num_agents, density = test_case
        test_name = f"test_set/{map_type}/{map_length}length_{num_agents}agents_{density}density.pth"
        
        with open(test_name, "rb") as f:
            tests = pickle.load(f)
        
        env = Environment()
        env.load(tests[0][0], tests[0][1], tests[0][2])
        
        fig = plt.figure(figsize=(10, 10))
        
        done = False
        obs, pos = env.observe()
        
        imgs = []
        while not done and env.steps < steps:
            imgs.append([])
            map_img = np.copy(env.map)
            for agent_id in range(env.num_agents):
                if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                    map_img[tuple(env.agents_pos[agent_id])] = 4
                else:
                    map_img[tuple(env.agents_pos[agent_id])] = 2
                    map_img[tuple(env.goals_pos[agent_id])] = 3
            map_img = map_img.astype(np.uint8)
            
            img = plt.imshow(color_map[map_img], animated=True)
            imgs[-1].append(img)
            
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(
                zip(env.agents_pos, env.goals_pos)
            ):
                text = plt.text(
                    agent_y, agent_x, i, color="black", ha="center", va="center",
                    fontsize=8, fontweight='bold'
                )
                imgs[-1].append(text)
                if not np.array_equal(env.agents_pos[i], env.goals_pos[i]):
                    text = plt.text(
                        goal_y, goal_x, i, color="black", ha="center", va="center",
                        fontsize=8, fontweight='bold'
                    )
                    imgs[-1].append(text)
            
            actions, _, _, _ = network.step(
                torch.from_numpy(obs.astype(np.float32)).to(device),
                torch.from_numpy(pos.astype(np.float32)).to(device),
            )
            (obs, pos), _, done, _ = env.step(actions)
        
        if done and env.steps < steps:
            map_img = np.copy(env.map)
            for agent_id in range(env.num_agents):
                if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                    map_img[tuple(env.agents_pos[agent_id])] = 4
                else:
                    map_img[tuple(env.agents_pos[agent_id])] = 2
                    map_img[tuple(env.goals_pos[agent_id])] = 3
            map_img = map_img.astype(np.uint8)
            
            img = plt.imshow(color_map[map_img], animated=True)
            for _ in range(steps - env.steps):
                imgs.append([])
                imgs[-1].append(img)
                for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(
                    zip(env.agents_pos, env.goals_pos)
                ):
                    text = plt.text(
                        agent_y, agent_x, i, color="black", ha="center", va="center",
                        fontsize=8, fontweight='bold'
                    )
                    imgs[-1].append(text)
                    if not np.array_equal(env.agents_pos[i], env.goals_pos[i]):
                        text = plt.text(
                            goal_y, goal_x, i, color="black", ha="center", va="center",
                            fontsize=8, fontweight='bold'
                        )
                        imgs[-1].append(text)
        
        ani = animation.ArtistAnimation(
            fig, imgs, interval=600, blit=True, repeat_delay=1000
        )
        
        os.makedirs("videos", exist_ok=True)
        output_file = f"videos/maze_50000_on_{map_type}_{map_length}length_{num_agents}agents_{density}density.gif"
        ani.save(output_file, writer="pillow")
        print(f"Animation saved to {output_file}")
        plt.close(fig)
        
    finally:
        configs.map_type = original_map_type


def save_map_png(map_type, test_case, output_dir="map_examples"):
    """
    Save a PNG example of a map type
    Args:
        map_type: map type to generate
        test_case: tuple of (length, num_agents, density)
        output_dir: directory to save PNGs
    """
    original_map_type = configs.map_type
    configs.map_type = map_type
    
    try:
        map_length, num_agents, density = test_case
        test_file = f"./test_set/{map_type}/{map_length}length_{num_agents}agents_{density}density.pth"
        
        if not os.path.exists(test_file):
            print(f"  Warning: Test set {test_file} not found. Skipping PNG generation.")
            return
        
        with open(test_file, "rb") as f:
            tests = pickle.load(f)
        
        # Get the map from first test case
        map_data = tests[0][0]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        map_img = np.copy(map_data).astype(np.uint8)
        ax.imshow(color_map[map_img])
        ax.set_title(f"{map_type.capitalize()} Map Example\n{map_length}x{map_length}, {num_agents} agents, density={density}", 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{map_type}_map_example.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Map PNG saved to {output_file}")
        plt.close(fig)
        
    finally:
        configs.map_type = original_map_type


if __name__ == "__main__":
    model_path = "./models/maze/50000.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found!")
        exit(1)
    
    # Map types to test on
    map_types = ['random', 'warehouse', 'house', 'maze']
    
    # Test case: (map_length, num_agents, density)
    test_case = (30, 6, 0.3)
    
    print("=" * 60)
    print("Testing maze/50000.pth model on different map environments")
    print("=" * 60)
    
    # Test the model on different map types
    results = test_maze_model_on_map_types(model_path, map_types, test_case)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    for map_type, result in results.items():
        if result:
            print(f"{map_type:12s}: Success={result['success']}, Steps={result['steps']}")
        else:
            print(f"{map_type:12s}: Test set not found")
    
    # Create animation for one map type (choose the first available)
    print("\n" + "=" * 60)
    print("Creating animation...")
    print("=" * 60)
    for map_type in map_types:
        if results.get(map_type):
            print(f"Creating animation for {map_type} map...")
            make_animation_cross_map(model_path, map_type, test_case, steps=100)
            break
    
    # Generate PNG examples for each map type
    print("\n" + "=" * 60)
    print("Generating map PNG examples...")
    print("=" * 60)
    for map_type in map_types:
        print(f"Generating PNG for {map_type} map...")
        save_map_png(map_type, test_case)
    
    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)
