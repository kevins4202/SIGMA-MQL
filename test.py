"""create test set and test model"""

import os
import random
import pickle
import csv
import multiprocessing as mp
from typing import Union, List, Tuple
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import Environment
from model import Network
import configs

torch.manual_seed(configs.test_seed)
np.random.seed(configs.test_seed)
random.seed(configs.test_seed)
test_num = 200
device = torch.device("cpu")
torch.set_num_threads(1)


def create_test(test_map_type, test_env_settings, num_test_cases):
    """
    Create test sets for a specific map type
    Temporarily sets configs.map_type to generate the correct map type
    """
    original_map_type = configs.map_type
    configs.map_type = test_map_type

    try:
        for map_length, num_agents, density in test_env_settings:

            name = "./test_set/{}/{}length_{}agents_{}density.pth".format(
                test_map_type, map_length, num_agents, density
            )
            os.makedirs(os.path.dirname(name), exist_ok=True)
            print(
                "-----{} map: {}length {}agents {}density-----".format(
                    test_map_type, map_length, num_agents, density
                )
            )

            tests = []

            env = Environment(
                fix_density=density, num_agents=num_agents, map_length=map_length
            )

            for _ in tqdm(range(num_test_cases)):
                tests.append(
                    (np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos))
                )
                env.reset(num_agents=num_agents, map_length=map_length)
            print()

            with open(name, "wb") as f:
                pickle.dump(tests, f)
    finally:
        configs.map_type = original_map_type


def create_fair_test(test_map_type, map_length, density, agent_counts, num_test_cases):
    """
    Create test sets with the SAME maps across different agent counts.
    This allows fair comparison of how agent count affects performance.
    
    Args:
        test_map_type: Map type (e.g., 'house', 'maze', 'random', 'warehouse')
        map_length: Size of the map
        density: Obstacle density
        agent_counts: List of agent counts to test (e.g., [3, 6, 12, 24])
        num_test_cases: Number of test cases to generate
    """
    original_map_type = configs.map_type
    configs.map_type = test_map_type
    
    try:
        print(f"===== Creating fair test sets for {test_map_type} =====")
        print(f"Map: {map_length}x{map_length}, density: {density}")
        print(f"Agent counts: {agent_counts}")
        
        # First, generate the base maps using the maximum agent count
        # (ensures maps have enough space for all agent configurations)
        max_agents = max(agent_counts)
        base_env = Environment(fix_density=density, num_agents=max_agents, map_length=map_length)
        
        # Store just the maps
        base_maps = []
        print(f"Generating {num_test_cases} base maps...")
        for _ in tqdm(range(num_test_cases)):
            base_maps.append(np.copy(base_env.map))
            base_env.reset(num_agents=max_agents, map_length=map_length)
        
        # Now create test files for each agent count using the same maps
        for num_agents in agent_counts:
            tests = []
            env = Environment(fix_density=density, num_agents=num_agents, map_length=map_length)
            
            print(f"Placing {num_agents} agents on maps...")
            for base_map in tqdm(base_maps):
                # Load the base map and generate new agent/goal positions
                env.load_map_only(base_map, num_agents)
                tests.append(
                    (np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos))
                )
            
            name = f"./test_set/{test_map_type}/{map_length}length_{num_agents}agents_{density}density_fair.pth"
            os.makedirs(os.path.dirname(name), exist_ok=True)
            
            with open(name, "wb") as f:
                pickle.dump(tests, f)
            
            print(f"Saved: {name}")
    
    finally:
        configs.map_type = original_map_type


# def render_test_case(model, test_case, number):

#     network = Network()
#     network.eval()
#     network.to(device)

#     with open(test_case, 'rb') as f:
#         tests = pickle.load(f)

#     model_name = model
#     while os.path.exists('./models/{}.pth'.format(model_name)):
#         state_dict = torch.load('./models/{}.pth'.format(model_name), map_location=device)
#         network.load_state_dict(state_dict)
#         env = Environment()

#         case = 2
#         show = False
#         show_steps = 100

#         fail = 0
#         steps = []

#         start = time.time()
#         for i in range(test_num):
#             env.load(tests[i][0], tests[i][1], tests[i][2])

#             done = False
#             network.reset()

#             while not done and env.steps < configs.max_episode_length:
#                 if i == case and show and env.steps < show_steps:
#                     env.render()

#                 obs, pos = env.observe()

#                 actions, _, _, _ = network.step(torch.FloatTensor(obs).to(device), torch.FloatTensor(pos).to(device))

#                 _, _, done, _ = env.step(actions)

#             steps.append(env.steps)

#             if not np.array_equal(env.agents_pos, env.goals_pos):
#                 fail += 1
#                 if show:
#                     print(i)

#             if i == case and show:
#                 env.close(True)

#         f_rate = (test_num-fail)/test_num
#         mean_steps = sum(steps)/test_num
#         duration = time.time()-start

#         print('--------------{}---------------'.format(model_name))
#         print('finish: %.4f' %f_rate)
#         print('mean steps: %.2f' %mean_steps)
#         print('time spend: %.2f' %duration)

#         model_name -= configs.save_interval


def test_model(model_list: List[Tuple[str, int]]):
    """
    Test models on specified map types (legacy function for backwards compatibility)
    Args:
        model_list: List of tuples (map_type, model_number), e.g., [('house', 10000), ('random', 5000)]
    """
    network = Network()
    network.eval()
    network.to(device)
    test_set = configs.test_env_settings

    pool = mp.Pool(mp.cpu_count())

    # Prepare CSV output
    csv_data = []
    csv_headers = [
        "map_type",
        "model",
        "length",
        "agents",
        "density",
        "success_rate",
        "avg_steps",
    ]

    for test_map_type, model in model_list:
        model_path = f"./models/{test_map_type}/{model}.pth"
        if not os.path.exists(model_path):
            print(
                f"Warning: Model {model_path} not found. Skipping {test_map_type}, model {model}."
            )
            continue

        print(f"Testing {test_map_type} model {model}...")
        state_dict = torch.load(model_path, map_location=device)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        for case in test_set:
            map_length, num_agents, density = case
            test_file = f"./test_set/{test_map_type}/{map_length}length_{num_agents}agents_{density}density.pth"
            if not os.path.exists(test_file):
                print(f"  Warning: Test set {test_file} not found. Skipping.")
                continue

            with open(test_file, "rb") as f:
                tests = pickle.load(f)

            tests = [(test, network) for test in tests]
            ret = pool.map(test_one_case, tests)

            success = 0
            avg_step = 0

            for i, j, _ in ret:
                success += i
                avg_step += j

            success_rate = success / len(ret) * 100
            avg_steps = avg_step / len(ret)

            # Add to CSV data
            csv_data.append(
                [
                    test_map_type,
                    model,
                    map_length,
                    num_agents,
                    density,
                    f"{success_rate:.2f}",
                    f"{avg_steps:.2f}",
                ]
            )

            print(
                f"  {map_length}length {num_agents}agents {density}density: {success_rate:.2f}% success, {avg_steps:.2f} avg steps"
            )

    # Save CSV file
    csv_filename = "test_results.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        writer.writerows(csv_data)

    print(f"\nResults saved to {csv_filename}")


def test_model_custom(
    model_map_type: str,
    model_number: int,
    test_map_type: str,
    test_cases: List[Tuple[int, int, float]],
    use_fair: bool = False,
    csv_filename: str = "test_results_custom.csv",
    make_animations: bool = False,
    animation_steps: int = 100
):
    """
    Test a specific model on custom test cases with flexible parameters.
    
    Args:
        model_map_type: Map type the model was trained on (e.g., 'house', 'random')
        model_number: Model checkpoint number (e.g., 27000)
        test_map_type: Map type of test cases to use (can differ from model_map_type)
        test_cases: List of (map_length, num_agents, density) tuples to test
        use_fair: If True, use '_fair.pth' test files (same maps across agent counts)
        csv_filename: Output CSV filename
        make_animations: If True, create animation of first test case for each test_case
        animation_steps: Max steps for each animation
    
    Example:
        test_model_custom(
            model_map_type='house',
            model_number=27000,
            test_map_type='random',
            test_cases=[(30, 3, 0.3), (30, 6, 0.3), (30, 12, 0.3)],
            use_fair=True,
            make_animations=True
        )
    """
    network = Network()
    network.eval()
    network.to(device)

    pool = mp.Pool(mp.cpu_count())

    # Load model
    model_path = f"./models/{model_map_type}/{model_number}.pth"
    #TODO REMOVE
    model_path = f"./models/save_model/model_{model_map_type}/{model_number}_{model_map_type}.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return
    
    print(f"Loading model: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    network.load_state_dict(state_dict)
    network.eval()
    network.share_memory()

    # Prepare CSV output
    csv_data = []
    csv_headers = [
        "model_map_type",
        "model_number",
        "test_map_type",
        "length",
        "agents",
        "density",
        "use_fair",
        "success_rate",
        "avg_steps",
    ]

    print(f"Testing on {test_map_type} test cases...")
    
    for map_length, num_agents, density in test_cases:
        suffix = "_fair" if use_fair else ""
        test_file = f"./test_set/{test_map_type}/{map_length}length_{num_agents}agents_{density}density{suffix}.pth"
        
        if not os.path.exists(test_file):
            print(f"  Warning: Test set {test_file} not found. Skipping.")
            continue

        with open(test_file, "rb") as f:
            tests = pickle.load(f)

        tests = [(test, network) for test in tests]
        ret = pool.map(test_one_case, tests)

        success = 0
        avg_step = 0

        for i, j, _ in ret:
            success += i
            avg_step += j

        success_rate = success / len(ret) * 100
        avg_steps = avg_step / len(ret)

        # Add to CSV data
        csv_data.append(
            [
                model_map_type,
                model_number,
                test_map_type,
                map_length,
                num_agents,
                density,
                use_fair,
                f"{success_rate:.2f}",
                f"{avg_steps:.2f}",
            ]
        )

        print(
            f"  {map_length}x{map_length}, {num_agents} agents, {density} density: {success_rate:.2f}% success, {avg_steps:.2f} avg steps"
        )
        
        # Create animation of first test case if requested
        if make_animations:
            make_animation_custom(
                model_map_type=model_map_type,
                model_number=model_number,
                test_map_type=test_map_type,
                test_case=(map_length, num_agents, density),
                use_fair=use_fair,
                steps=animation_steps,
                test_index=0
            )

    # Save CSV file
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        writer.writerows(csv_data)

    print(f"\nResults saved to {csv_filename}")


def test_one_case(args):

    env_set, network = args

    env = Environment()
    env.load(env_set[0], env_set[1], env_set[2])
    obs, pos = env.observe()

    done = False
    network.reset()

    step = 0
    section = 0
    while not done and env.steps < configs.max_episode_length:
        actions, _, _, _, sheaf_section = network.step_test(
            torch.as_tensor(obs.astype(np.float32)),
            torch.as_tensor(pos.astype(np.float32)),
        )
        (obs, pos), _, done, _ = env.step(actions)
        step += 1
        section += sheaf_section
    section = section / step
    return np.array_equal(env.agents_pos, env.goals_pos), step, section


def make_animation(
    test_map_type: str, model_name: int, test_case: tuple, steps: int = 25
):
    """
    visualize running results
    test_map_type: map type (e.g., 'house', 'random', 'maze')
    model_name: model number in 'models' file
    test_case: tuple of (length, num_agents, density) for the test case
    steps: how many steps to visualize in test case
    """
    color_map = np.array(
        [
            [255, 255, 255],  # white
            [190, 190, 190],  # gray
            [0, 191, 255],  # blue
            [255, 165, 0],  # orange
            [0, 250, 154],
        ]
    )  # green

    network = Network()
    network.eval()
    network.to(device)
    state_dict = torch.load(
        f"models/{test_map_type}/{model_name}.pth", map_location=device
    )
    network.load_state_dict(state_dict)

    test_name = f"test_set/{test_map_type}/{test_case[0]}length_{test_case[1]}agents_{test_case[2]}density.pth"

    with open(test_name, "rb") as f:
        tests = pickle.load(f)

    env = Environment()
    env.load(tests[0][0], tests[0][1], tests[0][2])

    fig = plt.figure()

    done = False
    obs, pos = env.observe()

    imgs = []
    while not done and env.steps < steps:
        imgs.append([])
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
                map[tuple(env.goals_pos[agent_id])] = 3
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)

        imgs[-1].append(img)

        for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(
            zip(env.agents_pos, env.goals_pos)
        ):
            text = plt.text(
                agent_y, agent_x, i, color="black", ha="center", va="center"
            )
            imgs[-1].append(text)
            text = plt.text(goal_y, goal_x, i, color="black", ha="center", va="center")
            imgs[-1].append(text)

        actions, _, _, _ = network.step(
            torch.from_numpy(obs.astype(np.float32)).to(device),
            torch.from_numpy(pos.astype(np.float32)).to(device),
        )
        (obs, pos), _, done, _ = env.step(actions)
        # print(done)

    if done and env.steps < steps:
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
                map[tuple(env.goals_pos[agent_id])] = 3
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)
        for _ in range(steps - env.steps):
            imgs.append([])
            imgs[-1].append(img)
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(
                zip(env.agents_pos, env.goals_pos)
            ):
                text = plt.text(
                    agent_y, agent_x, i, color="black", ha="center", va="center"
                )
                imgs[-1].append(text)
                text = plt.text(
                    goal_y, goal_x, i, color="black", ha="center", va="center"
                )
                imgs[-1].append(text)

    ani = animation.ArtistAnimation(
        fig, imgs, interval=600, blit=True, repeat_delay=1000
    )

    # Create videos directory if it doesn't exist
    os.makedirs("videos", exist_ok=True)

    # Save as GIF
    output_file = f"videos/{test_map_type}_{model_name}_{test_case[0]}length_{test_case[1]}agents_{test_case[2]}density_0.gif"
    ani.save(output_file, writer="pillow")


def make_animation_custom(
    model_map_type: str,
    model_number: int,
    test_map_type: str,
    test_case: tuple,
    use_fair: bool = False,
    steps: int = 100,
    test_index: int = 0
):
    """
    Create animation with flexible model/test case configuration.
    
    Args:
        model_map_type: Map type the model was trained on
        model_number: Model checkpoint number
        test_map_type: Map type of test cases
        test_case: Tuple of (length, num_agents, density)
        use_fair: If True, use '_fair.pth' test files
        steps: Max steps to visualize
        test_index: Which test case to animate (0 = first)
    """
    color_map = np.array(
        [
            [255, 255, 255],  # white - empty
            [190, 190, 190],  # gray - obstacle
            [0, 191, 255],    # blue - agent
            [255, 165, 0],    # orange - goal
            [0, 250, 154],    # green - agent on goal
        ]
    )

    network = Network()
    network.eval()
    network.to(device)
    
    # Load model - using the TODO path format from user's changes
    model_path = f"./models/save_model/model_{model_map_type}/{model_number}_{model_map_type}.pth"
    if not os.path.exists(model_path):
        # Fallback to standard path
        model_path = f"./models/{model_map_type}/{model_number}.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return
    
    state_dict = torch.load(model_path, map_location=device)
    network.load_state_dict(state_dict)

    # Load test case
    suffix = "_fair" if use_fair else ""
    test_file = f"./test_set/{test_map_type}/{test_case[0]}length_{test_case[1]}agents_{test_case[2]}density{suffix}.pth"
    
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found.")
        return

    with open(test_file, "rb") as f:
        tests = pickle.load(f)

    if test_index >= len(tests):
        print(f"Error: test_index {test_index} out of range (max {len(tests)-1})")
        return

    env = Environment()
    env.load(tests[test_index][0], tests[test_index][1], tests[test_index][2])

    fig = plt.figure()

    done = False
    obs, pos = env.observe()
    network.reset()

    imgs = []
    while not done and env.steps < steps:
        imgs.append([])
        map_display = np.copy(env.map)
        # Handle negative obstacle values
        map_display = (map_display != 0).astype(np.int_)
        
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map_display[tuple(env.agents_pos[agent_id])] = 4
            else:
                map_display[tuple(env.agents_pos[agent_id])] = 2
                map_display[tuple(env.goals_pos[agent_id])] = 3
        map_display = map_display.astype(np.uint8)

        img = plt.imshow(color_map[map_display], animated=True)
        imgs[-1].append(img)

        for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(
            zip(env.agents_pos, env.goals_pos)
        ):
            text = plt.text(agent_y, agent_x, i, color="black", ha="center", va="center")
            imgs[-1].append(text)
            text = plt.text(goal_y, goal_x, i, color="black", ha="center", va="center")
            imgs[-1].append(text)

        actions, _, _, _ = network.step(
            torch.from_numpy(obs.astype(np.float32)).to(device),
            torch.from_numpy(pos.astype(np.float32)).to(device),
        )
        (obs, pos), _, done, _ = env.step(actions)

    # If done early, pad with final frame
    if done and env.steps < steps:
        map_display = np.copy(env.map)
        map_display = (map_display != 0).astype(np.int_)
        
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map_display[tuple(env.agents_pos[agent_id])] = 4
            else:
                map_display[tuple(env.agents_pos[agent_id])] = 2
                map_display[tuple(env.goals_pos[agent_id])] = 3
        map_display = map_display.astype(np.uint8)

        img = plt.imshow(color_map[map_display], animated=True)
        for _ in range(min(10, steps - env.steps)):  # Pad with up to 10 frames
            imgs.append([])
            imgs[-1].append(img)
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(
                zip(env.agents_pos, env.goals_pos)
            ):
                text = plt.text(agent_y, agent_x, i, color="black", ha="center", va="center")
                imgs[-1].append(text)
                text = plt.text(goal_y, goal_x, i, color="black", ha="center", va="center")
                imgs[-1].append(text)

    ani = animation.ArtistAnimation(
        fig, imgs, interval=600, blit=True, repeat_delay=1000
    )

    # Create videos directory
    os.makedirs("videos", exist_ok=True)

    # Save as GIF
    fair_suffix = "_fair" if use_fair else ""
    output_file = f"videos/{model_map_type}_{model_number}_on_{test_map_type}_{test_case[0]}length_{test_case[1]}agents_{test_case[2]}density{fair_suffix}.gif"
    ani.save(output_file, writer="pillow")
    plt.close(fig)
    
    print(f"Saved animation: {output_file}")
    return output_file


def plot_random_maps(map_length: int = 10, density: float = 0.3):
    """
    Generate and plot a random map for each of the 4 map types.
    Uses the Environment class to generate maps (same pattern as create_test).
    """
    map_types = ['house', 'maze', 'random', 'warehouse']
    original_map_type = configs.map_type
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Binary colormap: white=empty (0), gray=obstacle (1)
    color_map = np.array([[255, 255, 255], [190, 190, 190]])
    
    try:
        for idx, map_type in enumerate(map_types):
            configs.map_type = map_type
            # Don't use fix_density for 'random' type - its generator uses triangular
            # distribution that requires a range, not a fixed value
            fix_dens = density if map_type != 'random' else None
            env = Environment(fix_density=fix_dens, num_agents=1, map_length=map_length)
            
            # Convert map to uint8 for color indexing (0=empty, 1=obstacle)
            # Some generators use -1 for obstacles, others use 1
            map_img = (env.map != 0).astype(np.uint8)
            
            axes[idx].imshow(color_map[map_img])
            axes[idx].set_title(f'{map_type.capitalize()} ({map_length}x{map_length})')
            axes[idx].axis('off')
    finally:
        configs.map_type = original_map_type
    
    plt.tight_layout()
    plt.savefig(f'random_maps_{map_length}x{map_length}_{density}density.png')
    plt.show()


def plot_test_cases(test_map_type: str, test_case: tuple, num_cases: int = 5):
    """
    Plot the first N maps from a saved test case .pth file.
    
    Args:
        test_map_type: Map type (e.g., 'house', 'maze', 'random', 'warehouse')
        test_case: Tuple of (length, num_agents, density)
        num_cases: Number of test cases to plot (default 5)
    """
    # Binary colormap: white=empty (0), gray=obstacle (1)
    color_map = np.array([[255, 255, 255], [190, 190, 190]])
    
    test_file = f"./test_set/{test_map_type}/{test_case[0]}length_{test_case[1]}agents_{test_case[2]}density_fair.pth"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    
    with open(test_file, "rb") as f:
        tests = pickle.load(f)
    
    num_to_plot = min(num_cases, len(tests))
    fig, axes = plt.subplots(1, num_to_plot, figsize=(4 * num_to_plot, 4))
    
    if num_to_plot == 1:
        axes = [axes]
    
    for idx in range(num_to_plot):
        env_map, agents_pos, goals_pos = tests[idx]
        
        # Convert map to display format (0=empty, 1=obstacle)
        map_img = (env_map != 0).astype(np.uint8)
        
        axes[idx].imshow(color_map[map_img])
        axes[idx].set_title(f'Case {idx}')
        axes[idx].axis('off')
    
    plt.suptitle(f'{test_map_type.capitalize()} - {test_case[0]}x{test_case[0]}, {test_case[1]} agents, {test_case[2]} density')
    plt.tight_layout()
    os.makedirs('test_set/images', exist_ok=True)
    plt.savefig(f'test_set/images/test_cases_{test_map_type}_{test_case[0]}length_{test_case[1]}agents.png')
    plt.show()


if __name__ == "__main__":

    # create test sets for all map types
    # for test_map_type in configs.test_map_types:
    #     print(f"\n========== Creating test sets for {test_map_type} ==========")
    #     create_test(
    #         test_map_type,
    #         test_env_settings=configs.test_env_settings,
    #         num_test_cases=configs.num_test_cases,
    #     )
    
    # for test_map_type in ['random']:
    #     create_fair_test(test_map_type, map_length=15, density=0.2, agent_counts=[3, 4, 5], num_test_cases=200)
    #     create_fair_test(test_map_type, map_length=30, density=0.3, agent_counts=[3, 6, 12, 24], num_test_cases=200)
    #     create_fair_test(test_map_type, map_length=45, density=0.4, agent_counts=[3, 12, 24, 48], num_test_cases=200)

    # test model across specified map types and models (legacy)
    # test_model(
    #     [
    #         ("house", 27000),
    #     ]
    # )

    # test model with custom parameters (model can differ from test cases)
    # test_model_custom(
    #     model_map_type='house',
    #     model_number=27000,
    #     test_map_type='random',
    #     test_cases=[(30, 3, 0.3), (30, 6, 0.3), (30, 12, 0.3)],
    #     use_fair=True,
    #     csv_filename='test_results_custom.csv'
    # )

    # test_model_custom(
    #     model_map_type='house',
    #     model_number=84000,
    #     test_map_type='house',
    #     test_cases=[(15, 3, 0.2), (15, 4, 0.2), (15, 5, 0.2), (30, 3, 0.3), (30, 6, 0.3), (30, 12, 0.3)],
    #     use_fair=True,
    #     csv_filename='house_cross_house_test.csv',
    #     make_animations=True,     
    #     animation_steps=100
    # )

    test_model_custom(
        model_map_type='house',
        model_number=84000,
        test_map_type='random',
        test_cases=[(15, 3, 0.2), (15, 4, 0.2), (15, 5, 0.2), (30, 3, 0.3), (30, 6, 0.3), (30, 12, 0.3)],
        use_fair=True,
        csv_filename='house_cross_random_test.csv',
        make_animations=True,     
        animation_steps=100
    )

    test_model_custom(
        model_map_type='house',
        model_number=84000,
        test_map_type='maze',
        test_cases=[(15, 3, 0.2), (15, 4, 0.2), (15, 5, 0.2), (30, 3, 0.3), (30, 6, 0.3), (30, 12, 0.3)],
        use_fair=True,
        csv_filename='house_cross_maze_test.csv',
        make_animations=True,     
        animation_steps=100
    )

    test_model_custom(
        model_map_type='house',
        model_number=84000,
        test_map_type='warehouse',
        test_cases=[(15, 3, 0.2), (15, 4, 0.2), (15, 5, 0.2), (30, 3, 0.3), (30, 6, 0.3), (30, 12, 0.3), (30, 24, 0.3), (45, 3, 0.4), (45, 12, 0.4), (45, 24, 0.4), (45, 48, 0.4)],
        use_fair=True,
        csv_filename='house_cross_warehouse_test.csv',
        make_animations=True,     
        animation_steps=100
    )



    # visualize result (example for one map type)
    # make_animation('house', 10000, (30, 24, 0.3), 100)

    # plot random 10x10 maps for each map type
    # plot_random_maps(map_length=25, density=0.3)

    # plot first 5 test cases from a test file
    # for test_map_type in ['random']:
    #     plot_test_cases(test_map_type, (15, 3, 0.2))
    #     plot_test_cases(test_map_type, (30, 3, 0.3))
    #     plot_test_cases(test_map_type, (45, 3, 0.4))



    # create fair test sets (same maps across agent counts)
    # create_fair_test('house', map_length=30, density=0.3, agent_counts=[3, 6, 12, 24], num_test_cases=200)
