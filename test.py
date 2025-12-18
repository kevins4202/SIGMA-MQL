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
    Test models on specified map types
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


if __name__ == "__main__":

    # create test sets for all map types
    # for test_map_type in configs.test_map_types:
    #     print(f"\n========== Creating test sets for {test_map_type} ==========")
    #     create_test(
    #         test_map_type,
    #         test_env_settings=configs.test_env_settings,
    #         num_test_cases=configs.num_test_cases,
    #     )

    # test model across specified map types and models
    test_model(
        [
            ("house", 27000),
        ]
    )

    # visualize result (example for one map type)
    # make_animation('house', 10000, (30, 24, 0.3), 100)
