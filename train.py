import os
import multiprocessing as mp
import random
import time

# Set multiprocessing start method to 'spawn' before any CUDA operations
# This is required when using CUDA with multiprocessing
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Start method already set, ignore
    pass

import torch
import numpy as np
import ray

from worker import GlobalBuffer, Learner, Actor
import configs

import wandb

os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def main(num_actors=configs.num_actors, log_interval=configs.log_interval):
    ray.init(local_mode=False)

    buffer = GlobalBuffer.remote()
    learner = Learner.remote(buffer)
    time.sleep(1)
    actors = [
        Actor.remote(i, 0.4 ** (1 + (i / (num_actors - 1)) * 7), learner, buffer)
        for i in range(num_actors)
    ]

    for actor in actors:
        actor.run.remote()

    while not ray.get(buffer.ready.remote()):
        time.sleep(5)
        learner_result = ray.get(learner.stats.remote(5))
        buffer_metrics = ray.get(buffer.stats.remote(5))
        # Handle both old format (just done) and new format (done, metrics)
        if isinstance(learner_result, tuple):
            done, learner_metrics = learner_result
        else:
            done = learner_result
            learner_metrics = {}
        
        # Log metrics to wandb
        if learner_metrics:
            wandb.log({f'train/{k}': v for k, v in learner_metrics.items()})
        if buffer_metrics:
            # Log buffer-specific metrics
            wandb.log({
                'buffer/buffer_size': buffer_metrics.get('buffer_size', 0),
                'buffer/buffer_update_speed': buffer_metrics.get('buffer_update_speed', 0)
            })
            # Log episode metrics
            wandb.log({
                'metrics/success_rate': buffer_metrics.get('success_rate', 0),
                'metrics/arrival_rate': buffer_metrics.get('arrival_rate', 0),
                'metrics/episode_length': buffer_metrics.get('episode_length', 0)
            })

    print("start training")
    buffer.run.remote()
    learner.run.remote()

    done = False
    while not done:
        time.sleep(log_interval)
        learner_result = ray.get(learner.stats.remote(log_interval))
        buffer_metrics = ray.get(buffer.stats.remote(log_interval))
        
        # Handle both old format (just done) and new format (done, metrics)
        if isinstance(learner_result, tuple):
            done, learner_metrics = learner_result
        else:
            done = learner_result
            learner_metrics = {}
        
        # Log metrics to wandb
        if learner_metrics:
            wandb.log({f'train/{k}': v for k, v in learner_metrics.items()})
        if buffer_metrics:
            # Log buffer-specific metrics
            wandb.log({
                'buffer/buffer_size': buffer_metrics.get('buffer_size', 0),
                'buffer/buffer_update_speed': buffer_metrics.get('buffer_update_speed', 0)
            })
            # Log episode metrics
            wandb.log({
                'metrics/success_rate': buffer_metrics.get('success_rate', 0),
                'metrics/arrival_rate': buffer_metrics.get('arrival_rate', 0),
                'metrics/episode_length': buffer_metrics.get('episode_length', 0)
            })
        
        print()


if __name__ == "__main__":
    wandb.init(
        entity="sigmamql",
        project="SIGMA-MQL",
        group="single_training",
        name=f"train_{configs.map_type}_{os.getpid()}",
    )
    main()
    wandb.finish()
