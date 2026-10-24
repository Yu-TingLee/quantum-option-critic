import argparse
import time

import numpy as np
import torch

from modules.logger import Logger
from modules.utils import make_env


def run_random_baseline(args):
    env = make_env(args.env)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if hasattr(env, "action_space") and env.action_space is not None:
        env.action_space.seed(args.seed)

    logger = Logger(
        logdir=args.logdir, run_name=f"{time.strftime('%m%d-%H%M')}_{args.env}_Random"
    )

    steps = 0
    while steps < args.max_steps_total:
        rewards = 0
        option_lengths = {0: []}

        _obs, _ = env.reset(seed=args.seed if logger.n_eps == 0 else None)
        done = False
        ep_steps = 0

        while (not done) and ep_steps < args.max_steps_ep:
            action = env.action_space.sample()

            _next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            rewards += reward

            steps += 1
            ep_steps += 1
            logger.log_data(steps, None, None, 0.0, 1.0)

        # Log episode-level results
        option_lengths[0].append(ep_steps)
        print(f"Total Steps: {steps} | Episode: {logger.n_eps} | Return: {rewards:.2f}")
        logger.log_episode(steps, rewards, option_lengths, ep_steps, 1.0)

    print("\nRandom Baseline Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", default="runs")
    parser.add_argument("--max_steps_ep", type=int, default=18000)
    parser.add_argument("--max_steps_total", type=int, default=int(1e6))

    args = parser.parse_args()
    run_random_baseline(args)
