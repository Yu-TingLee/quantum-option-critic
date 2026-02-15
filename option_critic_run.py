from copy import deepcopy
import numpy as np
import argparse
import torch
import time

from modules.option_critic import OptionCriticFeatures
from modules.experience_replay import ReplayBuffer
from modules.option_critic import critic_loss as critic_loss_fn
from modules.option_critic import actor_loss as actor_loss_fn
from utils import make_env, to_tensor, plot_circuits, print_param
from logger import Logger

import logging
logging.getLogger("pennylane").setLevel(logging.WARNING)

def run(args):
    env = make_env(args.env)

    device = torch.device('cpu')
    
    option_critic = OptionCriticFeatures(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        Qfeats=args.Qfeats,
        Qhead=args.Qhead,
        Qterm=args.Qterm,
        Qoption=args.Qoption,
        Qhead_scaling=args.Qhead_scaling
    )
    
    print_param(option_critic)
    plot_circuits(option_critic, env.observation_space.shape, device)
    
    # Global seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # seed the env via reset(seed=...) on the first reset
    if hasattr(env, "action_space") and env.action_space is not None:
        env.action_space.seed(args.seed)
    if hasattr(env, "observation_space") and env.observation_space is not None:
        env.observation_space.seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    
    tags = ""
    if args.Qfeats: tags += "F"    # Features
    if args.Qhead and not args.Qhead_scaling:  
        tags += "O"                # Option-Value Head
    elif args.Qhead and args.Qhead_scaling:
        tags += "AO"               # Option-Value, Affine Transformed
    if args.Qterm:  tags += "T"    # Terminations
    if args.Qoption:  tags += "P"  # Intra-Option Policies
    
    if tags:
        run_config = f"Hybrid_{tags}" 
    else:
        run_config = "Classical"

    logger = Logger(
        logdir=args.logdir,
        run_name=f"{time.strftime('%m%d-%H%M')}_{args.env}_{run_config}{args.exp}"
    )
        
    option_critic_prime = deepcopy(option_critic)
    optim = torch.optim.Adam(option_critic.parameters(), lr=args.learning_rate)
    
    steps = 0
    # if args.switch_goal:
    #     print(f"Current goal {env.goal}")
    
    while steps < args.max_steps_total:
        rewards = 0
        option_lengths = {opt: [] for opt in range(args.num_options)}

        obs, _ = env.reset(seed=args.seed if logger.n_eps == 0 else None)

        state = option_critic.get_state(to_tensor(obs))
        greedy_option = option_critic.greedy_option(state)
        current_option = 0

        # Goal switching experiment
        # if args.switch_goal and logger.n_eps == 1000:
        #     torch.save(
        #         {'model_params': option_critic.state_dict(), 'goal_state': env.goal},
        #         f'models/option_critic_seed={args.seed}_1k'
        #     )
        #     env.switch_goal()
        #     print(f"New goal {env.goal}")

        # if args.switch_goal and logger.n_eps > 2000:
        #     torch.save(
        #         {'model_params': option_critic.state_dict(), 'goal_state': env.goal},
        #         f'models/option_critic_seed={args.seed}_2k'
        #     )
        #     break

        done = False
        ep_steps = 0
        option_termination = True
        curr_op_len = 0

        while (not done) and ep_steps < args.max_steps_ep:
            epsilon = option_critic.epsilon
            
            # Option Switch (Epsilon-Greedy over Options)
            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = (
                    np.random.choice(args.num_options) if np.random.rand() < epsilon 
                    else greedy_option
                )
                curr_op_len = 0
                
            # Get action from intra-option policy
            action, logp, entropy = option_critic.get_action(state, current_option)

            # Push to buffer
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward

            # Learning updates
            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(
                    obs, current_option, logp, entropy,
                    reward, done, next_obs,
                    option_critic, option_critic_prime, args
                )
                loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
                    loss = loss + critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(to_tensor(next_obs))
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)

            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

            # Extract scaling parameters if they exist
            q_scaling = None
            q_bias = None
            if args.Qhead_scaling and hasattr(option_critic.Q, 'scaling'):
                # Detach from graph and convert to numpy array
                q_scaling = option_critic.Q.scaling.detach().cpu().numpy()
                q_bias = option_critic.Q.bias.detach().cpu().numpy()

            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon, 
                            qhead_scaling=q_scaling, qhead_bias=q_bias)
        option_lengths[current_option].append(curr_op_len)
        print(f"Total Steps: {steps} | Episode: {logger.n_eps} | Return: {rewards:.2f} | Epsilon: {epsilon:.4f}")
        logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Configuration
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logdir', default='runs')
    parser.add_argument('--exp', type=str, default='')
    # parser.add_argument('--switch_goal', action="store_true")
    
    # Training Params
    parser.add_argument('--learning-rate', type=float, default=.0005)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-history', type=int, default=10000)
    parser.add_argument('--update-frequency', type=int, default=4)
    parser.add_argument('--freeze-interval', type=int, default=200)
    parser.add_argument('--max_steps_ep', type=int, default=18000)
    parser.add_argument('--max_steps_total', type=int, default=int(1e6))

    # Option-Critic Params
    parser.add_argument('--num-options', type=int, default=2)
    parser.add_argument('--termination-reg', type=float, default=0.01)
    parser.add_argument('--entropy-reg', type=float, default=0.01)
    
    # Quantum Flags
    parser.add_argument("--Qfeats", action="store_true", help="Use VQC as feature trunk")
    parser.add_argument("--Qhead", action="store_true", help="Use VQC as Q-head")
    parser.add_argument("--Qterm", action="store_true", help="Use VQC as termination head")
    parser.add_argument("--Qoption", action="store_true", help="Use VQC as intra-option policies")
    parser.add_argument("--Qhead_scaling", action="store_true", help="Use scaling and bias in Q-head")
    
    run(parser.parse_args())
