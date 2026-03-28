from copy import deepcopy
from pathlib import Path
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

from modules.option_critic import OptionCriticFeatures
from modules.experience_replay import ReplayBuffer
from modules.option_critic import critic_loss as critic_loss_fn
from modules.option_critic import actor_loss as actor_loss_fn
from utils import make_env, to_tensor, print_param, plot_circuits
from logger import Logger

import logging
logging.getLogger("pennylane").setLevel(logging.WARNING)

def run(args):
    env = make_env(args.env)

    device = torch.device('cpu')
    option_critic = OptionCriticFeatures(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        layer_F = args.layer_F,
        layer_H = args.layer_H,
        env_name = args.env,
        n_qubits = env.observation_space.shape[0],
        num_options=args.num_options,
        Qfeats=args.Qfeats,
        Qoption_value=args.Qoption_value,
        Qterm=args.Qterm,
        Qoption_policies=args.Qoption_policies,
        Qhead_affine=args.Qhead_affine,
        no_scaling=args.no_scaling,
        no_entanglement=args.no_entanglement,
        hidden_neuron=args.hidden_neuron,
    )
    
    print_param(option_critic)
    plot_circuits(option_critic, env.observation_space.shape, device, env_name=args.env)
    
    # Global seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # seed the env
    if hasattr(env, "action_space") and env.action_space is not None:
        env.action_space.seed(args.seed)
    if hasattr(env, "observation_space") and env.observation_space is not None:
        env.observation_space.seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    
    tags = ""
    if args.Qfeats: tags += "F"    # Features
    if args.Qoption_value and not args.Qhead_affine:  
        tags += "O"                # Option-Value Head
    elif args.Qoption_value and args.Qhead_affine:
        tags += "AO"               # Option-Value, Affine Transformed
    if args.Qterm:  tags += "T"    # Terminations
    if args.Qoption_policies:  tags += "P"  # Intra-Option Policies
    if args.no_scaling:       tags += "_fixLam"
    if args.no_entanglement:  tags += "_noEntangle"
    
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
    
    while steps < args.max_steps_total:
        rewards = 0
        option_lengths = {opt: [] for opt in range(args.num_options)}

        obs, _ = env.reset(seed=args.seed if logger.n_eps == 0 else None)

        state = option_critic.get_state(to_tensor(obs))
        greedy_option = option_critic.greedy_option(state)
        current_option = 0


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
                # log VQC gradient
                if steps % 100 == 0:
                    logger.log_gradients(steps, option_critic)
                optim.step()

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(to_tensor(next_obs))
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)

            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

            # Extract affine trans. parameters if they exist
            option_value_weight = None
            option_value_bias = None
            if args.Qhead_affine:
                # Detach from graph and convert to numpy array
                option_value_weight = option_critic.option_value.weight.detach().cpu().numpy()
                option_value_bias = option_critic.option_value.bias.detach().cpu().numpy()

            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon, 
                            option_value_weight=option_value_weight, option_value_bias=option_value_bias)
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
    parser.add_argument("--Qfeats", action="store_true", help="Use VQC as feature extractor")
    parser.add_argument("--Qoption_value", action="store_true", help="Use VQC as option-value function")
    parser.add_argument("--Qterm", action="store_true", help="Use VQC as termination head")
    parser.add_argument("--Qoption_policies", action="store_true", help="Use VQC as intra-option policies")
    parser.add_argument("--Qhead_affine", action="store_true", help="Use weight and bias in option-value head")
    parser.add_argument("--layer_F", type=int, default=6, help="Number of layers in feature extractor")
    parser.add_argument("--layer_H", type=int, default=1, help="Number of layers in downstream heads")

    # Classical feature extractor
    parser.add_argument("--hidden_neuron", type=int, default=8, help="Hidden neuron count in classical feature extractor")

    # Ablation flags
    parser.add_argument("--no_scaling", action="store_true", help="Fix lambda=1")
    parser.add_argument("--no_entanglement", action="store_true", help="Remove CNOT gates (rotation-only ansatz)")

    run(parser.parse_args())