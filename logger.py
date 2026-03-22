import logging
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, logdir, run_name):
        self.log_name = os.path.join(logdir, run_name)
        self.start_time = time.time()
        self.n_eps = 0

        os.makedirs(self.log_name, exist_ok=True)
        self.writer = SummaryWriter(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.log_name, "logger.log")),
            ],
            datefmt="%Y/%m/%d %I:%M:%S %p",
        )

    def log_episode(self, steps, reward, option_lengths, ep_steps, epsilon):
        self.n_eps += 1

        logging.info(
            f"> ep {self.n_eps} done. total_steps={steps} | reward={reward} | "
            f"episode_steps={ep_steps}"
        )

        logging.info(f"episodic_rewards={reward}")
        for option, lens in option_lengths.items():
            avg_len = float(np.mean(lens)) if len(lens) > 0 else 0.0
            active = float(sum(lens) / ep_steps) if ep_steps > 0 else 0.0
            # logging.info(f"option_{option}_avg_length={avg_len}")
            # logging.info(f"option_{option}_active={active}")

        # tensorboard
        self.writer.add_scalar("episodic_rewards", reward, self.n_eps)
        self.writer.add_scalar("episodic_rewards_total_steps", reward, steps)
        for option, lens in option_lengths.items():
            self.writer.add_scalar(f"option_{option}_avg_length", np.mean(lens) if len(lens) > 0 else 0, self.n_eps)
            self.writer.add_scalar(f"option_{option}_active", (sum(lens) / ep_steps) if ep_steps > 0 else 0, self.n_eps)

    def log_data(self, step, actor_loss, critic_loss, entropy, epsilon, option_value_weight=None, option_value_bias=None):
        # tensorboard
        if actor_loss is not None:
            self.writer.add_scalar("actor_loss", actor_loss.item(), step)
        if critic_loss is not None:
            self.writer.add_scalar("critic_loss", critic_loss.item(), step)
        self.writer.add_scalar("policy_entropy", entropy, step)
        self.writer.add_scalar("epsilon", epsilon, step)
        if option_value_weight is not None:
            for i, val in enumerate(option_value_weight[0]):
                self.writer.add_scalar(f"Qoption_value_scaling_a/option_{i}", val, step)
        
        if option_value_bias is not None:
            for i, val in enumerate(option_value_bias[0]):
                self.writer.add_scalar(f"Qoption_value_b/option_{i}", val, step)
                
    def log_gradients(self, step, model):
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                clean_name = name.replace('.', '/')
                self.writer.add_scalar(f"Gradients_VQC/{clean_name}", param.grad.norm().item(), step)