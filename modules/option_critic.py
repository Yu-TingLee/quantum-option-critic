import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
from math import exp
from utils import to_tensor
from modules.vqc import VQC, Preprocessor

class OptionCriticFeatures(nn.Module):
    def __init__(self,
                in_features,
                num_actions,
                env_name,
                layer_F = 6,
                layer_H = 1,
                n_qubits = 4,
                num_options = 2,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.05,
                eps_decay=20000,
                Qfeats=False,
                Qoption_value=False,
                Qterm=False,
                Qoption_policies=False,
                Qhead_affine=False,
                no_scaling=False,
                no_entanglement=False):

        super(OptionCriticFeatures, self).__init__()

        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = torch.device("cpu")

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.num_steps = 0
        self.Qfeats = Qfeats
        self.Qoption_value = Qoption_value
        self.Qterm = Qterm
        self.Qoption_policies = Qoption_policies
        self.no_scaling = no_scaling
        self.no_entanglement = no_entanglement

        if self.Qfeats:
            self.features = QuantumFeatureTrunk(layers=layer_F, n_qubits=n_qubits, env_name=env_name,
                                                no_scaling=no_scaling, no_entanglement=no_entanglement)
        else:
            self.features = nn.Sequential(
                nn.Linear(in_features, 8),
                nn.ReLU(),
                nn.Linear(8, in_features)
            )
                    
        if self.Qoption_value:
            self.option_value = QuantumHead(layers=layer_H, out_dim=num_options, Qhead_affine=Qhead_affine, n_qubits=n_qubits,
                                            no_scaling=no_scaling, no_entanglement=no_entanglement)
        elif env_name in ['CartPole-v1']:
            self.option_value = nn.Linear(in_features, num_options)
        else:
            self.option_value = nn.Sequential(
                nn.Linear(in_features, 3),
                nn.ReLU(),
                nn.Linear(3, num_options)
            )

        if self.Qterm:
            self.terminations = QuantumHead(layers=layer_H, out_dim=num_options, n_qubits=n_qubits,
                                            no_scaling=no_scaling, no_entanglement=no_entanglement)
        elif env_name in ['CartPole-v1']:
            self.terminations = nn.Linear(in_features, num_options)
        else:
            self.terminations = nn.Sequential(
                nn.Linear(in_features, 3),
                nn.ReLU(),
                nn.Linear(3, num_options)
            )
            
        if self.Qoption_policies:
            self.option_policies = nn.ModuleList([
                QuantumHead(layers=layer_H, out_dim=num_actions, n_qubits=n_qubits,
                            no_scaling=no_scaling, no_entanglement=no_entanglement) for _ in range(num_options)
            ])
        elif env_name in ['CartPole-v1']:
            self.option_policies = nn.ModuleList([
                nn.Linear(in_features, num_actions) for _ in range(num_options)
            ])
        else:
            self.option_policies = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features, 3),
                    nn.ReLU(),
                    nn.Linear(3, num_actions)
                ) for _ in range(num_options)
            ])
            
        self.to(self.device)
        self.train()

    def get_state(self, obs):
        # Vector obs: (F,) -> (1,F)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        # Image obs: (C,H,W) -> (1,C,H,W)
        elif obs.dim() == 3:
            obs = obs.unsqueeze(0)

        obs = obs.to(self.device)
        return self.features(obs)

    def get_option_value(self, state):
        return self.option_value(state)
    
    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        option_value = self.get_option_value(state)
        next_option = option_value.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()
    
    def get_terminations(self, state):
        return self.terminations(state).sigmoid() 

    def get_action(self, state, option):
        logits = self.option_policies[option](state)
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy
    
    def greedy_option(self, state):
        option_value = self.option_value(state)
        return option_value.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if self.training:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = 0.05
        return eps

def actor_loss(obs, option, logp, entropy, reward, done, next_obs, model, model_prime, args):
    state = model.get_state(to_tensor(obs))
    next_state = model.get_state(to_tensor(next_obs))
    next_state_prime = model_prime.get_state(to_tensor(next_obs))

    option_term_prob = model.get_terminations(state)[:, option]
    next_option_term_prob = model.get_terminations(next_state)[:, option].detach()

    option_value = model.get_option_value(state).detach().squeeze()
    next_Q_prime = model_prime.get_option_value(next_state_prime).detach().squeeze()

    y = reward + (1 - done) * args.gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob  * next_Q_prime.max(dim=-1)[0])

    termination_loss = option_term_prob * (option_value[option].detach() - option_value.max(dim=-1)[0].detach() + args.termination_reg) * (1 - done)
    
    policy_loss = -logp * (y.detach() - option_value[option]) - args.entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    return actor_loss

def critic_loss(model, model_prime, data_batch, args):
    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options   = torch.LongTensor(options).to(model.device)
    rewards   = torch.FloatTensor(rewards).to(model.device)
    masks     = 1 - torch.FloatTensor(dones).to(model.device)

    states        = model.get_state(to_tensor(obs)).squeeze(0)
    option_value  = model.get_option_value(states)
    
    next_states_prime = model_prime.get_state(to_tensor(next_obs)).squeeze(0)
    next_Q_prime      = model_prime.get_option_value(next_states_prime)

    next_states            = model.get_state(to_tensor(next_obs)).squeeze(0)
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]

    y = rewards + masks * args.gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob  * next_Q_prime.max(dim=-1)[0])

    td_err = (option_value[batch_idx, options] - y.detach()).pow(2).mul(0.5).mean()
    return td_err

class QuantumFeatureTrunk(nn.Module):
    """
    obs -> preprocess_obs -> VQC -> (B,4)
    """
    def __init__(self, layers=6, n_qubits=4, env_name=None,
                 no_scaling=False, no_entanglement=False):
        super().__init__()
        self.device = torch.device("cpu")
        self.preprocessor = Preprocessor(env_name)
        self.vqc = VQC(n_qubits, layers, self.device,
                       no_scaling=no_scaling, no_entanglement=no_entanglement)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        obs = obs.to(self.device, dtype=torch.float32)
        x = self.preprocessor(obs)
            
        qfeat = self.vqc(x).to(dtype=torch.float32) # (B,4)
        return qfeat

class QuantumHead(nn.Module):
    """
    states(in_dim) ->  VQC -> (B, out_dim)
    """
    def __init__(self, n_qubits=4, layers=1, out_dim=2, Qhead_affine=False,
                 no_scaling=False, no_entanglement=False):
        super().__init__()
        self.device = torch.device("cpu")
        self.vqc = VQC(n_qubits, layers, self.device,
                       no_scaling=no_scaling, no_entanglement=no_entanglement)
        self.out_dim = out_dim
        self.Qhead_affine = Qhead_affine
        
        if self.Qhead_affine:
            self.weight = nn.Parameter(torch.ones(1, out_dim))
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = state.to(self.device, dtype=torch.float32)
        angles = 2.0 * torch.atan(x)
        qfeat = self.vqc(angles).to(dtype=torch.float32)
        out = qfeat[:, :self.out_dim]
        if self.Qhead_affine:
            out = (out * self.weight) + self.bias
            return out
        else:
            return out