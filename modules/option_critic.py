import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
from math import exp
from utils import to_tensor
from modules.vqc import VQC, preprocess_obs

class OptionCriticFeatures(nn.Module):
    def __init__(self,
                in_features,
                num_actions,
                num_options = 2,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.05,
                eps_decay=20000,
                Qfeats=False,
                Qhead=False,
                Qterm=False,
                Qoption=False,
                Qhead_scaling=False):

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
        self.Qhead = Qhead
        self.Qterm = Qterm
        self.Qoption = Qoption
        
        if self.Qfeats:
            self.features = QuantumFeatureTrunk(layers = 6)
        else:
            self.features = nn.Sequential(
                nn.Linear(in_features, 8),
                nn.ReLU(),
                nn.Linear(8, 4)
            )
                    
        if self.Qhead:
            self.Q = QuantumHead(layers = 1, out_dim = num_options, Qhead_scaling=Qhead_scaling)
        else:
            self.Q = nn.Linear(4, num_options)

        if self.Qterm:
            self.terminations = QuantumHead(layers = 1, out_dim = num_options)
        else:
            self.terminations = nn.Linear(4, num_options)
            
        if self.Qoption:
            self.option_policies = nn.ModuleList([
                QuantumHead(layers = 1, out_dim = num_actions) for _ in range(num_options)
            ])
        else:
            self.option_policies = nn.ModuleList([
                nn.Linear(4, num_actions) for _ in range(num_options)
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

    def get_Q(self, state):
        return self.Q(state)
    
    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
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
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

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

    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()

    y = reward + (1 - done) * args.gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob  * next_Q_prime.max(dim=-1)[0])

    termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + args.termination_reg) * (1 - done)
    
    policy_loss = -logp * (y.detach() - Q[option]) - args.entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    return actor_loss

def critic_loss(model, model_prime, data_batch, args):
    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options   = torch.LongTensor(options).to(model.device)
    rewards   = torch.FloatTensor(rewards).to(model.device)
    masks     = 1 - torch.FloatTensor(dones).to(model.device)

    states = model.get_state(to_tensor(obs)).squeeze(0)
    Q      = model.get_Q(states)
    
    next_states_prime = model_prime.get_state(to_tensor(next_obs)).squeeze(0)
    next_Q_prime      = model_prime.get_Q(next_states_prime)

    next_states            = model.get_state(to_tensor(next_obs)).squeeze(0)
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]

    y = rewards + masks * args.gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob  * next_Q_prime.max(dim=-1)[0])

    td_err = (Q[batch_idx, options] - y.detach()).pow(2).mul(0.5).mean()
    return td_err

class QuantumFeatureTrunk(nn.Module):
    """
    obs -> preprocess_obs -> VQC -> (B,4)
    """
    def __init__(self, layers = 2, n_qubits = 4):
        super().__init__()
        self.device = torch.device("cpu")
        self.vqc = VQC(n_qubits, layers, self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        obs = obs.to(self.device, dtype=torch.float32)

        x = preprocess_obs(obs) # angles in [-pi, pi]
        qfeat = self.vqc(x).to(dtype=torch.float32) # (B,4)
        return qfeat

class QuantumHead(nn.Module):
    """
    states(in_dim) ->  VQC -> (B, out_dim)
    """
    def __init__(self, n_qubits = 4, layers = 1, out_dim = 2, Qhead_scaling=False):
        super().__init__()
        self.device = torch.device("cpu")
        self.vqc = VQC(n_qubits, layers, self.device)
        self.out_dim = out_dim
        self.Qhead_scaling = Qhead_scaling
        
        if self.Qhead_scaling:
            self.scaling = nn.Parameter(torch.ones(1, out_dim))
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = state.to(self.device, dtype=torch.float32)
        angles = 2.0 * torch.atan(x)
        qfeat = self.vqc(angles).to(dtype=torch.float32)
        out = qfeat[:, :self.out_dim]
        if self.Qhead_scaling:
            out = (out * self.scaling) + self.bias
        else:
            return out