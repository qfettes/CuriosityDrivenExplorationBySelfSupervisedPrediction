import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from agents.BaseAgent import BaseAgent
from networks.networks import ActorCriticAtari 
from utils.RolloutStorage import RolloutStorage

from timeit import default_timer as timer

class Model(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='/tmp/gym'):
        super(Model, self).__init__(config=config, env=env, log_dir=log_dir)
        self.device = config.device
        self.conv_out = config.conv_out

        self.recurrent_policy = config.recurrent_policy_grad
        self.gru_size = config.gru_size

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.num_agents = config.num_agents
        self.value_loss_weight = config.value_loss_weight
        self.entropy_loss_weight = config.entropy_loss_weight
        self.rollout = config.rollout
        self.grad_norm_max = config.grad_norm_max

        self.static_policy = static_policy
        self.num_feats = env.observation_space.shape
        self.num_feats = (self.num_feats[0], *self.num_feats[1:])
        self.num_actions = env.action_space.n
        self.env = env

        self.declare_networks()
            
        if not self.recurrent_policy:
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.99, eps=1e-5)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=True)
        
        #move to correct device
        self.model = self.model.to(self.device)

        if self.static_policy:
            self.model.eval()
        else:
            self.model.train()

        self.rollouts = RolloutStorage(self.rollout, self.num_agents,
            self.num_feats, self.env.action_space, self.model.state_size,
            self.device, config.USE_GAE, config.gae_tau)

        self.value_losses = []
        self.entropy_losses = []
        self.policy_losses = []


    def declare_networks(self):
        self.model = ActorCriticAtari(self.num_feats, self.num_actions, self.conv_out, self.recurrent_policy, self.gru_size)
        

    def get_action(self, s, states, masks, deterministic=False):
        logits, values, states = self.model(s, states, masks)
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            actions = dist.probs.argmax(dim=1, keepdim=True)
        else:
            actions = dist.sample().view(-1, 1)

        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions)

        return values, actions, action_log_probs, states

    def evaluate_actions(self, s, actions, states, masks):
        logits, values, states = self.model(s, states, masks)

        dist = torch.distributions.Categorical(logits=logits)

        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = dist.entropy().mean()

        return values, action_log_probs, dist_entropy, states

    def get_values(self, s, states, masks):
        _, values, _ = self.model(s, states, masks)

        return values

    def compute_loss(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, states = self.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, 1),
            rollouts.states[0].view(-1, self.model.state_size),
            rollouts.masks[:-1].view(-1, 1))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        loss = action_loss + self.value_loss_weight * value_loss
        loss -= self.entropy_loss_weight * dist_entropy

        return loss, action_loss, value_loss, dist_entropy

    def update(self, rollout):
        loss, action_loss, value_loss, dist_entropy = self.compute_loss(rollout)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_max)
        self.optimizer.step()

        #self.save_loss(loss.item(), action_loss.item(), value_loss.item(), dist_entropy.item())

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    '''def save_loss(self, loss, policy_loss, value_loss, entropy_loss):
        super(Model, self).save_loss(loss)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropy_losses.append(entropy_loss)'''
