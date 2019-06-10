import numpy as np

import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from agents.BaseAgent import BaseAgent
from networks.networks import ActorCriticSMB
from utils.RolloutStorage import RolloutStorage

from timeit import default_timer as timer

class Model(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='/tmp/gym', tb_writer=None):
        super(Model, self).__init__(config=config, env=env, log_dir=log_dir, tb_writer=tb_writer)
        self.config = config
        self.static_policy = static_policy
        self.num_feats = env.observation_space.shape
        self.num_actions = env.action_space.n * len(config.adaptive_repeat)
        self.env = env

        self.declare_networks()
            
        if not self.config.recurrent_policy_grad:
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.config.LR, alpha=0.99, eps=1e-5)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LR, amsgrad=True)
        
        #move to correct device
        self.model = self.model.to(self.config.device)

        if self.static_policy:
            self.model.eval()
        else:
            self.model.train()

        self.config.rollouts = RolloutStorage(self.config.rollout, self.config.num_agents,
            self.num_feats, self.env.action_space, self.model.state_size,
            self.config.device, config.USE_GAE, config.gae_tau)

        self.value_losses = []
        self.entropy_losses = []
        self.policy_losses = []


    def declare_networks(self):
        self.model = ActorCriticSMB(self.num_feats, self.num_actions, self.config.recurrent_policy_grad, self.config.gru_size)
        

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

    def compute_loss(self, rollouts, next_value, frame):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        rollouts.compute_returns(next_value, self.config.GAMMA)

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

        loss = action_loss + self.config.value_loss_weight * value_loss
        loss -= self.config.entropy_loss_weight * dist_entropy

        self.tb_writer.add_scalar('Loss/Total Loss', loss.item(), frame)
        self.tb_writer.add_scalar('Loss/Policy Loss', action_loss.item(), frame)
        self.tb_writer.add_scalar('Loss/Value Loss', value_loss.item(), frame)
        self.tb_writer.add_scalar('Loss/Forward Dynamics Loss', 0., frame)
        self.tb_writer.add_scalar('Loss/Inverse Dynamics Loss', 0., frame)

        self.tb_writer.add_scalar('Policy/Entropy', dist_entropy.item(), frame)
        self.tb_writer.add_scalar('Policy/Value Estimate', values.detach().mean().item(), frame)

        self.tb_writer.add_scalar('Learning/Learning Rate', np.mean([param_group['lr'] for param_group in self.optimizer.param_groups]), frame)


        return loss, action_loss, value_loss, dist_entropy, 0.

    def update(self, rollout, next_value, frame):
        loss, action_loss, value_loss, dist_entropy, dynamics_loss = self.compute_loss(rollout, next_value, frame)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_max)
        self.optimizer.step()

        grad_norm = 0.
        grad_count = 0.
        for param in self.model.parameters():
            grad_norm += torch.sum(param.grad.abs()).item()
            grad_count += param.grad.numel()
        grad_norm /= grad_count
        #self.save_generic_stat(grad_norm, frame, 'grad_norms')
        self.tb_writer.add_scalar('Learning/Grad Norm', grad_norm, frame)

        return value_loss.item(), action_loss.item(), dist_entropy.item(), dynamics_loss

    def save_w(self, best=False):
      if best:
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best', 'model.dump'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'best', 'optim.dump'))
      
      torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'saved_model', 'model.dump'))
      torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'saved_model', 'optim.dump'))

    def load_w(self, best=False):
      if best:
        fname_model = os.path.join(self.log_dir, 'best', 'model.dump')
        fname_optim = os.path.join(self.log_dir, 'best', 'optim.dump')
      else:
        fname_model = os.path.join(self.log_dir, 'saved_model', 'model.dump')
        fname_optim = os.path.join(self.log_dir, 'saved_model', 'optim.dump')

      if os.path.isfile(fname_model):
        self.model.load_state_dict(torch.load(fname_model))

      if os.path.isfile(fname_optim):
        self.optimizer.load_state_dict(torch.load(fname_optim))
