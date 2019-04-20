#intrinsic curiosity a2c
#TODO: a2c nn with different architecture needed

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from agents.BaseAgent import BaseAgent
from networks.networks import ActorCriticSMB
from networks.special_units import IC_Features, IC_ForwardModel_Head, IC_InverseModel_Head
from utils.RolloutStorage import RolloutStorage

from timeit import default_timer as timer

class Model(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='/tmp/gym'):
        super(Model, self).__init__(config=config, env=env, log_dir=log_dir)
        self.config = config
        self.static_policy = static_policy
        self.num_feats = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.env = env

        self.declare_networks()
            
        if not self.config.recurrent_policy_grad:
            self.optimizer = optim.RMSprop(list(self.model.parameters())+list(self.ICMfeaturizer.parameters())+list(self.ICMForwardModel.parameters())+list(self.ICMBackwardModel.parameters()), lr=self.config.LR, alpha=0.99, eps=1e-5)
        else:
            self.optimizer = optim.Adam(list(self.model.parameters())+list(self.ICMfeaturizer.parameters())+list(self.ICMForwardModel.parameters())+list(self.ICMBackwardModel.parameters()), lr=self.config.LR)
        
        #move to correct device
        self.model = self.model.to(self.config.device)
        self.ICMfeaturizer = self.ICMfeaturizer.to(self.config.device)
        self.ICMForwardModel = self.ICMForwardModel.to(self.config.device)
        self.ICMBackwardModel = self.ICMBackwardModel.to(self.config.device)

        if self.static_policy:
            self.model.eval()
            self.ICMfeaturizer.eval()
            self.ICMForwardModel.eval()
            self.ICMBackwardModel.eval()
        else:
            self.model.train()
            self.ICMfeaturizer.train()
            self.ICMForwardModel.train()
            self.ICMBackwardModel.train()

        self.config.rollouts = RolloutStorage(self.config.rollout, self.config.num_agents,
            self.num_feats, self.env.action_space, self.model.state_size,
            self.config.device, config.USE_GAE, config.gae_tau)

        self.value_losses = []
        self.entropy_losses = []
        self.policy_losses = []

        #TODO: Left off here


    def declare_networks(self):
        self.model = ActorCriticSMB(self.num_feats, self.num_actions, self.config.recurrent_policy_grad, self.config.gru_size)
        self.ICMfeaturizer = IC_Features(self.num_feats)
        self.ICMForwardModel = IC_ForwardModel_Head(self.ICMfeaturizer.feature_size(), self.num_actions, self.ICMfeaturizer.feature_size())
        self.ICMBackwardModel = IC_InverseModel_Head(self.ICMfeaturizer.feature_size()*2, self.num_actions)
        

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

    def icm_get_features(self, s):
        return self.ICMfeaturizer(s)

    def icm_get_forward_outp(self, phi, actions):
        return self.ICMForwardModel(phi, actions)

    def icm_get_inverse_outp(self, phi, actions):
        num = phi.size(0) - self.config.num_agents
        cat_phi = torch.cat((phi[:num], phi[self.config.num_agents:]), dim=1)
        
        logits = self.ICMBackwardModel(cat_phi)

        return logits


    def get_values(self, s, states, masks):
        _, values, _ = self.model(s, states, masks)

        return values

    def compute_loss(self, rollouts, next_value=None):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        #icm calculations
        phi = self.icm_get_features(rollouts.observations.view(-1, *obs_shape))
        icm_obs_pred = self.icm_get_forward_outp(phi[:phi.size(0)-self.config.num_agents], rollouts.actions.view(-1, 1))
        icm_action_logits = self.icm_get_inverse_outp(phi, rollouts.actions.view(-1, 1))

        obs_diff = icm_obs_pred - phi[self.config.num_agents:]

        #add intrinsic reward
        if next_value is not None:
            with torch.no_grad():
                intr_reward = obs_diff.pow(2).sqrt().sum(dim=1)
                intr_reward = intr_reward.view(num_steps, num_processes, 1)
            self.config.rollouts.rewards += intr_reward.detach()
            self.config.rollouts.rewards = torch.clamp(self.config.rollouts.rewards, min=-1.0, max=1.0)
            self.config.rollouts.compute_returns(next_value, self.config.GAMMA)

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

        m = nn.CrossEntropyLoss()
        inverse_model_loss = m(icm_action_logits, rollouts.actions.view(-1))

        forward_model_loss = 0.5 * obs_diff.pow(2).mean()
        forward_model_loss *= float(icm_obs_pred.size(1)) #lenFeatures=288. Factored out to make hyperparams not depend on it.

        loss = action_loss + self.config.value_loss_weight * value_loss
        loss *= self.config.icm_lambda

        loss-= ((1.-self.config.icm_beta)*inverse_model_loss)
        loss-= (self.config.icm_beta*forward_model_loss)

        loss -= self.config.entropy_loss_weight * dist_entropy

        return loss, action_loss, value_loss, dist_entropy

    def update(self, rollout, next_value=None):
        loss, action_loss, value_loss, dist_entropy = self.compute_loss(rollout, next_value)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_max)
        self.optimizer.step()

        #self.save_loss(loss.item(), action_loss.item(), value_loss.item(), dist_entropy.item())

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    '''def save_loss(self, loss, policy_loss, value_loss, entropy_loss):
        super(Model, self).save_loss(loss)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropy_losses.append(entropy_loss)'''
