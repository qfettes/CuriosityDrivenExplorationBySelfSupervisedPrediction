import os, csv, random
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from agents.A2C import Model as A2C_Model
from networks.networks import ActorCriticSMB
from networks.special_units import IC_Features, IC_ForwardModel_Head, IC_InverseModel_Head
from utils.RolloutStorage import RolloutStorage

from timeit import default_timer as timer

class Model(A2C_Model):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='/tmp/gym', tb_writer=None):
        super(Model, self).__init__(config=config, env=env, log_dir=log_dir, tb_writer=tb_writer)

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


    def declare_networks(self):
        self.model = ActorCriticSMB(self.num_feats, self.num_actions, self.config.recurrent_policy_grad, self.config.gru_size)
        self.ICMfeaturizer = IC_Features(self.num_feats)
        self.ICMForwardModel = IC_ForwardModel_Head(self.ICMfeaturizer.feature_size(), self.num_actions, self.ICMfeaturizer.feature_size())
        self.ICMBackwardModel = IC_InverseModel_Head(self.ICMfeaturizer.feature_size()*2, self.num_actions)        

    def icm_get_features(self, s):
        return self.ICMfeaturizer(s)

    def icm_get_forward_outp(self, phi, actions):
        return self.ICMForwardModel(phi, actions)

    def icm_get_inverse_outp(self, phi, actions):
        num = phi.size(0) - self.config.num_agents
        cat_phi = torch.cat((phi[:num], phi[self.config.num_agents:]), dim=1)
        
        logits = self.ICMBackwardModel(cat_phi)

        return logits

    def compute_intrinsic_reward(self, rollout):
        obs_shape = rollout.observations.size()[2:]
        num_steps, num_processes, _ = rollout.rewards.size()

        minibatch_size = rollout.observations[:-1].view(-1, *obs_shape).size(0)//self.config.icm_minibatches
        all_intr_reward = torch.zeros(rollout.rewards.view(-1, 1).shape, device=self.config.device, dtype=torch.float)

        minibatches = list(range(self.config.icm_minibatches))
        random.shuffle(minibatches)
        for i in minibatches:
          start=i*(minibatch_size)
          end=start+minibatch_size
          
          #compute intrinsic reward
          with torch.no_grad():
            phi = self.icm_get_features(rollout.observations.view(-1, *obs_shape)[start:end+self.config.num_agents])
          
            icm_obs_pred = self.icm_get_forward_outp(phi[:-1*self.config.num_agents], rollout.actions.view(-1, 1)[start:end])
            obs_diff = icm_obs_pred - phi[self.config.num_agents:]
            intr_reward = obs_diff.pow(2).sqrt().sum(dim=1) * self.config.icm_prediction_beta
            
            all_intr_reward[start:end] = intr_reward.view(-1, 1)
        
        rollout.rewards += all_intr_reward.view(num_steps, num_processes, 1)
        rollout.rewards = torch.clamp(rollout.rewards, min=-1.0, max=1.0)

    def update_icm(self, rollout, frame):
        obs_shape = rollout.observations.size()[2:]
        action_shape = rollout.actions.size()[-1]
        num_steps, num_processes, _ = rollout.rewards.size()

        total_forward_loss = 0.
        total_inverse_loss = 0.

        minibatch_size = rollout.observations[:-1].view(-1, *obs_shape).size(0)//self.config.icm_minibatches
        all_intr_reward = torch.zeros(rollout.rewards.view(-1, 1).shape, device=self.config.device, dtype=torch.float)

        minibatches = list(range(self.config.icm_minibatches))
        random.shuffle(minibatches)
        for i in minibatches:
          #forward model loss
          start=i*(minibatch_size)
          end=start+minibatch_size

          phi = self.icm_get_features(rollout.observations.view(-1, *obs_shape)[start:end+self.config.num_agents])
          tmp = rollout.observations[1,0]==rollout.observations.view(-1, *obs_shape)[self.config.num_agents]
          
          icm_obs_pred = self.icm_get_forward_outp(phi[:-1*self.config.num_agents], rollout.actions.view(-1, 1)[start:end])
          
          #forward model loss
          obs_diff = icm_obs_pred - phi[self.config.num_agents:]
          forward_model_loss = 0.5 * obs_diff.pow(2).sum(dim=1).mean()
          forward_model_loss *= float(icm_obs_pred.size(1)) #lenFeatures=288. Factored out to make hyperparams not depend on it.

          #inverse model loss
          icm_action_logits = self.icm_get_inverse_outp(phi, rollout.actions.view(-1, 1)[start:end])
          m = nn.CrossEntropyLoss()
          inverse_model_loss = m(icm_action_logits, rollout.actions.view(-1)[start:end])

          #total loss
          forward_loss = (self.config.icm_loss_beta*forward_model_loss)
          inverse_loss = ((1.-self.config.icm_loss_beta)*inverse_model_loss)
          loss = (forward_loss+inverse_loss)/float(self.config.icm_minibatches)
          loss /= self.config.icm_lambda

          self.optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_max)
          self.optimizer.step()

          total_forward_loss += forward_loss.item()
          total_inverse_loss += inverse_loss.item()
          
        total_forward_loss /= float(self.config.icm_minibatches)
        total_forward_loss /= self.config.icm_lambda
        total_inverse_loss /= float(self.config.icm_minibatches)
        total_inverse_loss /= self.config.icm_lambda
        
        self.tb_writer.add_scalar('Loss/Forward Dynamics Loss', total_forward_loss, frame)
        self.tb_writer.add_scalar('Loss/Inverse Dynamics Loss', total_inverse_loss, frame)
        
        return total_forward_loss + total_inverse_loss

    def compute_loss(self, rollouts, next_value, frame):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        dynamics_loss = self.update_icm(rollouts, frame)
        self.compute_intrinsic_reward(rollouts)
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
        pg_loss = action_loss + self.config.value_loss_weight * value_loss

        loss = pg_loss - self.config.entropy_loss_weight * dist_entropy

        self.tb_writer.add_scalar('Loss/Total Loss', loss.item(), frame)
        self.tb_writer.add_scalar('Loss/Policy Loss', action_loss.item(), frame)
        self.tb_writer.add_scalar('Loss/Value Loss', value_loss.item(), frame)

        self.tb_writer.add_scalar('Policy/Entropy', dist_entropy.item(), frame)
        self.tb_writer.add_scalar('Policy/Value Estimate', values.detach().mean().item(), frame)

        self.tb_writer.add_scalar('Learning/Learning Rate', np.mean([param_group['lr'] for param_group in self.optimizer.param_groups]), frame)

        return loss, action_loss, value_loss, dist_entropy, dynamics_loss

    def save_w(self, best=False):
      if best:
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best', 'model.dump'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'best', 'optim.dump'))
        torch.save(self.ICMfeaturizer.state_dict(), os.path.join(self.log_dir, 'best', 'featurizer.dump'))
        torch.save(self.ICMBackwardModel.state_dict(), os.path.join(self.log_dir, 'best', 'backward.dump'))
        torch.save(self.ICMForwardModel.state_dict(), os.path.join(self.log_dir, 'best', 'forward.dump'))
      
      torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'saved_model', 'model.dump'))
      torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'saved_model', 'optim.dump'))
      torch.save(self.ICMfeaturizer.state_dict(), os.path.join(self.log_dir, 'saved_model', 'featurizer.dump'))
      torch.save(self.ICMBackwardModel.state_dict(), os.path.join(self.log_dir, 'saved_model', 'backward.dump'))
      torch.save(self.ICMForwardModel.state_dict(), os.path.join(self.log_dir, 'saved_model', 'forward.dump'))

    def load_w(self, best=False):
      if best:
        fname_model = os.path.join(self.log_dir, 'best', 'model.dump')
        fname_optim = os.path.join(self.log_dir, 'best', 'optim.dump')
        fname_featurizer = os.path.join(self.log_dir, 'best', 'featurizer.dump')
        fname_backward = os.path.join(self.log_dir, 'best', 'backward.dump')
        fname_forward = os.path.join(self.log_dir, 'best', 'forward.dump')
      else:
        fname_model = os.path.join(self.log_dir, 'saved_model', 'model.dump')
        fname_optim = os.path.join(self.log_dir, 'saved_model', 'optim.dump')
        fname_featurizer = os.path.join(self.log_dir, 'saved_model', 'featurizer.dump')
        fname_backward = os.path.join(self.log_dir, 'saved_model', 'backward.dump')
        fname_forward = os.path.join(self.log_dir, 'saved_model', 'forward.dump')

      if os.path.isfile(fname_model):
        self.model.load_state_dict(torch.load(fname_model))

      if os.path.isfile(fname_optim):
        self.optimizer.load_state_dict(torch.load(fname_optim))

      if os.path.isfile(fname_featurizer):
        self.ICMfeaturizer.load_state_dict(torch.load(fname_featurizer))

      if os.path.isfile(fname_backward):
        self.ICMBackwardModel.load_state_dict(torch.load(fname_backward))

      if os.path.isfile(fname_forward):
        self.ICMForwardModel.load_state_dict(torch.load(fname_forward))
