import gym
gym.logger.set_level(40)

import argparse, pickle

import numpy as np

import torch
import torch.nn.functional as F

from IPython.display import clear_output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from timeit import default_timer as timer
from datetime import timedelta
import os
import glob

from utils.wrappers import make_env_a2c_smb
from utils.plot import plot_reward
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from utils.hyperparameters import PolicyConfig

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--algo', default='a2c',
					help='algorithm to use: a2c | ppo')
parser.add_argument('--lr', type=float, default=7e-4,
					help='learning rate (default: 7e-4)')
parser.add_argument('--gamma', type=float, default=0.99,
					help='discount factor for rewards (default: 0.99)')
parser.add_argument('--use-gae', action='store_true', default=False,
					help='use generalized advantage estimation')
parser.add_argument('--tau', type=float, default=0.95,
					help='gae parameter (default: 0.95)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
					help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
					help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=0.5,
					help='max norm of gradients (default: 0.5)')
parser.add_argument('--num-processes', type=int, default=16,
					help='how many training CPU processes to use (default: 16)')
parser.add_argument('--num-steps', type=int, default=5,
					help='number of forward steps in A2C (default: 5)')
parser.add_argument('--ppo-epoch', type=int, default=3,
					help='number of ppo epochs (default: 3)')
parser.add_argument('--num-mini-batch', type=int, default=32,
					help='number of batches for ppo (default: 32)')
parser.add_argument('--clip-param', type=float, default=0.1,
					help='ppo clip parameter (default: 0.1)')
parser.add_argument('--num-frames', type=int, default=1e7,
					help='number of frames to train (default: 1e7)')
parser.add_argument('--env-name', default='SuperMarioBros-1-1-v0',
					help='environment to train on (default: SuperMarioBros-1-1-v0)')
parser.add_argument('--recurrent-policy', action='store_true', default=False,
					help='use a recurrent policy')
parser.add_argument('--gru-size', type=int, default=256,
					help='number of output units for main gru (default: 256)')
parser.add_argument('--conv-out', type=int, default=32,
					help='number of output filters for final convolutional layer (default: 32)')
parser.add_argument('--reward-type', type=str, default='none',
                    choices=('none', 'sparse', 'dense'),
					help='Type of reward. Choices = {none, sparse, dense}. (default: none))')
parser.add_argument('--stack-frames', type=int, default=4,
					help='Number of frames to stack (default: 4)')
parser.add_argument('--action-repeat', type=int, default=6,
					help='Number of times to repeat action (default: 6)')
args = parser.parse_args()

if args.algo == 'a2c':
    from agents.A2C import Model
elif args.algo == 'ppo':
    from agents.PPO import Model
else:
    print("INVALID ALGORITHM. ABORT.")
    exit()
    
if args.recurrent_policy:
    model_architecture = 'recurrent/'
else:
    model_architecture = 'feedforward/'

config = PolicyConfig()
config.algo = args.algo
config.env_id = args.env_name

#preprocessing
config.stack_frames = args.stack_frames
config.action_repeat = args.action_repeat
config.reward_type = args.reward_type

#Recurrent control
config.recurrent_policy_grad = args.recurrent_policy
config.conv_out = args.conv_out
config.gru_size = args.gru_size

#Noisy Nets Control
config.USE_NOISY_NETS = False
config.SIGMA_INIT = 0.5

#ppo control
config.ppo_epoch = args.ppo_epoch
config.num_mini_batch = args.num_mini_batch
config.ppo_clip_param = args.clip_param

#a2c control
config.num_agents=args.num_processes
config.rollout=args.num_steps
config.USE_GAE = args.use_gae
config.gae_tau = args.tau

#misc agent variables
config.GAMMA=args.gamma
config.LR=args.lr
config.entropy_loss_weight=args.entropy_coef
config.value_loss_weight=args.value_loss_coef
config.grad_norm_max = args.max_grad_norm

config.MAX_FRAMES=int(args.num_frames / config.num_agents / config.rollout)

def save_config(config, base_dir):
    tmp_device = config.device
    config.device = None
    pickle.dump(config, open(os.path.join(base_dir, 'config.dump'), 'wb'))
    config.device = tmp_device

def train(config):
    base_dir = os.path.join('./results/', args.algo, model_architecture, config.env_id)
    try:
        os.makedirs(base_dir)
    except OSError:
        files = glob.glob(os.path.join(base_dir, '*.*'))
        for f in files:
            os.remove(f)
    
    log_dir = os.path.join(base_dir, 'logs/')
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
            
    model_dir = os.path.join(base_dir, 'saved_model/')
    try:
        os.makedirs(model_dir)
    except OSError:
        files = glob.glob(os.path.join(model_dir, '*.dump'))
        for f in files:
            os.remove(f)
    
    #save configuration for later reference
    save_config(config, base_dir)

    seed = np.random.randint(0, int(1e6))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    #torch.set_num_threads(1)

    envs = [make_env_a2c_smb(config.env_id, seed, i, log_dir, stack_frames=config.stack_frames, action_repeat=config.action_repeat, reward_type=config.reward_type) for i in range(config.num_agents)]
    envs = SubprocVecEnv(envs) if config.num_agents > 1 else DummyVecEnv(envs)

    model = Model(env=envs, config=config, log_dir=base_dir)

    obs = envs.reset()
    obs = torch.from_numpy(obs.astype(np.float32)).to(config.device)

    model.rollouts.observations[0].copy_(obs)
    
    episode_rewards = np.zeros(config.num_agents, dtype=np.float)
    final_rewards = np.zeros(config.num_agents, dtype=np.float)

    start=timer()

    print_step = 1
    print_threshold = 100
    
    for frame_idx in range(1, config.MAX_FRAMES+1):
        for step in range(config.rollout):
            
            with torch.no_grad():
                values, actions, action_log_prob, states = model.get_action(
                                                            model.rollouts.observations[step],
                                                            model.rollouts.states[step],
                                                            model.rollouts.masks[step])
            
            cpu_actions = actions.view(-1).cpu().numpy()
    
            obs, reward, done, _ = envs.step(cpu_actions)
            obs = torch.from_numpy(obs.astype(np.float32)).to(config.device)

            episode_rewards += reward
            masks = 1. - done.astype(np.float32)
            final_rewards *= masks
            final_rewards += (1. - masks) * episode_rewards
            episode_rewards *= masks

            rewards = torch.from_numpy(reward.astype(np.float32)).view(-1, 1).to(config.device)
            masks = torch.from_numpy(masks).to(config.device).view(-1, 1)

            obs *= masks.view(-1, 1, 1, 1)

            model.rollouts.insert(obs, states, actions.view(-1, 1), action_log_prob, values, rewards, masks)
            
        with torch.no_grad():
            next_value = model.get_values(model.rollouts.observations[-1],
                                model.rollouts.states[-1],
                                model.rollouts.masks[-1])

        model.rollouts.compute_returns(next_value, config.GAMMA)
            
        value_loss, action_loss, dist_entropy = model.update(model.rollouts)
        
        model.rollouts.after_update()

        if frame_idx % print_threshold == 0:
            #save_model
            if frame_idx % (print_threshold*10) == 0:
                model.save_w()
            
            #print
            end = timer()
            total_num_steps = (frame_idx + 1) * config.num_agents * config.rollout
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(frame_idx, total_num_steps,
                       int(total_num_steps / (end - start)),
                       np.mean(final_rewards),
                       np.median(final_rewards),
                       np.min(final_rewards),
                       np.max(final_rewards), dist_entropy,
                       value_loss, action_loss))
            #plot
            if frame_idx % (print_threshold * 1) == 0:
                try:
                    # Sometimes monitor doesn't properly flush the outputs
                    plot_reward(log_dir, config.env_id, 'A2C', config.MAX_FRAMES * config.num_agents * config.rollout, bin_size=10, smooth=1, time=timedelta(seconds=int(timer()-start)), ipynb=False)
                except IOError:
                    pass
    #final print
    try:
        # Sometimes monitor doesn't properly flush the outputs
        plot_reward(log_dir, config.env_id, 'A2C', config.MAX_FRAMES * config.num_agents * config.rollout, bin_size=10, smooth=1, time=timedelta(seconds=int(timer()-start)), ipynb=False)
    except IOError:
        pass
    model.save_w()
    envs.close()
    
if __name__=='__main__':
    train(config)
