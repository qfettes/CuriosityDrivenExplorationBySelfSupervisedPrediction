import numpy as np
from collections import deque

import gym
from gym import spaces
from gym import wrappers
from gym.spaces.box import Box

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

import os, cv2

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, LazyFrames

#from utils.ICM_wrappers import *

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        #return observation.transpose(2, 0, 1)
        return np.array(observation).transpose(2, 1, 0)

def make_env_a2c_atari(env_id, seed, rank, log_dir):
    def _thunk():
        env = make_atari(env_id)
        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=True)

        obs_shape = env.observation_space.shape
        env = WrapPyTorch(env)

        return env
    return _thunk

def make_env_a2c_smb(env_id, seed, rank, log_dir, dim=42, stack_frames=4, adaptive_repeat=[6], reward_type='none', sticky=0., vid=False, base_dir=''):
    def _thunk():
        
        env = gym_super_mario_bros.make(env_id)
        env.seed(seed + rank)
        if vid:
            env = wrappers.Monitor(env, os.path.join(base_dir, 'video'), force=True)

        env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        env = ProcessFrameMario(env, reward_type=reward_type, dim=dim)
        env = smb_warp_frame(env, dim=dim)
        env = smb_scale_frame(env)
        env = smb_stack_and_repeat(env, stack_frames, adaptive_repeat, sticky)
        env = WrapPyTorch(env)

        return env
    return _thunk

class ProcessFrameMario(gym.Wrapper):
    def __init__(self, env=None, reward_type=None, dim=42):
        super(ProcessFrameMario, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, dim, dim), dtype=np.uint8)
        self.prev_time = 400
        self.prev_stat = 0
        self.prev_score = 0
        self.prev_dist = 40
        self.max_dist = 3156.
        self.reward_type = reward_type

        self.curr_score = 0.

    def step(self, action): #pylint: disable=method-hidden
        obs, rew, done, info = self.env.step(action)

        if self.reward_type == 'none':
            reward = 0.
        elif self.reward_type == 'sparse':
            reward = 0. 
            if done: 
                if info['flag_get']:
                    reward = 100. #note tailored for a gamma of 0.99
        elif self.reward_type == 'dense':
            reward = rew
            reward += (info["score"] - self.curr_score) / 40.
            self.curr_score = info["score"]
            if done:
                if info["flag_get"]:
                    reward += 50
                else:
                    reward -= 50
            reward = reward / 10.
        else: 
            return None

        return obs, reward, done, info

    def reset(self): #pylint: disable=method-hidden
        self.prev_time = 400
        self.prev_stat = 0
        self.prev_score = 0
        self.prev_dist = 40
        return self.env.reset()

    def change_level(self, level):
        self.env.change_level(level)

class smb_warp_frame(gym.ObservationWrapper):
    def __init__(self, env, dim=42):
        """Warp frames to dim x dim as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = dim
        self.height = dim
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class smb_scale_frame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class smb_stack_and_repeat(gym.Wrapper):
    def __init__(self, env, k, adaptive_repeat, sticky):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.adaptive_repeat = adaptive_repeat
        self.num_actions = env.action_space.n
        self.frames = deque([], maxlen=k)
        self.sticky = sticky
        self.prev_action = None
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self): #pylint: disable=method-hidden
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, a): #pylint: disable=method-hidden
        repeat_len = a // self.num_actions
        action = a % self.num_actions

        is_sticky = np.random.rand()
        if is_sticky >= self.sticky or self.prev_action is None:
            self.prev_action = action
        ob, reward, done, info = self.env.step(self.prev_action)
        
        self.frames.append(ob)
        total_reward = reward
        
        for i in range(1, self.adaptive_repeat[repeat_len]):
            if not done:
                is_sticky = np.random.rand()
                if is_sticky >= self.sticky or self.prev_action is None:
                    self.prev_action = action
                ob, reward, done, info = self.env.step(self.prev_action)

                total_reward += reward
                self.frames.append(ob)
            else:
                self.frames.append(ob)
        return self._get_ob(), total_reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


'''class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        if observation is not None:    # for future meta implementation
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

            unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
            unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

            return (observation - unbiased_mean) / (unbiased_std + 1e-8)
        else:
            return observation

    def change_level(self, level):
        self.env.change_level(level)'''

ACTIONS = [
    ['NOOP'],
    ['up'],
    ['down'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['B'],
    ['A', 'B']
]