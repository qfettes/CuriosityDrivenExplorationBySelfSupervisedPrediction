import numpy as np
from collections import deque

import gym
from gym import spaces
from gym.spaces.box import Box

import gym_super_mario_bros
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

def make_env_a2c_smb(env_id, seed, rank, log_dir, stack_frames=4, action_repeat=6, reward_type='none', sticky=0.):
    def _thunk():
        
        env = gym_super_mario_bros.make(env_id)
        env.seed(seed + rank)

        env = BinarySpaceToDiscreteSpaceEnv(env, ACTIONS)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        env = ProcessFrameMario(env, reward_type=reward_type)
        env = smb_warp_frame(env)
        env = smb_scale_frame(env)
        env = smb_stack_and_repeat(env, stack_frames, action_repeat, sticky)
        env = WrapPyTorch(env)

        return env
    return _thunk

class ProcessFrameMario(gym.Wrapper):
    def __init__(self, env=None, reward_type=None):
        super(ProcessFrameMario, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 42, 42), dtype=np.uint8)
        self.prev_time = 400
        self.prev_stat = 0
        self.prev_score = 0
        self.prev_dist = 40
        self.max_dist = 3000.
        self.reward_type = reward_type

    def step(self, action): #pylint: disable=method-hidden
        obs, _, done, info = self.env.step(action)

        if self.reward_type == 'none':
            reward = 0.
        elif self.reward_type == 'sparse':
            reward = 0. 
            if done: 
                if info['flag_get']:
                    reward = 1.
        elif self.reward_type == 'dense':
            reward = float(info['x_pos']) - self.prev_dist
            reward /= self.max_dist
            self.prev_dist = float(info['x_pos'])
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
    def __init__(self, env):
        """Warp frames to 42x42 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 42
        self.height = 42
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
    def __init__(self, env, k, action_repeat, sticky):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.action_repeat = action_repeat
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

    def step(self, action): #pylint: disable=method-hidden
        is_sticky = np.random.rand()
        if is_sticky >= self.sticky or self.prev_action is None:
            self.prev_action = action
        ob, reward, done, info = self.env.step(self.prev_action)
        
        self.frames.append(ob)
        total_reward = reward
        
        for i in range(1, self.action_repeat):
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