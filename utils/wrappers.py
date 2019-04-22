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

def make_env_a2c_smb(env_id, seed, rank, log_dir, stack_frames=4, action_repeat=6, deterministic_repeat=False, reward_type='none'):
    def _thunk():
        
        env = gym_super_mario_bros.make(env_id)
        env.seed(seed + rank)

        env = BinarySpaceToDiscreteSpaceEnv(env, ACTIONS)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        env = ProcessFrameMario(env, reward_type=reward_type)
        env = smb_warp_frame(env)
        env = smb_scale_frame(env)
        env = smb_stack_and_repeat(env, stack_frames, action_repeat, deterministic_repeat)
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
    def __init__(self, env, k, action_repeat, deterministic_repeat):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.action_repeat = action_repeat
        self.deterministic_repeat = deterministic_repeat
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self): #pylint: disable=method-hidden
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action): #pylint: disable=method-hidden
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        total_reward = reward

        if self.deterministic_repeat:
            rep = self.action_repeat
        else:
            rep = np.random.randint(np.max((self.action_repeat-1, 1)), self.action_repeat+2)
        
        for i in range(1, rep):
            if not done:
                ob, reward, done, info = self.env.step(action)
                total_reward += reward
                self.frames.append(ob)
            else:
                self.frames.append(ob)
        return self._get_ob(), total_reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

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