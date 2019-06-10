import torch
import math

class PolicyConfig(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_type = 'dense'

        #icm
        self.icm_loss_beta = 0.2
        self.icm_prediction_beta = 0.01
        self.icm_lambda = 0.1
        self.icm_minibatches = 32

        #noisy nets
        self.noisy_nets=False
        self.sigma_init=0.5

        #meta infor
        self.algo = None
        self.env_id = None

        #POLICY GRADIENT EXCLUSIVE PARAMETERS
        self.recurrent_policy_grad = False
        self.stack_frames = 4
        self.adaptive_repeat = [4]
        self.gru_size = 512

        #PPO controls
        self.ppo_epoch = 3
        self.num_mini_batch = 32
        self.ppo_clip_param = 0.1

        #a2c control
        self.num_agents=8
        self.rollout=128
        self.USE_GAE = True
        self.gae_tau = 0.95

        #misc agent variables
        self.entropy_loss_weight=0.01
        self.value_loss_weight=1.0
        self.grad_norm_max = 0.5
        self.GAMMA=0.99
        self.LR=1e-4

        #Learning control variables
        self.MAX_FRAMES=100000

        #data logging parameters
        self.ACTION_SELECTION_COUNT_FREQUENCY = 1000