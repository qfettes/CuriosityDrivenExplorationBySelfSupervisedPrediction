import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########Actor Critic Architectures#########
class ActorCriticSMB(nn.Module):
    def __init__(self, input_shape, num_actions, use_gru=False, gru_size=256):
        super(ActorCriticSMB, self).__init__()
        self.use_gru = use_gru
        self.gru_size = gru_size

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0),
                    nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1))
        self.conv2 = init_(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1))
        self.conv3 = init_(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1))
        self.conv4 = init_(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1))

        if use_gru:
            self.gru = nn.GRUCell(self.feature_size(input_shape), self.gru_size)

            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)
        else:
            init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0),
                        nn.init.calculate_gain('relu'))
            self.fc1 = init_(nn.Linear(self.feature_size(input_shape), self.gru_size))

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0), gain=1)

        self.critic_linear = init_(nn.Linear(self.gru_size, 1))

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.actor_linear = init_(nn.Linear(self.gru_size, num_actions))
        
        self.train()

    def forward(self, inputs, states, masks):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        if self.use_gru:
            x = x.view(x.size(0), -1)
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
                N = states.size(0)
                T = int(x.size(0) / N)

                # unflatten
                x = x.view(T, N, x.size(1))

                # Same deal with masks
                masks = masks.view(T, N, 1)

                outputs = []
                for i in range(T):
                    hx = states = self.gru(x[i], states*masks[i])     
                    outputs.append(hx)

                # assert len(outputs) == T
                # x is a (T, N, -1) tensor
                x = torch.stack(outputs, dim=0)
                # flatten
                x = x.view(T * N, -1)
        else:
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))

        value = self.critic_linear(x)
        logits = self.actor_linear(x)

        return logits, value, states

    def layer_init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        
        return module

    def feature_size(self, input_shape):
        return self.conv4(self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape))))).view(1, -1).size(1)
    
    @property
    def state_size(self):
        if self.use_gru:
            return self.gru_size
        else:
            return 1

class ActorCriticAtari(nn.Module):
    def __init__(self, input_shape, num_actions, use_gru=False, gru_size=512):
        super(ActorCriticAtari, self).__init__()
        self.use_gru = use_gru
        self.gru_size = gru_size

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0),
                    nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        if use_gru:
            self.fc1 = init_(nn.Linear(self.feature_size(input_shape), 512))
            self.gru = nn.GRUCell(512, self.gru_size)

            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)
        else:
            init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0),
                        nn.init.calculate_gain('relu'))
            self.fc1 = init_(nn.Linear(self.feature_size(input_shape), self.gru_size))

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0), gain=1)

        self.critic_linear = init_(nn.Linear(self.gru_size, 1))

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.actor_linear = init_(nn.Linear(self.gru_size, num_actions))
        
        self.train()

    def forward(self, inputs, states, masks):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        if self.use_gru:
            x = x.view(x.size(0), -1)
            if inputs.size(0) == states.size(0):
                x = F.relu(self.fc1(x))
                x = states = self.gru(x, states * masks)
            else:
                # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
                N = states.size(0)
                T = int(x.size(0) / N)

                # unflatten
                x = F.relu(self.fc1(x))
                x = x.view(T, N, x.size(1))

                # Same deal with masks
                masks = masks.view(T, N, 1)

                outputs = []
                for i in range(T):
                    hx = states = self.gru(x[i], states*masks[i])     
                    outputs.append(hx)

                # assert len(outputs) == T
                # x is a (T, N, -1) tensor
                x = torch.stack(outputs, dim=0)
                # flatten
                x = x.view(T * N, -1)
        else:
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))

        value = self.critic_linear(x)
        logits = self.actor_linear(x)

        return logits, value, states

    def layer_init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        
        return module

    def feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1, -1).size(1)
    
    @property
    def state_size(self):
        if self.use_gru:
            return self.gru_size
        else:
            return 1