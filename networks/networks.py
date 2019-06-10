import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########Actor Critic Architectures#########
class ActorCriticSMB(nn.Module):
    def __init__(self, input_shape, num_actions, use_gru=False, gru_size=256, dropout=0.2):
        super(ActorCriticSMB, self).__init__()
        self.use_gru = use_gru
        self.gru_size = gru_size

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0),
                    nn.init.calculate_gain('relu'))

        #self.dropout2d = nn.Dropout2d(dropout)
        
        self.conv1 = init_(nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1))
        #c1_out = self.conv1(torch.zeros(1, *input_shape))
        #self.layernorm_conv1 = nn.LayerNorm(c1_out.size()[1:])

        self.conv2 = init_(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1))
        #c2_out = self.conv2(c1_out)
        #self.layernorm_conv2 = nn.LayerNorm(c2_out.size()[1:])

        self.conv3 = init_(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1))
        #c3_out = self.conv3(c2_out)
        #self.layernorm_conv3 = nn.LayerNorm(c3_out.size()[1:])

        self.conv4 = init_(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1))
        #c4_out = self.conv4(c3_out)
        #self.layernorm_conv4 = nn.LayerNorm(c4_out.size()[1:])

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
        #x = self.dropout2d(inputs)
        x = F.relu(self.conv1(inputs))
        #x = self.layernorm_conv1(x)

        #x = self.dropout2d(x)
        x = F.relu(self.conv2(x))
        #x = self.layernorm_conv2(x)

        #x = self.dropout2d(x)
        x = F.relu(self.conv3(x))
        #x = self.layernorm_conv3(x)

        #x = self.dropout2d(x)
        x = F.relu(self.conv4(x))
        #x = self.layernorm_conv4(x)

        return self.head_only(x, inputs, states, masks)

    def head_only(self, x, inputs, states, masks):
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