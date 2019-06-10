import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_one_hot(labels, C=2):
    if len(labels.shape) < 2:
      labels = labels.unsqueeze(dim=1)
    one_hot = torch.zeros((labels.size(0), C), dtype=torch.float, device=device)
    target = one_hot.scatter_(1, labels, 1)
        
    return target

class IC_Features(nn.Module):
    def __init__(self, input_shape):
        super(IC_Features, self).__init__()
        
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        
    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(x.size(0), -1)

        return x

    def feature_size(self):
        return self.conv4(self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape))))).view(1, -1).size(1)

class IC_InverseModel_Head(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(IC_InverseModel_Head, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, self.num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits

class IC_ForwardModel_Head(nn.Module):
    def __init__(self, input_shape, num_actions, output_shape):
        super(IC_ForwardModel_Head, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.input_shape+num_actions, 256)
        self.fc2 = nn.Linear(256, output_shape)
        
    def forward(self, phi, a):
        a_onehot = make_one_hot(a, self.num_actions)
        x = torch.cat((phi.detach(), a_onehot), dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x