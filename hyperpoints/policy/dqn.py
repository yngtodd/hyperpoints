import torch.nn as nn
import torch.nn.functional as F


class Hyperparameters:

    def __init__(self, kernel1=5, kernel2=5, kernel3=5, dropout=0.0):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3
        self.dropout = dropout


class DQN(nn.Module):

    def __init__(self, num_actions=2, config=Hyperparameters()):
        super(DQN, self).__init__()
        self.hparams = config
        self.conv1 = nn.Conv2d(3, 16, kernel_size=self.hparams.kernel1, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=self.hparams.kernel2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self.hparams.kernel3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(448, 224)
        self.fc2 = nn.Linear(224, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(F.dropout(self.fc1(x, p=self.hparams.dropout)))
        return self.fc2(x)
