from turtle import hideturtle
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.fc_1 = nn.Linear(in_size, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_3 = nn.Linear(hidden_size, out_size)
    

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x


class LeeXRDNet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv_1 = nn.Conv1d(1, 64, 20)
        self.pool_1 = nn.MaxPool1d(3, stride=3)

        self.conv_2 = nn.Conv1d(64, 64, 15)
        self.pool_2 = nn.MaxPool1d(2, stride=3) # size 2 stride 3?! data loss?

        self.conv_3 = nn.Conv1d(64, 64, 10, stride=2)
        self.pool_3 = nn.MaxPool1d(1, stride=2) # data loss again?

        self.fc_1 = nn.Linear(7744, 2500) # Difference due to padding changes
        self.fc_2 = nn.Linear(2500, 1000)

        self.fc_out = nn.Linear(1000, out_size)


    def forward(self, x):
        x = self.pool_1(F.relu(self.conv_1(x)))
        x = self.pool_2(F.relu(self.conv_2(x)))
        x = self.pool_3(F.relu(self.conv_3(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_out(x)
        return x