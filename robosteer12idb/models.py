import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_hidden):
        super().__init__()
        self.fc_1 = nn.Linear(in_size, hidden_size)
        self.fc_mid = nn.ModuleList()
        for _ in range(num_hidden):
            self.fc_mid.append(nn.Linear(hidden_size, hidden_size))
        self.fc_3 = nn.Linear(hidden_size, out_size)
    

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        for layer in self.fc_mid:
            x = F.relu(layer(x))
        x = self.fc_3(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv_1 = nn.Conv1d(1, 64, 20)
        self.pool_1 = nn.MaxPool1d(3, stride=3)

        self.conv_2 = nn.Conv1d(64, 64, 15)
        self.pool_2 = nn.MaxPool1d(3, stride=3)

        self.fc_1 = nn.Linear(960, 1024)
        self.fc_2 = nn.Linear(1024, 1024)

        self.fc_out = nn.Linear(1024, out_size)


    def forward(self, x):
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=1)
        elif len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=0)
            x = torch.unsqueeze(x, dim=0)
        x = self.pool_1(F.relu(self.conv_1(x)))
        x = self.pool_2(F.relu(self.conv_2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_out(x)
        return x
