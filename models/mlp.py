import torch.nn.functional as F
from torch import nn


class MLPNet (nn.Module):
    def __init__(self,class_size):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(1*2,1024)   
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, class_size)
        # self.fc3 = nn.Linear(512, 10)
        # self.dropout1 = nn.Dropout2d(0.01)
        # self.dropout2 = nn.Dropout2d(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.dropout1(x)
        # x = self.dropout2(x)
        return self.fc3(x)
