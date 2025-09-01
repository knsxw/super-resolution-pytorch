import torch.nn as nn
import torch.nn.functional as F

class WorkingSRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(WorkingSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, stride=1, padding=4, bias=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, stride=1, padding=2, bias=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x