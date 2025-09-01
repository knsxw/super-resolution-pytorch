import torch.nn as nn
import torch.nn.functional as F

class WorkingESPCN(nn.Module):
    def __init__(self, scale_factor=2, num_channels=3):
        super(WorkingESPCN, self).__init__()
        self.scale_factor = scale_factor
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        # Conservative initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x