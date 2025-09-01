import torch
import torch.nn as nn

class MinimalSR(nn.Module):
    def __init__(self):
        super(MinimalSR, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias.data)
        self.conv.weight.data[:, :, 1, 1] = torch.eye(3)

    def forward(self, x):
        return self.conv(x) + x
