import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv64Features(nn.Module):

    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        # izlazi se normiraju na jedinični vektor po dimneziji značajki.
        x = F.normalize(x, p=2, dim=1)
        # da bi provjerili da je norma = 1:
        # torch.norm(x[0].view(64))
        return x
