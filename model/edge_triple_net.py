import torch
from torch import nn


class BaseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(16, 32, 3), nn.ReLU(), nn.Dropout2d(0.25))
        self.conv_2 = nn.Sequential(nn.Conv2d(32, 32, 3, dilation=2), nn.ReLU(), nn.Dropout2d(0.25))
        self.conv_3 = nn.Sequential(nn.Conv2d(32, 32, 3, dilation=2), nn.ReLU(), nn.Dropout2d(0.25))

    def forward(self, X):
        X = self.conv_1(X)
        X = self.conv_2(X)
        X = self.conv_3(X)
        return X


class EdgeripleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e_conv = nn.Sequential(nn.Conv2d(1, 16, 3), nn.ReLU(), nn.Dropout2d(0.25))
        self.i_conv = nn.Sequential(nn.Conv2d(1, 16, 3), nn.ReLU(), nn.Dropout2d(0.25))
        self.base_net = BaseNet()
        self.conv = nn.Sequential(nn.Conv2d(32, 32, 3, dilation=2), nn.ReLU(), nn.Dropout2d(0.25))
        self.pool = nn.AvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.line_1 = nn.Sequential(nn.Linear(32 * 6 * 6, 128), nn.ReLU(), nn.Dropout1d(0.25))
        self.line_2 = nn.Sequential(nn.Linear(128, 10), nn.Softmax(dim=1))

    def forward(self, X, Y):
        X = self.e_conv(X)
        Y = self.i_conv(Y)
        Z = X + Y
        X = self.base_net(X)
        Y = self.base_net(Y)
        Z = self.base_net(Z)
        X += Y + Z
        X = self.conv(X)
        X = self.pool(X)
        X = self.flatten(X)
        X = self.line_1(X)
        X = self.line_2(X)
        return X


# X = torch.rand((1, 1, 28, 28))
# Y = torch.rand((1, 1, 28, 28))
# net = EdgeripleNet()

# R = net(X, Y)
# print(R, torch.sum(R))
