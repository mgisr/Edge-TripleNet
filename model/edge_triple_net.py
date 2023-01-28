from torch.optim import SGD
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class BaseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(16, 32, 3)
        self.conv_2 = nn.Conv2d(32, 32, 3, dilation=2)
        self.conv_3 = nn.Conv2d(32, 32, 3, dilation=2)

    def forward(self, X):
        X = F.dropout2d(F.relu(self.conv_1(X)), 0.25)
        X = F.dropout2d(F.relu(self.conv_2(X)), 0.25)
        X = F.dropout2d(F.relu(self.conv_3(X)), 0.25)
        return X


class EdgeripleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e_conv = nn.Conv2d(1, 16, 3)
        self.i_conv = nn.Conv2d(1, 16, 3)
        self.base_net = BaseNet()
        self.conv = nn.Conv2d(32, 32, 3, dilation=2)
        self.pool = nn.AvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.line_1 = nn.Linear(32 * 6 * 6, 128)
        self.line_2 = nn.Linear(128, 10)

    def forward(self, X, Y):
        X = F.dropout2d(F.relu(self.e_conv(X)), 0.25)
        Y = F.dropout2d(F.relu(self.i_conv(X)), 0.25)
        Z = X + Y
        X = self.base_net(X)
        Y = self.base_net(Y)
        Z = self.base_net(Z)
        X += Y + Z
        X = F.dropout2d(F.relu(self.conv(X)), 0.25)
        X = self.pool(X)
        X = self.flatten(X)
        X = F.dropout2d(F.relu(self.line_1(X)), 0.25)
        X = self.line_2(X)
        return X


class LightEdgeripleNet(pl.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_index):
        X, Y, z = batch
        z_hat = self.model(X, Y)
        loss = F.cross_entropy(z_hat, z)

        return loss

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), 1e-3)
        return optimizer
