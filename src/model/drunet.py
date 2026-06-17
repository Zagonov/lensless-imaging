from torch import nn


def make_blocks(channels, n_blocks):
    return nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks):
        super().__init__()
        self.block = nn.Sequential(
            *[ResidualBlock(in_channels) for _ in range(n_blocks)],
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            *[ResidualBlock(out_channels) for _ in range(n_blocks)],
        )

    def forward(self, x):
        return self.block(x)


class DRUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        channels=(32, 64, 128, 256),
        n_blocks=4,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.head = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.down1 = DownBlock(c1, c2, n_blocks)
        self.down2 = DownBlock(c2, c3, n_blocks)
        self.down3 = DownBlock(c3, c4, n_blocks)
        self.body = make_blocks(c4, n_blocks)
        self.up3 = UpBlock(c4, c3, n_blocks)
        self.up2 = UpBlock(c3, c2, n_blocks)
        self.up1 = UpBlock(c2, c1, n_blocks)
        self.tail = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x

        x1 = self.head(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.body(x4)
        x = self.up3(x) + x3
        x = self.up2(x) + x2
        x = self.up1(x) + x1

        return self.tail(x) + residual
