from torch import nn


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True,
                 pool: nn.Module = None, activation: nn.Module = None, bn=True, dim=2):
        """
        Convolution Layer with pool or activation
        :param in_channels: in_channels of Convolution
        :param out_channels: in_channels of Convolution
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param pool:
        :param activation:
        :param bn:
        """
        super().__init__()
        if dim == 1:
            conv_fn = lambda: nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        elif dim == 3:
            conv_fn = lambda: nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        else:
            conv_fn = lambda: nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        model = [
            conv_fn()
        ]

        if pool is not None:
            model.append(pool)

        if not bias and bn:
            model.append(nn.BatchNorm2d(out_channels))

        if activation is not None:
            model.append(activation)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim=256, mid_dim=256, out_dim=256, bn=False):
        super(ResidualBlock, self).__init__()
        # Residual Block
        self.residual_block = nn.Sequential(
            ConvModule(in_dim, mid_dim, kernel_size=3, padding=1, bias=not bn, activation=nn.ReLU()),
            ConvModule(mid_dim, out_dim, kernel_size=3, padding=1, bias=not bn),
        )

        self.relu = nn.ReLU()

    def forward(self, x):  # x is (B, patches, in_dim)
        out = self.residual_block(x)  # F(x)
        out = out + x  # F(x) + x
        return self.relu(out)


class BottleNeck(nn.Module):
    def __init__(self, in_dim=256, mid_dim=64, out_dim=256, bn=True):
        super(BottleNeck, self).__init__()
        self.bottleneck = nn.Sequential(
            ConvModule(in_channels=in_dim, out_channels=mid_dim, kernel_size=1, bias=not bn, activation=nn.ReLU()),
            ConvModule(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, padding=1, bias=not bn, activation=nn.ReLU()),
            ConvModule(in_channels=mid_dim, out_channels=out_dim, kernel_size=1, bias=not bn),
        )

        self.relu = nn.ReLU()

    def forward(self, x):  # x is (B, patches, in_dim)
        fx = self.bottleneck(x)  # F(x)
        out = fx + x  # F(x) + x
        return self.relu(out)
