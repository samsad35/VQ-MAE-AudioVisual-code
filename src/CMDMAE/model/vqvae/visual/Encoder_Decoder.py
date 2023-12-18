import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                     for _ in range(self.num_residual_layers)])

    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens // 2,
                                kernel_size=4,
                                stride=2, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                out_channels=num_hiddens,
                                kernel_size=4,
                                stride=2, padding=1)

        # self._conv_2_add = nn.Conv2d(in_channels=num_hiddens,
        #                              out_channels=num_hiddens,
        #                              kernel_size=4,
        #                              stride=2, padding=1)

        self.conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1, padding=1)

        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = F.relu(x)

        x = self.conv_2(x)
        x = F.relu(x)

        # x = self._conv_2_add(x)
        # x = F.relu(x)

        x = self.conv_3(x)
        return self.residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels=1):
        super(Decoder, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1, padding=1)

        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=num_residual_hiddens)

        self.conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                               out_channels=num_hiddens // 2,
                                               kernel_size=4,
                                               stride=2, padding=1)

        # self._conv_trans_2_add = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
        #                                             out_channels=num_hiddens // 2,
        #                                             kernel_size=4,
        #                                             stride=2, padding=1)

        self.conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                               out_channels=out_channels,
                                               kernel_size=4,
                                               stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv_1(inputs)

        x = self.residual_stack(x)

        x = self.conv_trans_1(x)
        x = F.relu(x)

        # x = self._conv_trans_2_add(x)
        # x = F.relu(x)

        return self.conv_trans_2(x)
