import torch
import torch.nn as nn
import torch.nn.init as init
import math
from math import sqrt
import common

class OldNet(nn.Module):
    def __init__(self, upscale_factor):
        super(OldNet, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d( 3, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3_1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3_2 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
        self.conv4_r = nn.Conv2d(1, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.conv4_g = nn.Conv2d(1, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.conv4_b = nn.Conv2d(1, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#        super(Conv2d, self).__init__(
#            in_channels, out_channels, kernel_size, stride, padding, dilation,
#            False, _pair(0), groups, bias)

        self._initialize_weights()

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))+y
        y = self.relu(self.conv3(y))+y
        y = self.relu(self.conv3_1(y)) + y
        x = self.conv3_2(y) + x
        r, g, b = torch.split(x, 1, dim=1)
        r = self.pixel_shuffle(self.conv4_r(r))
        g = self.pixel_shuffle(self.conv4_g(g))
        b = self.pixel_shuffle(self.conv4_b(b))
        out = torch.cat((r, g, b), 1)
        #print(out.size())
        return out

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3_1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3_2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4_r.weight)
        init.orthogonal_(self.conv4_g.weight)
        init.orthogonal_(self.conv4_b.weight)


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv_i1 = nn.Conv2d(in_channels=3, out_channels=3*(upscale_factor ** 2), kernel_size=13, stride=1, padding=6, bias=False)
#        self.conv_i1 = nn.Conv2d( 3, 3*(upscale_factor ** 2), (13, 13), (1, 1), (1, 1))
        self.conv_i2 = nn.Conv2d(in_channels=3*(upscale_factor ** 2), out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)
#        self.conv_i2 = nn.Conv2d(3*(upscale_factor ** 2), 64, (7, 7), (1, 1), (1, 1))
        self.conv_r1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv_r2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv_r3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv_o1 = nn.Conv2d(64, 3*(upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        xp = self.relu(self.conv_i1(x))
        y = self.relu(self.conv_i2(xp))
        y = self.relu(self.conv_r1(y))+y
        y = self.relu(self.conv_r2(y))+y
        y = self.relu(self.conv_r3(y))+y
        y = self.conv_o1(y)
        y = y + xp
        out = self.pixel_shuffle(y)
        return out

    def _initialize_weights(self):
        init.orthogonal_(self.conv_i1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv_i2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv_r1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv_r2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv_r3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv_o1.weight)


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class SRResNet(nn.Module):
    def __init__(self):
        super(SRResNet, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale2x(out)
        out = self.conv_output(out)
        return out


class _NetD(nn.Module):
    def __init__(self):
        super(_NetD, self).__init__()

        self.features = nn.Sequential(

            # input is (3) x 96 x 96
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 96 x 96
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 96 x 96
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 48 x 48
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 48 x 48
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 24 x 24
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 12 x 12
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 12 x 12
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):

        out = self.features(input)

        # state size. (512) x 6 x 6
        out = out.view(out.size(0), -1)

        # state size. (512 x 6 x 6)
        out = self.fc1(out)

        # state size. (1024)
        out = self.LeakyReLU(out)

        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.view(-1, 1).squeeze(1)


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual_layer(out)
        out = torch.add(out, residual)
        out = self.upscale2x(out)
        out = self.conv_output(out)
        return out


class SRCNN(nn.Module):
    def __init__(self, upscale_factor):
        super(SRCNN, self).__init__()

        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3 = nn.Conv2d(64, 3*(upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = (self.conv_3(x))
        out = self.pixel_shuffle(x)
        return out

    def _initialize_weights(self):
        init.orthogonal_(self.conv_1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv_2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv_3.weight)




class EDSR_Block(nn.Module):
    def __init__(self):
        super(EDSR_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self._initialize_weights()

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight)
        init.orthogonal_(self.conv2.weight)

class EDSR2(nn.Module):
    def __init__(self):
        super(EDSR, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.edsr_residual_block = self.make_layer(EDSR_Block, 32)

        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.edsr_residual_block(out)
        out = torch.add(out, residual)
        out = self.upscale2x(out)
        out = self.conv_output(out)
        return out


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class EDSR(nn.Module):
    def __init__(self):
        super(EDSR, self).__init__()
        conv = common.default_conv
        n_resblocks = 16
        n_feats = 64
        kernel_size = 3
        scale = 2
        n_colors = 3
        res_scale = 0.1
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(255)
        self.add_mean = common.MeanShift(255, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        #x = self.add_mean(x)

        return x
