import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class CBlock(nn.Module):
    def __init__(self):
        super(CBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.cbam = CBAM(64)

    def forward(self, x):
        res = self.conv1(x)
        res = res + x
        res = self.conv2(res)
        # res=self.calayer(res)
        # res=self.palayer(res)
        res = self.cbam(res)
        res += x
        return res


class CBlock1(nn.Module):
    def __init__(self):
        super(CBlock1, self).__init__()
        # self.conv1=conv(dim, dim, kernel_size, bias=True)
        # self.act1=nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1, bias=True)

        self.cbam = CBAM(256)

    def forward(self, x):
        res = self.conv1(x)
        res = res + x
        res = self.conv2(res)
        # res=self.calayer(res)
        # res=self.palayer(res)
        res = self.cbam(res)
        res += x
        return res


class ChannelAttention(nn.Module):
    def __init__(self, nc, norm_layer = nn.BatchNorm2d):
        super(ChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True)
        self.bn1 = norm_layer(nc)
        self.relu = nn.ReLU(nc)
        self.conv2 = nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True)
        self.bn2 = norm_layer(nc)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(nc, nc//2, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(nc//2, nc, 1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        se = self.gap(x1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return se*x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class AN(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, scale_factor=1):
        super(AN, self).__init__()
        filters = np.array([64, 128, 256, 512, 1024])
        filters = filters // scale_factor
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Att3 = ChannelAttention(nc=128)
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Att2 = ChannelAttention(nc=64)
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])

        self.Conv_1x1 = nn.Conv2d(filters[0], 2, kernel_size=1, stride=1, padding=0)


        self.cblock1 = CBlock1()
        self.cblock2 = CBlock1()
        self.cblock3 = CBlock1()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.cblock1(x3)
        x4 = self.cblock2(x4)
        x4 = self.cblock3(x4)

        d3 = self.Up3(x4)
        x2 = self.Att3(x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class MC(nn.Module):
    def __init__(self):
        super(MC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 64, 3, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(1, 64, 5, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, 64, 7, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 64, 1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(input)
        x3 = self.conv3(input)
        x4 = self.conv4(input)
        x5 = torch.cat([x1, x2, x3, x4], 1)
        x6 = self.conv5(x5)
        x7 = self.conv6(x6)
        out = self.conv7(x7)
        return out


class YCrCb(nn.Module):
    def __init__(self):
        super(YCrCb, self).__init__()
        self.y_weight = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).cuda()
        self.cb_weight = torch.tensor([-0.169, -0.331, 0.500]).view(1, 3, 1, 1).cuda()
        self.cr_weight = torch.tensor([0.500, -0.419, -0.081]).view(1, 3, 1, 1).cuda()

    def forward(self, img):
        y = F.conv2d(img, self.y_weight)
        cb = F.conv2d(img, self.cb_weight) + 0.5
        cr = F.conv2d(img, self.cr_weight) + 0.5
        ycrcb = torch.cat([y, cr, cb], dim=1)
        return ycrcb, y, torch.cat([cr, cb], dim=1)


class RGB(nn.Module):
    def __init__(self):
        super(RGB, self).__init__()
        self.rgb_weight = torch.tensor([1.0, 1.402, 1.772]).view(1, 3, 1, 1).cuda()

    def forward(self, ycrcb):
        y, cr, cb = ycrcb.split(1, dim=1)
        r = y + 1.402 * (cr - 0.5)
        g = y - 0.34414 * (cb - 0.5) - 0.71414 * (cr - 0.5)
        b = y + 1.772 * (cb - 0.5)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb
    

class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, 1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2, 3, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.mc = MC()
        # self.cbam = CBAM(64)
        self.cblock = CBlock()
        self.ycrcb = YCrCb()
        self.an = AN()
        self.rgb = RGB()

    def forward(self, input):
        x1, x2, x3 = self.ycrcb(input)
        x1 = self.conv3(x1)
        # x4 = self.cbam(x1)
        x4 = self.cblock(x1)
        x5 = self.mc(x2)
        x6 = torch.cat([x4, x5], 1)
        x6 = self.conv1(x6)
        out1 = self.conv2(x6)
        x3 = self.conv4(x3)
        x3 = self.an(x3)
        out2 = torch.cat([out1, x3], 1)
        out2 = self.rgb(out2)
        return out2


class fusion(nn.Module):
    def __init__(self):
        super(fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(10, 60, 1, padding=0, bias=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(60, 3, 7, padding=0, bias=True),
            nn.Tanh()
        )

    def forward(self, input):
        out = self.fusion(input)
        return out


class HSV(nn.Module):
    def __init__(self):
        super(HSV, self).__init__()

    def forward(self, input):
        # Assuming input is in range [0, 1]
        r, g, b = input.split(1, dim=1)

        max_c, _ = input.max(dim=1, keepdim=True)
        min_c, _ = input.min(dim=1, keepdim=True)
        diff = max_c - min_c

        h = torch.where(max_c == min_c, torch.zeros_like(max_c),
                        torch.where(max_c == r, 60.0 * (g - b) / diff + 360.0,
                        torch.where(max_c == g, 60.0 * (b - r) / diff + 120.0,
                        60.0 * (r - g) / diff + 240.0)))
        h = h % 360

        s = torch.where(max_c == 0, torch.zeros_like(max_c), diff / max_c)
        v = max_c

        # Stack h, s, v channels
        out = torch.cat([v, s], dim=1)
        return out


class ASM(nn.Module):
    def __init__(self):
        super(ASM, self).__init__()

    def forward(self, input1, input2):
        out = input1 * input2 - input2 + 1
        return out


class PF(nn.Module):
    def __init__(self):
        super(PF, self).__init__()
        self.pf1 = nn.Sequential(
            nn.Conv2d(6, 1, 1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.pf2 = nn.Sequential(
            nn.Conv2d(6, 1, 1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.pf3 = nn.Sequential(
            nn.Conv2d(6, 1, 1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.pf4 = nn.Sequential(
            nn.Conv2d(6, 1, 1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.upsample = F.upsample_nearest

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        shape_out = (height, width)

        x1 = F.avg_pool2d(input, 32)
        x1 = self.pf1(x1)
        x1 = self.upsample(x1, size=shape_out)

        x2 = F.avg_pool2d(input, 16)
        x2 = self.pf1(x2)
        x2 = self.upsample(x2, size=shape_out)

        x3 = F.avg_pool2d(input, 8)
        x3 = self.pf1(x3)
        x3 = self.upsample(x3, size=shape_out)

        x4 = F.avg_pool2d(input, 4)
        x4 = self.pf1(x4)
        x4 = self.upsample(x4, size=shape_out)

        out = torch.cat([input, x1, x2, x3, x4], 1)
        return out

class SPM(nn.Module):
    def __init__(self):
        super(SPM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, bias=True, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, bias=True, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, bias=True, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, bias=True, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, bias=True, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(67, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv17 = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.HSV = HSV()
        self.ASM = ASM()

    def forward(self,x):
        x1 = self.HSV(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)

        x2 = self.conv3(x1)
        x2 = self.conv4(x2)
        x2 = self.conv5(x2)

        x3 = self.conv6(x2)
        x3 = self.conv7(x3)
        x3 = self.conv8(x3)
        x3 = self.conv9(x3)
        x3 = torch.cat([x3, x1], 1)

        x4 = self.conv10(x3)
        x4 = self.conv11(x4)
        x4 = self.conv12(x4)
        x4 = torch.cat([x4, x2], 1)

        x5 = self.conv13(x4)
        x5 = self.conv14(x5)
        x5 = torch.cat([x5, x], 1)

        x6 = self.conv15(x5)
        x6 = self.conv16(x6)
        x6 = self.conv17(x6)

        out = self.ASM(x, x6)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.spm = SPM()
        self.colornet = ColorNet()
        self.pf = PF()
        self.fusion = fusion()

    def forward(self, x):
        out1 = self.spm(x)

        out2 = self.colornet(x)

        x1 = torch.cat([out1, out2], 1)
        x1 = self.pf(x1)
        out3 = self.fusion(x1)
        return out1, out2, out3


class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.BatchNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1))
'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)  # ,

            # nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
'''
if __name__ == '__main__':
    net = Net().cuda()
    input_tensor = torch.Tensor(np.random.random((1, 3, 1500, 1000))).cuda()
    start = time.time()
    out = net(input_tensor)
    end = time.time()
    print('Process Time: %f' % (end - start))
    print(input_tensor.shape)