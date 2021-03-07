import math

import torch
import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


# bneck
class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

# 判断 输入通道数和升维通道数是否一致 一致则不需要升维
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(

                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),

                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),

                # Squeeze-and-Excite
                # attention
                SELayer(hidden_dim) if use_se else nn.Identity(),

                h_swish() if use_hs else nn.ReLU(inplace=True),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
#  残差边是否使用
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            #`   k,   t,   c, SE,HS,s
            # k： 深度可分离卷积核大小
            # t:  bneck结构每次所需的升维比例
            # c： 当前bneck输出的通道数
            # SE/HS activiation gunc
            # s: whether needs 高和宽的压缩
                # 208,208,16 -> 208,208,16
                [3,   1,  16, 0, 0, 1],

                # two bnecks
                # 208,208,16 -> 104,104,24
                [3,   4,  24, 0, 0, 2],
                [3,   3,  24, 0, 0, 1],

                #three bnecks
            ###【52,52,40】保存为有效特征层作为panet构建
                # 104,104,24 -> 52,52,40
                [5,   3,  40, 1, 0, 2],
                [5,   3,  40, 1, 0, 1],
                [5,   3,  40, 1, 0, 1],

                # 52,52,40 -> 26,26,80
                [3,   6,  80, 0, 1, 2],
                [3, 2.5,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],

            ###【26,26,112】保存为有效特征层作为panet构建
                # 26,26,80 -> 26,26,112
                [3,   6, 112, 1, 1, 1],
                [3,   6, 112, 1, 1, 1],

            ###【13,13,160】保存为有效特征层作为panet构建
                # 26,26,112 -> 13,13,160
                [5,   6, 160, 1, 1, 2],
                [5,   6, 160, 1, 1, 1],
                [5,   6, 160, 1, 1, 1]
        ]

        input_channel = _make_divisible(16 * width_mult, 8)
        # 416,416,3 -> 208,208,16
        # def conv_3x3_bn(inp, oup, stride):

        # the first conv at the beginning!!!!
        layers = [conv_3x3_bn(3, input_channel, 2)]

        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        ### nn.Sequential 输入要么事orderdict,要么事一系列的模型，遇到上述的list，必须用*号进行转化
        ### 传入实参的时候，加上*号，可以将列表中的元素拆成一个个的元素
        self.features = nn.Sequential(*layers)

        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenet_v3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load('./model_data/mobilenetv3-large-1cd25616.pth')
        model.load_state_dict(state_dict, strict=True)
    return model

