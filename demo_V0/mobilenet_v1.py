import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils
from torch.autograd import Variable


def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, ),
        nn.BatchNorm2d(oup),
        # 符合某一个特征的中间值，使劲儿放大；不符合的，一刀切掉。
        # inplace为True,将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出。
        # 省去了反复申请与释放内存的时间，直接代替原来的值
        nn.ReLU6(inplace=True)
    )
    
def conv_dw(inp, oup, stride = 1):
    return nn.Sequential(
        # part1

        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        # part2
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 416,416,3 -> 208,208,32
            conv_bn(3, 32, 2),
            # 208,208,32 -> 208,208,64
            conv_dw(32, 64, 1), 

            # 208,208,64 -> 104,104,128
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            # 104,104,128 -> 52,52,256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1), 
        )
            # 52,52,256 -> 26,26,512
        self.stage2 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1), 
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
            # 26,26,512 -> 13,13,1024
        self.stage3 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        # 定输入数据和输出数据feature map的大小，自适应算法能够自动帮助我们计算核的大小和每次移动的步长
        # target output size of 1*1
        self.avg = nn.AdaptiveAvgPool2d((1,1))


        # 当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量。
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)

        # guess (1,1,1024)???
        print(x.shape)
        # 这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成4列，那不确定的地方就可以写成-1
        x = x.view(-1, 1024)
        print(x.shape)
        x = self.fc(x)
        return x

def mobilenet_v1(pretrained=False, progress=True):
    model = MobileNetV1()
    if pretrained:
        print("mobilenet_v1 has no pretrained model")
    return model

if __name__ == "__main__":
    import torch
    from torchsummary import summary

    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mobilenet_v1().to(device)
    # summary:看到模型的参数个数, 模型占用的内容.
    summary(model, input_size=(3, 416, 416))
