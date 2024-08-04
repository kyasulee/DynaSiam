import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class SEModule(nn.Module):
    def __init__(self,
                 inplanes,
                 reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(inplanes, inplanes // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inplanes // reduction, inplanes, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class Bottle2neck(nn.Module):
    expansion = 2
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 baseWidth=26,
                 scale = 4,
                 ratio=8,
                 stype='normal'):
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0))) # 52
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

        self.se = SEModule(planes * self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        # 对于stype=='stage‘的时候不用加上前一小块的输出结果，而是直接 sp = spx[i]
        # 是因为输入输出的尺寸不一致（通道数不一样），所以没法加起来
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        # 在这里需要加pool的原因是因为，对于每一个layer的stage模块，它的stride是不确定，layer1的stride=1
        # layer2、3、4的stride=2，前三小块都经过了stride=2的3*3卷积，而第四小块是直接送到y中的，但它必须要pool一下
        # 不然尺寸和不能和前面三个小块对应上，无法完成最后的econcat操作
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# res2net50
class Encoder(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 basic_chan=26,
                 scale=4,
                 ratio=8):
        super(Encoder, self).__init__()

        self.inplanes = 64
        self.baseWidth = basic_chan
        self.scale = scale

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                nn.ReLU(inplace=True)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample,
                            stype='stage',
                            baseWidth=self.baseWidth,
                            scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes,
                                planes,
                                baseWidth=self.baseWidth,
                                scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = torch.squeeze(x, dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)


        return x1, x2, x3, x4


def res2net50():
    model = Encoder(Bottle2neck, [3, 4, 6, 3])
    return model

if __name__ == "__main__":
    input = torch.rand(1, 3, 224, 224)
    encoder = res2net50()
    print(encoder)
    out = encoder(input)
    print(out[0].shape, out[1].shape, out[2].shape, out[3].shape)
    # torch.Size([1, 128, 56, 56])
    # torch.Size([1, 256, 28, 28])
    # torch.Size([1, 512, 14, 14])
    # torch.Size([1, 1024, 7, 7])