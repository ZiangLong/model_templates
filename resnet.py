import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    exp = 1
    
    def __init__(self, in_ch, out_ch, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != self.exp * out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, self.exp * out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(self.exp * out_ch)
            )
            
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        z = self.shortcut(x)
        out = F.relu(y + z)
        return out
        
class Bottleneck(nn.Module):
    exp = 4
    def __init__(self, in_ch, out_ch, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, self.exp * out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.exp * out_ch)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != self.exp * out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, self.exp * out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(self.exp * out_ch)
            )
            
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        z = self.shortcut(x)
        out = F.relu(y + z)
        return out

class ResNet(nn.Module):
    def __init__(self, Basis, num_channels, num_classes, Ch_list, Len_list, pic_size = 32):
        super(ResNet, self).__init__()
        prev, cur = num_channels, Ch_list[0]
        self.conv1 = nn.Conv2d(prev, cur, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cur)
        prev = cur
        self.all_block = []
        for i in range(len(Ch_list)):
            cur_stage = []
            cur = Ch_list[i]
            for j in range(len(Len_list)):
                s = 2 if j == 0 and i != 0 else 1   
                cur_stage.append(Basis(prev, cur, s))
                prev = cur * Basis.exp
            self.all_block.append(cur_stage)
        self.linear = nn.Linear(prev, num_classes)
        self.pool_ker = pic_size >> (len(Ch_list) - 1)
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.all_block)):
            for j in range(len(self.all_block[i])):
                y = self.all_block[i][j](y)
        y = F.avg_pool2d(y, self.pool_ker)
        y = y.view(y.size(0), -1)
        y = self.linear(y)
        return y

class ResNet_Cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.in_ch = 64
        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.stage1 = self._make_layer(block, 64 , num_blocks[0], 1)
        self.stage2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.stage3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.stage4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.linear = nn.Linear(512 * block.exp, num_classes)
        
    def _make_layer(self, block, out_ch, num_b, stride):
        layers = []
        for s in [stride] + [1] * (num_b - 1):
            layers.append(block(self.in_ch, out_ch, s))
            self.in_ch = out_ch * block.exp
        return nn.Sequential(*layers)
        
    def forward(self, x):
        #   N * 3 * 32 * 32
        y = F.relu(self.bn1(self.conv1(x)))
        #   N * 64 * 32 * 32
        y = self.stage1(y)
        #   N * 64 * 32 * 32
        y = self.stage2(y)
        #   N * 128 * 16 * 16
        y = self.stage3(y)
        #   N * 256 * 8 * 8
        y = self.stage4(y)
        #   N * 512 * 4 * 4
        y = F.avg_pool2d(y, 4)
        #   N * 512 * 1 * 1
        y = y.view(y.size(0), -1)
        #   N * 512
        y = self.linear(y)
        #   N * 10
        return y
        
def ResNet18():
    return ResNetCifar(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNetCifar(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNetCifar(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNetCifar(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNetCifar(Bottleneck, [3,8,36,3])

'''
def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
    net = ResNet34()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
    net = ResNet50()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
    net = ResNet101()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
    net = ResNet152()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

test()
'''
