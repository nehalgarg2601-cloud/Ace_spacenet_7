import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# BasicBlock (used in stages 2-4)
# --------------------------------------------------

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes):

        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        if inplanes != planes:
            self.downsample = nn.Conv2d(inplanes, planes, 1)
        else:
            self.downsample = None


    def forward(self, x):

        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


# --------------------------------------------------
# Bottleneck (used in Stage1)
# --------------------------------------------------

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes):

        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Conv2d(inplanes, planes * 4, 1) if inplanes != planes*4 else None


    def forward(self, x):

        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


# --------------------------------------------------
# HR Module 
# --------------------------------------------------

class HRModule(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.num_branches = len(channels)
        self.channels = channels

        self.branches = nn.ModuleList()

        for c in channels:
            block = nn.Sequential(
                BasicBlock(c, c),
                BasicBlock(c, c),
                BasicBlock(c, c),
                BasicBlock(c, c)
            )
            self.branches.append(block)

        #  Fusion layers
        self.fuse_layers = nn.ModuleList()

        for i in range(self.num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(self.num_branches):

                if i == j:
                    fuse_layer.append(nn.Identity())

                elif j > i:
                    # Upsample
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, bias=False),
                        nn.BatchNorm2d(channels[i])
                    ))

                else:
                    # Downsample
                    ops = []
                    in_ch = channels[j]
                    for k in range(i - j):
                        out_ch = channels[i] if k == i - j - 1 else in_ch
                        ops.append(nn.Sequential(
                            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU(inplace=True)
                        ))
                        in_ch = out_ch
                    fuse_layer.append(nn.Sequential(*ops))

            self.fuse_layers.append(fuse_layer)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, xs):

        # Process branches
        xs = [branch(x) for branch, x in zip(self.branches, xs)]

        fused = []

        for i in range(self.num_branches):
            y = 0
            for j in range(self.num_branches):

                x = xs[j]

                if i == j:
                    y = y + x

                elif j > i:
                    x = self.fuse_layers[i][j](x)
                    x = F.interpolate(x, size=xs[i].shape[2:], mode="bilinear", align_corners=False)
                    y = y + x

                else:
                    x = self.fuse_layers[i][j](x)
                    y = y + x

            fused.append(self.relu(y))

        return fused


# --------------------------------------------------
# HRNet-W48
# --------------------------------------------------

class HRNetW48(nn.Module):

    def __init__(self, in_channels=3, num_classes=2):

        super().__init__()

        # Stem

        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        # Stage1

        self.layer1 = nn.Sequential(
            Bottleneck(64, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )

        # Transition1 

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 48, 3, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 96, 3, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage2

        self.stage2 = HRModule([48, 96])

        # Transition2

        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(96, 192, 3, stride=2, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage3

        self.stage3 = nn.Sequential(
            HRModule([48, 96, 192]),
            HRModule([48, 96, 192]),
            HRModule([48, 96, 192]),
            HRModule([48, 96, 192])
        )

        # Transition3

        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(192, 384, 3, stride=2, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage4

        self.stage4 = nn.Sequential(
            HRModule([48, 96, 192, 384]),
            HRModule([48, 96, 192, 384]),
            HRModule([48, 96, 192, 384])
        )

        # Head

        self.head = nn.Conv2d(48+96+192+384, num_classes, 1)


    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.layer1(x)

        x_list = [t(x) for t in self.transition1]

        x_list = self.stage2(x_list)

        x_list.append(self.transition2[2](x_list[-1]))

        for m in self.stage3:
            x_list = m(x_list)

        x_list.append(self.transition3[3](x_list[-1]))

        for m in self.stage4:
            x_list = m(x_list)

        size = x_list[0].shape[2:]

        x0 = x_list[0]
        x1 = F.interpolate(x_list[1], size=size, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x_list[2], size=size, mode="bilinear", align_corners=False)
        x3 = F.interpolate(x_list[3], size=size, mode="bilinear", align_corners=False)

        x = torch.cat([x0, x1, x2, x3], dim=1)

        return self.head(x)