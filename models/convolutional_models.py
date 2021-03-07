import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_parts import *

#################################
# U-net for speech dereverberation
# base implementation on http://github.com//milesial/Pytorch-UNet
#################################

class UNetRev(nn.Module):
    def __init__(self, n_channels, bilinear=True, confine = True):
        super(UNetRev, self).__init__()
        self.confine = confine
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        output = self.tanh(x) if self.confine else x
        return x

#################################
# weights initialization for GAN model
#################################

def weights_init(m):
    """
    Initialise weights of the model.
    """
    if (type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif (type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


#################################
# GAN discriminator
# Use BCEWithLogitsLoss if you dont confine the output
#################################
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, (7, 7), 2, 1)

        self.conv2 = nn.Conv2d(64, 128, (7, 7), 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, (7, 7), 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, (7, 7), 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1024, (7, 7), 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc9 = nn.Linear(7168, 1)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1)

        x = torch.reshape(x, (x.shape[0], -1))
        x = F.leaky_relu(self.fc9(x), 0.1)
        # x = self.sig(x)
        return x