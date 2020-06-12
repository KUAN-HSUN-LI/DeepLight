import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torchvision


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        modules = list(list(model.children())[:-1][0].children())[:-2]
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        out = self.features(x)
        return out


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channel, out_channel, 3, 2, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x, output_size):
        x = self.upsample(x, output_size=output_size)
        x = self.bn(x)
        x = self.relu6(x)

        return x


class DeepLight(nn.Module):
    def __init__(self):
        super(DeepLight, self).__init__()
        self.mobileNetV2 = MobileNetV2()
        self.bn1 = nn.BatchNorm2d(160)
        self.fc = nn.Linear(160*3*8, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.upsample1 = UpSample(16, 128)
        self.upsample2 = UpSample(128, 64)
        self.upsample3 = UpSample(64, 32)
        self.relu6 = nn.ReLU6(inplace=True)
        self.out = nn.Conv2d(32, 9, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, img):
        b, c, w, h = img.shape
        features = self.mobileNetV2(img)
        x = self.bn1(features)
        x = self.relu6(x)
        x = x.view(b, -1)
        x = self.fc(x)
        x = self.bn2(x)
        x = self.relu6(x)
        x = x.view([-1, 16, 4, 4])
        x = self.upsample1(x, output_size=(8, 8))
        x = self.upsample2(x, output_size=(16, 16))
        x = self.upsample3(x, output_size=(32, 32))
        out = self.tanh(self.out(x))

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, 4, 2, 1))
        self.elu = nn.ELU(inplace=True)
        self.fc = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.elu(x)
        x = self.conv3(x)
        x = self.elu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    n = DeepLight()
    x = torch.randn([2, 3, 81, 240])
    print(n(x).shape)

    d = Discriminator()
    x = torch.randn([1, 3, 32, 32])
    print(d(x))
