import torch.nn as nn
class DensityRatioEstNet(nn.Module):
    def __init__(self, ngf, image_size, num_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, ngf, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(ngf * image_size**2, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = x + self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
