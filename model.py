import torch.nn as nn


class TextOrientationCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False), # 128, 256 -> 64, 128
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False), # 64, 128 -> 32, 64
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False), # 32, 64 -> 32, 64            
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(8),
            
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False), # 32, 64 -> 32, 64
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(8),
            
            nn.Flatten(),
            nn.Linear(8 * 32 * 64, 100),
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            
            nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze(1)
