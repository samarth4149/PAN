import torch.nn as nn
from torchvision import models


class SiameseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseNet, self).__init__()
        model = models.resnet18(pretrained=pretrained)
        self.featext = nn.Sequential(*(list(model.children())[:-1]))

    def get_features(self, x):
        x = self.featext(x)
        return x.view(len(x), -1)

    def forward(self, x1, x2, x3):
        x1, x2, x3 = self.featext(x1), self.featext(x2), self.featext(x3)
        dist1 = (x1 - x2).pow(2).sum(1)
        dist2 = (x1 - x3).pow(2).sum(1)
        return dist1, dist2

