import torch.nn as nn

from torchvision import models

class FaceIdentification(nn.Module):
    def __init__(self, num_classes):
        super(FaceIdentification, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        out = self.fc(features)
        return out

    def extract_features(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features