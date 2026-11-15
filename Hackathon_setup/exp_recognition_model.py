import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

# Definition of classes as dictionary
classes = {0: 'ANGER', 1: 'DISGUST', 2: 'FEAR', 3: 'HAPPINESS', 4: 'NEUTRAL', 5: 'SADNESS', 6: 'SURPRISE'}


class facExpRec(torch.nn.Module):
    """Same architecture as trained in the Hackathon4b notebook."""

    def __init__(self, num_classes=7):
        super(facExpRec, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
            )

        self.features = nn.Sequential(
            conv_block(1, 64),     # 48 -> 24
            conv_block(64, 128),   # 24 -> 12
            conv_block(128, 256),  # 12 -> 6
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Sample Helper function
def rgb2gray(image):
    return image.convert('L')


# Must match the transform used while training (48x48 grayscale)
trnscm = transforms.Compose([rgb2gray, transforms.Resize((48, 48)), transforms.ToTensor()])
