import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Must match the transform used while training the Siamese network
# (100x100, grayscale -> single channel, ToTensor)
trnscm = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])


class Siamese(torch.nn.Module):
    """Same architecture as SiameseNetwork trained in the Hackathon4a notebook."""

    def __init__(self):
        super(Siamese, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5),
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# Fallback class list if class_mapping.json (saved by the Stage 3 notebook
# cells) is not found alongside the model files at deployment time.
classes = ['person1', 'person2', 'person3', 'person4', 'person5', 'person6', 'person7']
