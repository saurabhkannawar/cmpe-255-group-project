import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models, transforms
import torch
from PIL import Image

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_block(3, 64)
        self.conv2 = self.conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(self.conv_block(128, 128), self.conv_block(128, 128))

        self.conv3 = self.conv_block(128, 256, pool=True)
        self.conv4 = self.conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(self.conv_block(512, 512), self.conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 10))


    def conv_block(self, in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)


    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


def load_checkpoint(file_path):
    model=CustomModel()
    model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
    return model


def predict(image_path):
    resnet = load_checkpoint('files/cifar10-1.pth')
    transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
    ])

    img = Image.open(image_path)
    img = transform(img)
    batch_t = torch.unsqueeze(img, 0)
    resnet.eval()
    out = resnet(batch_t)
    with open('files/cifar_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
        
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 10
    _, indices = torch.sort(out, descending=True)
    out = [(classes[idx], prob[idx].item()) for idx in indices[0][:10]]
    
    return out


        
