import torch
from torchvision import models
import torch.nn as nn

device = "cpu"

model = models.resnet50(pretrained=False).to(device)

model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)

model.load_state_dict(torch.load('./models/dog-trainer.h5'))

