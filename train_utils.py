from dataclasses import dataclass
from torchvision import models, transforms
import torch.nn as nn
import torch


def get_model(device):
  # load a pre-trained model
  # also, we configure the model from the 
  model = models.resnet50(pretrained=True).to(device)

  for param in model.parameters():
    param.requires_grad = False

  model.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 2)
  ).to(device)

  return model

def get_transformers():
  normalizer = transforms.Normalize(
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5]
  )

  transformers = {
    'train': transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalizer
    ]),
    'test': transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      normalizer
    ])
  }

  return transformers

@dataclass
class TrainDto:
  loaders: dict
  criterion: object
  optimizer: object
  dataset: dict
  epochs: int
  device: str

def model_train(model, phase: str, dto: TrainDto):
  running_loss = 0.0
  running_corrects = 0.0

  for x, y in dto.loaders[phase]:
    inputs = x.to(dto.device)
    labels = y.to(dto.device)

    outputs = model(inputs)
    loss = dto.criterion(outputs, labels)

    if phase == 'train':
      dto.optimizer.zero_grad()
      loss.backward()
      dto.optimizer.step()
    
    _, prediction = torch.max(outputs, 1)
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(prediction == labels.data)
  
  epoch_loss = running_loss / len(dto.dataset[phase])
  eopch_corrects = running_corrects.double() / len(dto.dataset[phase])

  print('{} loss: {:.4f}, correct: {:.4f}'.format(phase, epoch_loss, eopch_corrects))

def perform_training(model, dto: TrainDto):
  for epoch in range(dto.epochs):
    for phase in ['train', 'test']:
      model_train(model, phase, dto)
  return model