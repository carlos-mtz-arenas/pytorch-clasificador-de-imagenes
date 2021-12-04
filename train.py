from torchvision import datasets
import torch.nn as nn
import torch
import torch.optim as optim

from train_utils import get_transformers, get_model, TrainDto, perform_training


transformers = get_transformers()

data_set = {
  'train': datasets.ImageFolder('./images/train', transformers['train']),
  'test': datasets.ImageFolder('./images/test', transformers['test'])
}

data_loaders = {
  'train': torch.utils.data.DataLoader(
    data_set['train'],
    batch_size=32,
    shuffle=True,
    num_workers=2
  ),
  'test': torch.utils.data.DataLoader(
    data_set['test'],
    batch_size=32,
    shuffle=False,
    num_workers=2
  )
}

# TODO: if you have cuda cores available, you can just switch it manually
device = "cpu"

model = get_model(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

dto = TrainDto(
  device=device,
  criterion=criterion,
  loaders=data_loaders,
  optimizer=optimizer,
  dataset=data_set,
  epochs=10
)

trained_model = perform_training(model, dto)

torch.save(trained_model.state_dict(), './models/dog-trainer.h5')