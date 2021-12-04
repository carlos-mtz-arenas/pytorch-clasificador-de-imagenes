import torch
from torchvision import models, transforms
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image

device = "cpu"

model = models.resnet50(pretrained=False).to(device)

model.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 2)).to(device)

model.load_state_dict(torch.load('./models/dog-trainer.h5'))

images = ['./images/validations/sitting.jpg',
          './images/validations/standing.jpg']

img_list = [Image.open(img) for img in images]

# normalizer and transofrmers are the same as the ones
# that we defined for the "test" phase during the training process
normalizer = transforms.Normalize(
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5]
)

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalizer
])

validation_batch = torch.stack(
    [transformer(img).to(device) for img in img_list])

prediction_tensor = model(validation_batch)

# transform the predictions to a probabilistic value
prediction_probabilistic = F.softmax(
    prediction_tensor, dim=1).cpu().data.numpy()

print(prediction_probabilistic)

for i, img in enumerate(img_list):
    print("{} {:.0f}% Sitting, {:.0f}% Standing".format(img.filename, 100*prediction_probabilistic[i, 0],
                                                          100*prediction_probabilistic[i, 1]))
