import torch
from torchvision.models import densenet121
from densenet1d import DenseNet1D

# Print original 2D DenseNet structure
model_2d = densenet121()
print("2D DenseNet:")
print(model_2d)

# Print 1D DenseNet structure
model_1d = DenseNet1D(channel=3, num_classes=1000)
print("1D DenseNet:")
print(model_1d)