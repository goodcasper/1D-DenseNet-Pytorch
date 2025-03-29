from densenet1d import DenseNet1D
import torch

# Suppose the input is a batch of 8 samples, each with 3 channels and 1000 time points
x = torch.randn(8, 3, 1000)
channel = x.size(1)

# 初始化模型
model = DenseNet1D(channel=channel, num_classes=0)

# 前向傳遞
out = model(x)
print(out.shape)  # The output tensor shape would be torch.Size([8, feature_dimension])