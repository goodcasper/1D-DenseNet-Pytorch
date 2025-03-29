import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# Define a single dense layer (1D version) used inside Dense Blocks
class _DenseLayer1D(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate):
        super(_DenseLayer1D, self).__init__()

        # First BatchNorm + ReLU + 1x1 Conv (bottleneck layer)
        self.add_module('norm1', nn.BatchNorm1d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv1d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))

        # Second BatchNorm + ReLU + 3x3 Conv (produces new features)
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2',
                        nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate  # Dropout probability

    def forward(self, x):
        # Sequentially apply all layers defined above
        new_features = super(_DenseLayer1D, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        # Concatenate input and new features (DenseNet connectivity)
        return torch.cat([x, new_features], 1)


# Define a Dense Block composed of multiple DenseLayers
class _DenseBlock1D(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, drop_rate):
        super(_DenseBlock1D, self).__init__()
        for i in range(num_layers):
            # Input channels grow with each added layer due to concatenation
            layer = _DenseLayer1D(
                in_channels + i * growth_rate, growth_rate, bn_size, drop_rate
            )
            self.add_module(f'denselayer{i + 1}', layer)


# Define a Transition layer used between DenseBlocks (downsampling + channel reduction)
class _Transition1D(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition1D, self).__init__()
        self.add_module('norm', nn.BatchNorm1d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))  # Downsample sequence length by half


# Define the full DenseNet1D architecture
class DenseNet1D(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=5, channel=3):
        super(DenseNet1D, self).__init__()

        # Initial convolution layer (also known as the "stem")
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(channel, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm1d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),  # Downsample input
        ]))

        # Stack DenseBlocks and Transition layers
        num_features = num_init_features  # Current number of feature channels
        for i, num_layers in enumerate(block_config):
            # Add DenseBlock
            block = _DenseBlock1D(
                num_layers=num_layers,
                in_channels=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features += num_layers * growth_rate  # Update channel count

            # Add Transition layer between blocks (except after the last one)
            if i != len(block_config) - 1:
                trans = _Transition1D(in_channels=num_features, out_channels=num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # Final BatchNorm layer after last DenseBlock
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))

        self.num_classes = num_classes

        # Classification layer (optional, depending on num_classes)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Feature extraction
        features = self.features(x)
        out = F.relu(features, inplace=True)

        # Global average pooling over time dimension, flatten to shape [B, C]
        out = F.adaptive_avg_pool1d(out, 1).view(features.size(0), -1)

        # If in classification mode, apply final linear layer + sigmoid
        if self.num_classes != 0:
            out = self.classifier(out)
            out = torch.sigmoid(out)

        return out
