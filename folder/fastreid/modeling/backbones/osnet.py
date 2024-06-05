# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# based on:
# https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py

import logging
import copy
import torch
from torch import nn

from fastreid.layers import get_norm
from fastreid.utils import comm
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY

logger = logging.getLogger(__name__)
model_urls = {
    'osnet_x1_0':
        'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY',
    'osnet_x0_75':
        'https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq',
    'osnet_x0_5':
        'https://drive.google.com/uc?id=16DGLbZukvVYgINws8u8deSaOqjybZ83i',
    'osnet_x0_25':
        'https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs',
    'osnet_ibn_x1_0':
        'https://drive.google.com/uc?id=1sr90V6irlYYDd4_4ISU2iruoRG8J__6l'
}


##########
# Basic layers
##########
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            bn_norm,
            stride=1,
            padding=0,
            groups=1,
            IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, bn_norm, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, bn_norm, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = get_norm(bn_norm, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, bn_norm, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups
        )
        self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.
    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels, bn_norm):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


##########
# Building blocks for omni-scale feature learning
##########
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
            self,
            in_channels,
            num_gates=None,
            return_gates=False,
            gate_activation='sigmoid',
            reduction=16,
            layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None: num_gates = in_channels
        self.return_gates = return_gates

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm: self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = nn.Identity()
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None: x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.gate_activation(x)
        if self.return_gates: return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(
            self,
            in_channels,
            out_channels,
            bn_norm,
            IN=False,
            bottleneck_reduction=4,
            **kwargs
    ):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels, bn_norm)
        self.conv2a = LightConv3x3(mid_channels, mid_channels, bn_norm)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels, bn_norm),
            LightConv3x3(mid_channels, mid_channels, bn_norm),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels, bn_norm),
            LightConv3x3(mid_channels, mid_channels, bn_norm),
            LightConv3x3(mid_channels, mid_channels, bn_norm),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels, bn_norm),
            LightConv3x3(mid_channels, mid_channels, bn_norm),
            LightConv3x3(mid_channels, mid_channels, bn_norm),
            LightConv3x3(mid_channels, mid_channels, bn_norm),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn_norm)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels, bn_norm)
        self.IN = None
        if IN: self.IN = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return self.relu(out)


##########
# Network architecture
##########
class OSNet(nn.Module):
    """Omni-Scale Network.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(
            self,
            blocks,
            layers,
            channels,
            bn_norm,
            IN=False,
            **kwargs
    ):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1

        # convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, bn_norm, stride=2, padding=3, IN=IN)
      #  self.conv1_clone = ConvLayer(3, channels[0], 7, bn_norm, stride=2, padding=3, IN=IN)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            bn_norm,
            reduce_spatial_size=True,
            IN=IN
        )
        # self.conv2_clone = self._make_layer(
        #     blocks[0],
        #     layers[0],
        #     channels[0],
        #     channels[1],
        #     bn_norm,
        #     reduce_spatial_size=True,
        #     IN=IN
        # )        
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            bn_norm,
            reduce_spatial_size=True
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            bn_norm,
            reduce_spatial_size=False
        )
        #self.conv5 = Conv1x1(channels[3], channels[3], bn_norm)
        self.conv51 = Conv1x1(channels[3], channels[3], bn_norm)
        self.conv52 = Conv1x1(channels[3], channels[3], bn_norm)
        self.conv53= Conv1x1(channels[3], channels[3], bn_norm)

        self.view_predictor1 = nn.Conv2d(in_channels= 256, out_channels= 128, kernel_size= (3, 3), stride= 2, padding= 1)
        self.view_predictor2 = nn.Conv2d(in_channels= 128, out_channels= 256, kernel_size= (3, 3), stride= 2, padding= 1)
        self.view_predictor3 = nn.Conv2d(in_channels= 256, out_channels= 512,kernel_size= (3, 3), stride= 2, padding= 1)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.view_predictor4 = nn.Linear(in_features= 1*1*512, out_features= 3)
        self.view_predictor5 = nn.Softmax()
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()


        #--------------------
        self._init_params()

    def _make_layer(
            self,
            block,
            layer,
            in_channels,
            out_channels,
            bn_norm,
            reduce_spatial_size,
            IN=False
    ):
        layers = []

        layers.append(block(in_channels, out_channels, bn_norm, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, bn_norm, IN=IN))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels, bn_norm),
                    nn.AvgPool2d(2, stride=2),
                )
            )

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #z = copy.deepcopy(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        y = x
        # y = self.conv1_clone(z)
        # y = self.maxpool2(y)
        # y = self.conv2_clone(y)
        # y = self.view_predictor1(y)
        # y = self.bn1(y)
        # y = self.relu(y)
        # y = self.view_predictor2(y)
        # y = self.bn2(y)
        # y = self.relu(y)
        # y = self.view_predictor3(y)
        # y = self.relu(y)
        # y = self.globalpool(y)
        # y = y.view(-1, 512)
        # y = self.dropout(y)
        # y = self.view_predictor4(y)
        # y = self.view_predictor5(y)
           
        x = self.conv3(x)
        x = self.conv4(x)
        #x = self.conv5(x)
        # x1 = self.conv51(x)
        # x2 = self.conv52(x)
        # x3 = self.conv53(x)

        # y1 = y[:, 0]
        # y2 = y[:, 1]
        # y3 = y[:, 2]
        # x = x1*y1[:,None, None, None] + x2*y2[:, None, None, None] + x3*y3[:,None, None, None]
        return x, y


def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown
    from collections import OrderedDict
    import warnings
    import logging

    logger = logging.getLogger(__name__)

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    filename = key + '_imagenet.pth'
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        logger.info(f"Pretrain model don't exist, downloading from {model_urls[key]}")
        if comm.is_main_process():
            gdown.download(model_urls[key], cached_file, quiet=False)

    comm.synchronize()

    state_dict = torch.load(cached_file, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    return model_dict


@BACKBONE_REGISTRY.register()
def build_osnet_backbone():
    """
    Create a OSNet instance from config.
    Returns:
        OSNet: a :class:`OSNet` instance
    """

    # fmt: off

    # fmt: on

    num_blocks_per_stage = [2, 2, 2]
    num_channels_per_stage = {
        "x1_0": [64, 256, 384, 512],
        "x0_75": [48, 192, 288, 384],
        "x0_5": [32, 128, 192, 256],
        "x0_25": [16, 64, 96, 128]}['x1_0']
    model = OSNet([OSBlock, OSBlock, OSBlock], num_blocks_per_stage, num_channels_per_stage,
                  'BN', IN=False)
    return model
