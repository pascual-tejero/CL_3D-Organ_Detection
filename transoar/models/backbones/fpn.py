"""Module containing code of the FPN backbone. It is built from https://github.com/bwittmann/transoar."""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from typing import Sequence, Type, Tuple, Union, List, Optional, Dict
import torch
import torch.nn as nn
from torch import Tensor
import math
import numpy as np
from functools import reduce

# **************************************************
#                    Encoder
# **************************************************

class EncoderCnnBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=1,
        bias=False,
        affine=True,
        eps=1e-05
    ):
        super().__init__()

        conv_block_1 = [
            nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=bias
            ),
            nn.InstanceNorm3d(num_features=out_channels, affine=affine, eps=eps),
            nn.ReLU(inplace=True)
        ]

        conv_block_2 = [
            nn.Conv3d(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=1, padding=padding,
                bias=bias
            ),
            nn.InstanceNorm3d(num_features=out_channels, affine=affine, eps=eps),
            nn.ReLU(inplace=True)
        ]

        self._block = nn.Sequential(
            *conv_block_1,
            *conv_block_2
        )

    def forward(self, x):
        return self._block(x)



class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._num_stages = config['num_stages']

        # Determine channels of encoder fmaps
        encoder_out_channels = torch.tensor([config['start_channels'] * 2**stage for stage in range(self._num_stages)])

        # Estimate required stages
        required_stages = set([int(fmap[-1]) for fmap in config['out_fmaps']])
        if False:
            required_stages.add(0)
        self._required_stages = required_stages

        earliest_required_stage = min(required_stages)

        # LATERAL
        # Reduce lateral connections if not needed
        lateral_in_channels = encoder_out_channels[earliest_required_stage:]
        lateral_out_channels = lateral_in_channels.clip(max=config['fpn_channels'])

        self._lateral = nn.ModuleList()
        for in_channels, out_channels in zip(lateral_in_channels, lateral_out_channels):
            self._lateral.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
        self._lateral_levels = len(self._lateral)

        # OUT
        # Ensure that relevant stages have channels according to fpn_channels
        out_in_channels = [lateral_out_channels[-self._num_stages + required_stage].item() for required_stage in required_stages]
        out_out_channels = torch.full((len(out_in_channels),), int(config['fpn_channels'])).tolist()
        out_out_channels[0] = int(config['fpn_channels'])

        self._out = nn.ModuleList()
        for in_channels, out_channels in zip(out_in_channels, out_out_channels):
            self._out.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))

        #  UP
        self._up = nn.ModuleList()
        for level in range(len(lateral_out_channels)-1):
            self._up.append(
                nn.ConvTranspose3d(
                    in_channels=list(reversed(lateral_out_channels))[level], out_channels=list(reversed(lateral_out_channels))[level+1],
                    kernel_size=list(reversed(config['strides']))[level], stride=list(reversed(config['strides']))[level]
                )
            )

    def forward(self, x):
        # Forward lateral
        lateral_out = [lateral(fmap) for lateral, fmap in zip(self._lateral, list(x.values())[-self._lateral_levels:])]

        # Forward up
        up_out = []
        for idx, x in enumerate(reversed(lateral_out)):
            if idx != 0:
                x = x + up
            
            if idx < self._lateral_levels - 1:
                up = self._up[idx](x)

            up_out.append(x)

        # Forward out
        out_fmaps = zip(reversed(up_out), self._required_stages)
        cnn_outputs = {stage: self._out[idx](fmap) for idx, (fmap, stage) in enumerate(out_fmaps)}
        return cnn_outputs
    

class FPN(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels   = config['in_channels']
        kernel_size   = config['kernel_size']
        emb_dim       = config['start_channels']
        data_size     = config['data_size']
        self.out_fmaps = config['out_fmaps'] 

        num_stages = int(math.log2(min(config['data_size'])))-1
        strides = [1]+ [2 for _ in range(num_stages-1)]
        kernel_sizes = [kernel_size for _ in range(num_stages)]

        config['num_stages']  = num_stages
        config['strides']  = strides


        # Build encoder
        self._encoder = nn.ModuleList()
        for k in range(num_stages):
            blk = EncoderCnnBlock(
                    in_channels=in_channels,
                    out_channels=emb_dim,
                    kernel_size=kernel_sizes[k],
                    stride=strides[k]
                )
            self._encoder.append(blk)

            in_channels = emb_dim
            emb_dim *= 2

        # Build decoder
        self._decoder = Decoder(config)

    def forward(self, x):
        down = {}
        for stage_id, module in enumerate(self._encoder):
            x = module(x)
            down['C' + str(stage_id)] = x
        #[print('down',key, item.shape) for key,item in down.items()]

        up = self._decoder(down)
        #[print('up', key, item.shape) for key,item in up.items()]
        return up

    def init_weights(self):
        pass    # TODO
