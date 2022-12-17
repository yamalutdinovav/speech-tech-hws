from typing import List

import torch
from torch import nn


class QuartzNetBlock(torch.nn.Module):
    def __init__(
        self,
        feat_in: int,
        filters: int,
        repeat: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        residual: bool,
        separable: bool,
        dropout: float,
    ):

        super().__init__()

        if not residual:
            self.res = None
        else:
            res_list = self._build_conv_block(
                in_channels=feat_in,
                out_channels=filters,
                kernel_size=1,
                stride=1,
                dilation=1,
                separable=False,
                norm=True,
                activation=False,
            )
            self.res = nn.Sequential(*res_list)

        self.conv = nn.ModuleList()
        self._build_repeated_blocks(feat_in, filters, kernel_size, stride, dilation, separable, dropout, repeat)

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res:
            residual = self.res(x)
        for layer in self.conv:
            x = layer(x)
        if self.res:
            x += residual
        return self.out(x)
    
    def _build_repeated_blocks(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        separable: bool,
        dropout: float,
        repeat: bool) -> List[nn.Module]:
        if repeat > 1:
            common_conv_params = {
                'kernel_size': kernel_size,
                'stride': stride,
                'dilation': dilation,
                'separable': separable,
                'out_channels': out_channels,
                'norm': True,
                'dropout': dropout,
            }
            for block_num in range(repeat):
                if block_num == 0:
                    self.conv.extend(self._build_conv_block(in_channels=in_channels, activation=True, **common_conv_params))
                elif block_num == repeat - 1:
                    self.conv.extend(self._build_conv_block(in_channels=out_channels, activation=False, **common_conv_params))
                else:
                    self.conv.extend(self._build_conv_block(in_channels=out_channels, activation=True, **common_conv_params))
        else:
            self.conv.extend(self._build_conv_block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                separable=separable,
                norm=True,
                activation=True,
                dropout=dropout,
            ))

    @staticmethod
    def _build_conv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        separable: bool = True,
        norm: bool = True,
        activation: bool = True,
        dropout: float = 0.0) -> List[nn.Module]:

        padding = (dilation * (kernel_size - 1)) // 2
        layers = []
        if not separable:
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                )
            )
        else:
            layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    kernel_size=kernel_size,
                    groups=in_channels
                ),
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                )
            ])

        if norm:
            norm_layer = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
            layers.append(norm_layer)
        
        if activation:
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        return layers
        



class QuartzNet(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.stride_val = 1

        layers = []
        feat_in = conf.feat_in
        for block in conf.blocks:
            layers.append(QuartzNetBlock(feat_in, **block))
            self.stride_val *= block.stride**block.repeat
            feat_in = block.filters

        self.layers = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, features_length: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.layers(features)
        encoded_len = (
            torch.div(features_length - 1, self.stride_val, rounding_mode="trunc") + 1
        )

        return encoded, encoded_len