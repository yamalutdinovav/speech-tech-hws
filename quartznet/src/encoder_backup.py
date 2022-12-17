from typing import List

import torch
from torch import nn


class ChannelShuffle(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[-1]
        x = x.view(-1, self.groups, self.channels_per_group, shape)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, self.groups * self.channels_per_group, shape)
        return x



def get_conv_block(
    feat_in: int,
    filters: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    separable: bool,
    activation: bool = True
) -> List[nn.Module]:
    if separable:
        padding = (kernel_size - 1) // 2 if dilation == 1 else (dilation * kernel_size) // 2 - 1
        layers = [
            nn.Conv1d(feat_in, feat_in, kernel_size, stride=stride, dilation=dilation, groups=feat_in, padding=padding),
            nn.Conv1d(feat_in, filters, kernel_size=1, stride=1, dilation=1, groups=1)
        ]
    else:
        layers = [
            nn.Conv1d(feat_in, filters, kernel_size, stride=stride, dilation=dilation, groups=1)
        ]
    layers.append(nn.BatchNorm1d(filters))
    if separable:
        layers.append(ChannelShuffle(filters, feat_in))
    if activation:
        layers.append(nn.ReLU())
    return layers



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
        self.residual = residual

        self.res = None
        if self.residual:
            res_list = get_conv_block(
                feat_in, 
                filters,
                kernel_size=1,
                stride=1,
                dilation=1,
                separable=False,
                activation=False
            )
            self.res = nn.Sequential(*res_list)

        conv_list = get_conv_block(feat_in, filters, kernel_size, stride, dilation, separable)
        for _ in range(repeat - 1):
            conv_list.extend(get_conv_block(filters, filters, kernel_size, stride, dilation, separable))

        self.conv = nn.Sequential(*conv_list)

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.set_detect_anomaly(True):
            output = self.conv(x)
            if self.residual:
                output = output + self.res(x)
            output = self.out(output)
            return output


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

