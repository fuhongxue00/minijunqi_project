
# -*- coding: utf-8 -*-
"""极小UNet骨干 + 多头策略（部署/选子/方向）+ 价值头。
输入通道：10个棋子ID（含EMPTY与UNKNOWN_ENEMY）+ side_to_move + no_battle_ratio + is_deploy_phase = 13
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..constants import BOARD_H, BOARD_W

HW = BOARD_H * BOARD_W

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.cv = nn.Conv2d(c_in, c_out, k, padding=k//2)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.cv(x)))

class TinyUNet(nn.Module):
    def __init__(self, c_in=13, base=16):
        super().__init__()
        self.down1 = nn.Sequential(ConvBNAct(c_in, base), ConvBNAct(base, base))
        self.pool1 = nn.MaxPool2d(2)  # 6x6 -> 3x3
        self.mid = nn.Sequential(ConvBNAct(base, base*2), ConvBNAct(base*2, base*2))
        self.up = nn.ConvTranspose2d(base*2, base, 2, stride=2)  # 3x3 -> 6x6
        self.fuse = ConvBNAct(base*2, base)
        self.head_deploy_cell = nn.Linear(base*BOARD_H*BOARD_W, HW)
        self.head_select = nn.Linear(base*BOARD_H*BOARD_W, HW)
        self.head_dir = nn.Linear(base*BOARD_H*BOARD_W, 4)
        self.head_value = nn.Linear(base*BOARD_H*BOARD_W, 1)
    def forward(self, x):
        d1 = self.down1(x)
        m = self.mid(self.pool1(d1))
        up = self.up(m)
        if up.shape[-2:] != d1.shape[-2:]:
            up = F.interpolate(up, size=d1.shape[-2:], mode='nearest')
        f = self.fuse(torch.cat([d1, up], dim=1))
        flat = f.flatten(1)
        return {
            'deploy_cell_logits': self.head_deploy_cell(flat),
            'select_piece_logits': self.head_select(flat),
            'move_dir_logits': self.head_dir(flat),
            'value': self.head_value(flat).squeeze(-1)
        }
