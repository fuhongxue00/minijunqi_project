
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict
import torch
import torch.nn.functional as F
import numpy as np
from ..constants import BOARD_H, BOARD_W, Player
from ..board import Board
from .net import TinyUNet

DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

def encode_obs(board: Board, viewer: Player, side_to_move: Player, no_battle_counter: int, no_battle_limit: int, reveal_all: bool=False, is_deploy: bool=False) -> torch.Tensor:
    """返回 (C,H,W) 张量。通道：10个ID、side_to_move、no_battle_ratio、is_deploy。"""
    from ..constants import PieceID
    obs = board.observe(viewer, reveal_all=reveal_all, is_deploy=is_deploy)
    H, W = BOARD_H, BOARD_W
    C = 13
    x = np.zeros((C, H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            vid = obs[r][c]
            if 0 <= vid <= 9:
                x[vid, r, c] = 1.0
    x[10,:,:] = 1.0 if side_to_move == viewer else 0.0
    x[11,:,:] = float(no_battle_counter) / max(1,no_battle_limit)
    x[12,:,:] = 1.0 if is_deploy else 0.0
    return torch.from_numpy(x)

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    logits = logits.clone()
    logits[mask==0] = -1e9
    return F.softmax(logits, dim=dim)

class SharedPolicy:
    def __init__(self, device='cpu'):
        self.net = TinyUNet().to(device)
        self.device = device
        self.net.eval()
    def load(self, path: str): self.net.load_state_dict(torch.load(path, map_location=self.device))
    def save(self, path: str): torch.save(self.net.state_dict(), path)

    # 部署策略：固定顺序给定棋子，只预测落点
    def select_deploy(self, board: Board, viewer: Player, piece_to_place, no_battle_counter: int, no_battle_limit: int, temperature: float=1.0):
        with torch.no_grad():
            x = encode_obs(board, viewer, side_to_move=viewer, no_battle_counter=no_battle_counter, no_battle_limit=no_battle_limit, is_deploy=True).unsqueeze(0).to(self.device)
            out = self.net(x)
        cell_mask = torch.zeros((BOARD_H*BOARD_W,), dtype=torch.float32, device=self.device)
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                ok = board.can_place(viewer, piece_to_place, (r,c))
                cell_mask[r*BOARD_W+c] = 1.0 if ok else 0.0
        lc = out['deploy_cell_logits'].squeeze(0) / max(1e-6, temperature)
        pc = masked_softmax(lc, cell_mask, dim=0).cpu().numpy()
        if pc.sum() <= 0:
            print('policy.py出现softmax后的负值')
            for idx in range(BOARD_H*BOARD_W):
                if cell_mask[idx] > 0:
                    return divmod(idx, BOARD_W)
            return (0,0)
        idx = np.random.choice(BOARD_H*BOARD_W, p=pc)
        return divmod(idx, BOARD_W)

    # 行动策略：选子 + 方向
    def select_move(self, board: Board, viewer: Player, side_to_move: Player, no_battle_counter: int, no_battle_limit: int, temperature: float=1.0):
        with torch.no_grad():
            x = encode_obs(board, viewer, side_to_move=side_to_move, no_battle_counter=no_battle_counter, no_battle_limit=no_battle_limit, is_deploy=False).unsqueeze(0).to(self.device)
            out = self.net(x)
        select_mask = torch.zeros((BOARD_H*BOARD_W,), dtype=torch.float32, device=self.device)
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                p = board.get((r,c))
                if p is None or p.owner != viewer or not p.can_move(): continue
                for dr,dc in DIRS:
                    rr,cc=r+dr,c+dc
                    if 0<=rr<BOARD_H and 0<=cc<BOARD_W:
                        q=board.get((rr,cc))
                        if q is None or q.owner != viewer:
                            select_mask[r*BOARD_W+c]=1.0; break
        ls = out['select_piece_logits'].squeeze(0) / max(1e-6, temperature)
        ps = masked_softmax(ls, select_mask, dim=0).cpu().numpy()
        if ps.sum() <= 0: return (0,0),(0,0)
        idx = np.random.choice(BOARD_H*BOARD_W, p=ps)
        r, c = divmod(idx, BOARD_W)
        dir_mask = torch.zeros((4,), dtype=torch.float32, device=self.device)
        for k,(dr,dc) in enumerate(DIRS):
            rr,cc=r+dr,c+dc
            if 0<=rr<BOARD_H and 0<=cc<BOARD_W:
                q=board.get((rr,cc))
                if q is None or q.owner != viewer:
                    dir_mask[k]=1.0
        ld = out['move_dir_logits'].squeeze(0) / max(1e-6, temperature)
        pd = masked_softmax(ld, dir_mask, dim=0).cpu().numpy()
        if pd.sum() <= 0: return (r,c),(r,c)
        k = np.random.choice(4, p=pd)
        dr,dc=DIRS[k]
        return (r,c),(r+dr,c+dc)
