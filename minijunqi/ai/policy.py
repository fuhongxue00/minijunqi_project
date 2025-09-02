
# # -*- coding: utf-8 -*-
# from __future__ import annotations
# from typing import Tuple, Dict
# import torch
# import torch.nn.functional as F
# import numpy as np
# from ..constants import BOARD_H, BOARD_W, Player
# from ..board import Board
# from .net import TinyUNet

# DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

# def encode_obs(board: Board, viewer: Player, side_to_move: Player, no_battle_counter: int, no_battle_limit: int, reveal_all: bool=False, is_deploy: bool=False) -> torch.Tensor:
#     """返回 (C,H,W) 张量。通道：10个ID、side_to_move、no_battle_ratio、is_deploy。"""
#     from ..constants import PieceID
#     obs = board.observe(viewer, reveal_all=reveal_all, hide_enemy_positions=is_deploy and not reveal_all)
#     H, W = BOARD_H, BOARD_W
#     C = 13
#     x = np.zeros((C, H, W), dtype=np.float32)
#     for r in range(H):
#         for c in range(W):
#             vid = obs[r][c]
#             if 0 <= vid <= 9:
#                 x[vid, r, c] = 1.0
#     x[10,:,:] = 1.0 if side_to_move == viewer else 0.0
#     x[11,:,:] = float(no_battle_counter) / max(1,no_battle_limit)
#     x[12,:,:] = 1.0 if is_deploy else 0.0
#     return torch.from_numpy(x)

# def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
#     logits = logits.clone()
#     logits[mask==0] = -1e9
#     return F.softmax(logits, dim=dim)

# class SharedPolicy:
#     def __init__(self, device='cpu'):
#         self.net = TinyUNet().to(device)
#         self.device = device
#         self.net.eval()
#     def load(self, path: str): self.net.load_state_dict(torch.load(path, map_location=self.device))
#     def save(self, path: str): torch.save(self.net.state_dict(), path)

#     # 部署策略：固定顺序给定棋子，只预测落点
#     def select_deploy(self, board: Board, viewer: Player, piece_to_place, no_battle_counter: int, no_battle_limit: int, temperature: float=1.0):
#         with torch.no_grad():
#             x = encode_obs(board, viewer, side_to_move=viewer, no_battle_counter=no_battle_counter, no_battle_limit=no_battle_limit, is_deploy=True).unsqueeze(0).to(self.device)
#             out = self.net(x)
#         cell_mask = torch.zeros((BOARD_H*BOARD_W,), dtype=torch.float32, device=self.device)
#         for r in range(BOARD_H):
#             for c in range(BOARD_W):
#                 ok = board.can_place(viewer, piece_to_place, (r,c))
#                 cell_mask[r*BOARD_W+c] = 1.0 if ok else 0.0
#         lc = out['deploy_cell_logits'].squeeze(0) / max(1e-6, temperature)
#         pc = masked_softmax(lc, cell_mask, dim=0).cpu().numpy()
#         if pc.sum() <= 0:
#             print('policy.py出现softmax后的负值')
#             for idx in range(BOARD_H*BOARD_W):
#                 if cell_mask[idx] > 0:
#                     return divmod(idx, BOARD_W)
#             return (0,0)
#         idx = np.random.choice(BOARD_H*BOARD_W, p=pc)
#         return divmod(idx, BOARD_W)

#     # 行动策略：选子 + 方向
#     def select_move(self, board: Board, viewer: Player, side_to_move: Player, no_battle_counter: int, no_battle_limit: int, temperature: float=1.0):
#         with torch.no_grad():
#             x = encode_obs(board, viewer, side_to_move=side_to_move, no_battle_counter=no_battle_counter, no_battle_limit=no_battle_limit, is_deploy=False).unsqueeze(0).to(self.device)
#             out = self.net(x)
#         select_mask = torch.zeros((BOARD_H*BOARD_W,), dtype=torch.float32, device=self.device)
#         for r in range(BOARD_H):
#             for c in range(BOARD_W):
#                 p = board.get((r,c))
#                 if p is None or p.owner != viewer or not p.can_move(): continue
#                 for dr,dc in DIRS:
#                     rr,cc=r+dr,c+dc
#                     if 0<=rr<BOARD_H and 0<=cc<BOARD_W:
#                         q=board.get((rr,cc))
#                         if q is None or q.owner != viewer:
#                             select_mask[r*BOARD_W+c]=1.0; break
#         ls = out['select_piece_logits'].squeeze(0) / max(1e-6, temperature)
#         ps = masked_softmax(ls, select_mask, dim=0).cpu().numpy()
#         if ps.sum() <= 0: return (0,0),(0,0)
#         idx = np.random.choice(BOARD_H*BOARD_W, p=ps)
#         r, c = divmod(idx, BOARD_W)
#         dir_mask = torch.zeros((4,), dtype=torch.float32, device=self.device)
#         for k,(dr,dc) in enumerate(DIRS):
#             rr,cc=r+dr,c+dc
#             if 0<=rr<BOARD_H and 0<=cc<BOARD_W:
#                 q=board.get((rr,cc))
#                 if q is None or q.owner != viewer:
#                     dir_mask[k]=1.0
#         ld = out['move_dir_logits'].squeeze(0) / max(1e-6, temperature)
#         pd = masked_softmax(ld, dir_mask, dim=0).cpu().numpy()
#         if pd.sum() <= 0: return (r,c),(r,c)
#         k = np.random.choice(4, p=pd)
#         dr,dc=DIRS[k]
#         return (r,c),(r+dr,c+dc)


# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict, Deque
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

try:
    from ..constants import BOARD_H, BOARD_W, Player
except Exception:
    class Player:
        RED = 0
        BLUE = 1
    BOARD_H, BOARD_W = 6, 6

try:
    from ..board import Board
except Exception:
    Board = object

from .net import PolicyNet, HISTORY_STEPS, PIECE_PLANES_PER_FRAME, STATE_EXTRA_C, RESERVED_EXTRA_C

DIRS = [(-1,0),(1,0),(0,-1),(0,1)]
HW = BOARD_H * BOARD_W

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    logits = logits.clone()
    logits[mask==0] = -1e9
    return F.softmax(logits, dim=dim)

def rotate_hw(x_hw: torch.Tensor) -> torch.Tensor:
    """Rotate HxW (or BxHxW) tensor by 180 degrees."""
    if x_hw.ndim == 2:
        return torch.flip(x_hw, dims=(0,1))
    elif x_hw.ndim == 3:
        return torch.flip(x_hw, dims=(1,2))
    else:
        raise ValueError('rotate_hw expects 2D or 3D tensor')

def rotate_chw(x_chw: torch.Tensor) -> torch.Tensor:
    """Rotate CxHxW tensor by 180 degrees."""
    return torch.flip(x_chw, dims=(1,2))

def encode_board_planes(board: Board, viewer: Player, reveal_all: bool=False, is_deploy: bool=False) -> torch.Tensor:
    """Return (10,H,W) one-hot planes (placeholder if project API differs)."""
    # Attempt to use project's own observation API; otherwise fall back to an all-zeros tensor.
    H, W = BOARD_H, BOARD_W
    C = PIECE_PLANES_PER_FRAME
    x = np.zeros((C, H, W), dtype=np.float32)
    try:
        obs = board.observe(viewer, reveal_all=reveal_all, hide_enemy_positions=is_deploy and not reveal_all)
        for r in range(H):
            for c in range(W):
                vid = obs[r][c]
                if 0 <= vid < C:
                    x[vid, r, c] = 1.0
    except Exception:
        pass
    return torch.from_numpy(x)  # (10,H,W)

def stack_history(history: Deque[torch.Tensor], current_10chw: torch.Tensor, steps: int = HISTORY_STEPS) -> torch.Tensor:
    while len(history) >= steps:
        history.popleft()
    history.append(current_10chw)
    H, W = current_10chw.shape[-2:]
    frames = list(history)
    if len(frames) < steps:
        pad = [torch.zeros_like(current_10chw) for _ in range(steps - len(frames))]
        frames = pad + frames
    return torch.cat(frames, dim=0)  # (steps*10, H, W)

class SharedPolicy:
    def __init__(self, net=None,device='cpu'):
        self.net = PolicyNet().to(device) if net is None else net
        self.device = device
        self.net.eval()
        self._hist: Dict[Player, Deque[torch.Tensor]] = {
            getattr(Player, 'RED', 0): deque(maxlen=HISTORY_STEPS),
            getattr(Player, 'BLUE', 1): deque(maxlen=HISTORY_STEPS),
        }

    def reset_history(self):
        for k in list(self._hist.keys()):
            self._hist[k].clear()

    def load(self, path: str):
        sd = torch.load(path, map_location=self.device)
        self.net.load_state_dict(sd, strict=True)



    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def _encode_obs(self, board, viewer, side_to_move, no_battle_ratio: float, is_deploy: bool):
        planes10 = encode_board_planes(board, viewer, reveal_all=False, is_deploy=is_deploy)  # (10,H,W)
        if viewer == getattr(Player, 'BLUE', 1):
            planes10 = rotate_chw(planes10)
        hist = stack_history(self._hist[viewer], planes10)  # (80,H,W)
        H, W = BOARD_H, BOARD_W
        s_side = torch.full((1,H,W), 1.0 if side_to_move == viewer else 0.0, dtype=torch.float32)
        s_ratio = torch.full((1,H,W), float(no_battle_ratio), dtype=torch.float32)
        s_deploy = torch.full((1,H,W), 1.0 if is_deploy else 0.0, dtype=torch.float32)
        state = torch.cat([s_side, s_ratio, s_deploy], dim=0)  # (3,H,W)
        reserved = torch.zeros((RESERVED_EXTRA_C, H, W), dtype=torch.float32)
        x = torch.cat([hist, state, reserved], dim=0)  # (90,H,W)
        return x

    def select_deploy(self, board, viewer, piece_to_place, no_battle_ratio: float = 0.0, temperature: float = 1.0):
        # with torch.no_grad():
        x = self._encode_obs(board, viewer, side_to_move=viewer, no_battle_ratio=no_battle_ratio, is_deploy=True).unsqueeze(0).to(self.device)
        out = self.net(x)  # extras default zeros
        cell_mask_rot = torch.zeros((HW,), dtype=torch.float32, device=self.device)
        # Project-specific legality query; fallback: allow all
        try:
            for r in range(BOARD_H):
                for c in range(BOARD_W):
                    ok = board.can_place(viewer, piece_to_place, (r,c))
                    cell_mask_rot[r*BOARD_W+c] = 1.0 if ok else 0.0
        except Exception:
            cell_mask_rot[:] = 1.0
        if viewer == getattr(Player, 'BLUE', 1):
            mask_hw = cell_mask_rot.view(BOARD_H, BOARD_W)
            mask_hw = rotate_hw(mask_hw)
            cell_mask_rot = mask_hw.flatten()
        lc = out['deploy_cell_logits'].squeeze(0) / max(1e-6, temperature)
        pc_tensor = masked_softmax(lc, cell_mask_rot, dim=0)
        pc = pc_tensor.cpu().detach().numpy()
        if pc.sum() <= 0:
            sel_idx = int((cell_mask_rot > 0).nonzero()[0]) if (cell_mask_rot > 0).any() else 0
        else:
            sel_idx = int(np.random.choice(HW, p=pc))
        rr, cc = divmod(sel_idx, BOARD_W)
        if viewer == getattr(Player, 'BLUE', 1):
            rr, cc = BOARD_H-1-rr, BOARD_W-1-cc
        return (rr, cc), pc_tensor

    def select_move(self, board, viewer, side_to_move, no_battle_ratio: float, temperature: float = 1.0):
        # with torch.no_grad():
        #     x = self._encode_obs(board, viewer, side_to_move=side_to_move, no_battle_ratio=no_battle_ratio, is_deploy=False).unsqueeze(0).to(self.device)
        #     out1 = self.net(x)
        x = self._encode_obs(board, viewer, side_to_move=side_to_move, no_battle_ratio=no_battle_ratio, is_deploy=False).unsqueeze(0).to(self.device)
        out1 = self.net(x)
        
        select_mask = torch.zeros((BOARD_H, BOARD_W), dtype=torch.float32, device=self.device)
        try:
            for r in range(BOARD_H):
                for c in range(BOARD_W):
                    p = board.get((r,c))
                    if not p or getattr(p, 'owner', viewer) != viewer or not getattr(p, 'can_move', lambda: True)():
                        continue
                    for dr,dc in DIRS:
                        rr,cc=r+dr,c+dc
                        if 0<=rr<BOARD_H and 0<=cc<BOARD_W:
                            q=board.get((rr,cc))
                            if (q is None) or getattr(q, 'owner', -1) != viewer:
                                select_mask[r,c]=1.0; break
        except Exception:
            select_mask[:] = 1.0
        mask_sel_rot = rotate_hw(select_mask) if viewer==getattr(Player,'BLUE',1) else select_mask
        ls = out1['select_piece_logits'].squeeze(0) / max(1e-6, temperature)
        ps_tensor = masked_softmax(ls, mask_sel_rot.flatten(), dim=0)
        ps = ps_tensor.cpu().detach().numpy()
        if ps.sum() <= 0:
            idx = int(mask_sel_rot.flatten().argmax().item())
        else:
            idx = int(np.random.choice(HW, p=ps))
        rr_sel, cc_sel = divmod(idx, BOARD_W)
        if viewer == getattr(Player,'BLUE',1):
            rr_sel, cc_sel = BOARD_H-1-rr_sel, BOARD_W-1-cc_sel
        src = (rr_sel, cc_sel)

        # target head extra channel 0 = selected piece one-hot (canonical orientation)
        sel_onehot = torch.zeros((1,1,BOARD_H,BOARD_W), dtype=torch.float32, device=self.device)
        rr_can, cc_can = rr_sel, cc_sel
        if viewer == getattr(Player,'BLUE',1):
            rr_can, cc_can = BOARD_H-1-rr_sel, BOARD_W-1-cc_sel
        sel_onehot[:,0, rr_can, cc_can] = 1.0
        # with torch.no_grad():
        out2 = self.net(x, target_extra=sel_onehot)

        target_mask = torch.zeros((BOARD_H, BOARD_W), dtype=torch.float32, device=self.device)
        try:
            for dr,dc in DIRS:
                rr,cc=src[0]+dr, src[1]+dc
                if 0<=rr<BOARD_H and 0<=cc<BOARD_W:
                    q=board.get((rr,cc))
                    if (q is None) or getattr(q,'owner',-1) != viewer:
                        target_mask[rr,cc]=1.0
        except Exception:
            target_mask[:] = 1.0
        mask_tgt_rot = rotate_hw(target_mask) if viewer==getattr(Player,'BLUE',1) else target_mask
        lt = out2['target_cell_logits'].squeeze(0) / max(1e-6, temperature)
        pt_tensor = masked_softmax(lt, mask_tgt_rot.flatten(), dim=0)
        pt = pt_tensor.cpu().detach().numpy()
        if pt.sum() <= 0:
            j = int(mask_tgt_rot.flatten().argmax().item())
        else:
            j = int(np.random.choice(HW, p=pt))
        rr_t, cc_t = divmod(j, BOARD_W)
        if viewer == getattr(Player,'BLUE',1):
            rr_t, cc_t = BOARD_H-1-rr_t, BOARD_W-1-cc_t
        dst = (rr_t, cc_t)
        # 起点坐标，终点坐标，起点概率(形状拉平），终点拉平的概率
        return src, dst, ps_tensor, pt_tensor
