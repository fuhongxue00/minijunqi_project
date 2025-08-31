
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import copy
from .constants import BOARD_H, BOARD_W, Player, PieceID, STRENGTH, FLAG_ALLOWED_COLS
from .pieces import Piece

Coord = Tuple[int, int]

class Board:
    def __init__(self):
        self.grid: List[List[Optional[Piece]]] = [[None for _ in range(BOARD_W)] for _ in range(BOARD_H)]
    def clone(self) -> 'Board':
        return copy.deepcopy(self)
    def in_bounds(self, r, c): return 0<=r<BOARD_H and 0<=c<BOARD_W
    def get(self, rc: Coord): r,c=rc; return self.grid[r][c]
    def set(self, rc: Coord, p: Optional[Piece]): r,c=rc; self.grid[r][c]=p
    def iter_coords(self):
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                yield (r,c)
    def home_rows(self, player: Player):
        return [0,1] if player==Player.BLUE else [BOARD_H-2, BOARD_H-1]
    def last_row(self, player: Player): return 0 if player==Player.BLUE else BOARD_H-1
    def is_home_cell(self, player: Player, rc: Coord):
        r,c=rc; return r in self.home_rows(player)
    def can_place(self, player: Player, pid: PieceID, rc: Coord) -> bool:
        r,c=rc
        if not self.in_bounds(r,c): return False
        if self.get((r,c)) is not None: return False
        if not self.is_home_cell(player, (r,c)): return False
        if pid==PieceID.FLAG:
            if r!=self.last_row(player): return False
            if c not in FLAG_ALLOWED_COLS: return False
        return True
    def place(self, player: Player, pid: PieceID, rc: Coord) -> bool:
        if not self.can_place(player, pid, rc): return False
        self.set(rc, Piece(pid, player)); return True
    DIRS=[(-1,0),(1,0),(0,-1),(0,1)]
    def neighbors(self, rc: Coord):
        r,c=rc; out=[]; 
        for dr,dc in Board.DIRS:
            rr,cc=r+dr,c+dc
            if self.in_bounds(rr,cc): out.append((rr,cc))
        return out
    def can_move_from_to(self, player: Player, src: Coord, dst: Coord) -> bool:
        p=self.get(src)
        if p is None or p.owner!=player or not p.can_move(): return False
        if dst not in self.neighbors(src): return False
        q=self.get(dst)
        if q is not None and q.owner==player: return False
        return True
    def compare(self, a: Piece, d: Piece) -> str:
        if d.pid==PieceID.FLAG: return 'attacker'
        if a.pid==PieceID.BOMB or d.pid==PieceID.BOMB: return 'both'
        sa,sd=STRENGTH[a.pid],STRENGTH[d.pid]
        if sa>sd: return 'attacker'
        if sa<sd: return 'defender'
        return 'both'
    def move(self, player: Player, src: Coord, dst: Coord) -> Dict:
        if not self.can_move_from_to(player, src, dst):
            return {'ok':False,'reason':'illegal'}
        p=self.get(src); q=self.get(dst)
        if q is None:
            self.set(dst,p); self.set(src,None)
            return {'ok':True,'type':'move','flag_captured':False}
        outcome=self.compare(p,q); flag_captured=(q.pid==PieceID.FLAG)
        if outcome=='attacker':
            self.set(dst,p); self.set(src,None)
            return {'ok':True,'type':'capture','result':'attacker','flag_captured':flag_captured}
        elif outcome=='defender':
            self.set(src,None)
            return {'ok':True,'type':'capture','result':'defender','flag_captured':False}
        else:
            self.set(src,None); self.set(dst,None)
            return {'ok':True,'type':'capture','result':'both','flag_captured':flag_captured}
    def observe(self, viewer: Player, reveal_all: bool=False):
        obs=[[0 for _ in range(BOARD_W)] for _ in range(BOARD_H)]
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                p=self.grid[r][c]
                if p is None: obs[r][c]=int(PieceID.EMPTY)
                elif reveal_all or p.owner==viewer: obs[r][c]=int(p.pid)
                else: obs[r][c]=int(PieceID.UNKNOWN_ENEMY)
        return obs
