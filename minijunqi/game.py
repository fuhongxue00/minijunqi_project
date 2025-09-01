
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from .board import Board, Coord
from .constants import Player, PieceID, INITIAL_POOL, DEFAULT_NO_BATTLE_DRAW, BOARD_H, BOARD_W

@dataclass
class GameConfig:
    no_battle_draw_steps: int = DEFAULT_NO_BATTLE_DRAW

@dataclass
class GameState:
    board: Board = field(default_factory=Board)
    turn: Player = Player.RED
    phase: str = 'deploy'
    pools: Dict[Player, Dict[PieceID, int]] = field(default_factory=lambda: {
        Player.RED: dict(INITIAL_POOL),
        Player.BLUE: dict(INITIAL_POOL),
    })
    no_battle_counter: int = 0
    winner: Optional[Player] = None
    end_reason: Optional[str] = None

class Game:
    def __init__(self, cfg: GameConfig = GameConfig()):
        self.cfg = cfg
        self.state = GameState()
    def can_deploy_any(self, player: Player) -> bool:
        return any(v>0 for v in self.state.pools[player].values())
    def deploy(self, player: Player, pid: PieceID, rc: Coord) -> bool:
        if self.state.phase != 'deploy': return False
        pool = self.state.pools[player]
        if pool.get(pid,0) <= 0: return False
        if not self.state.board.place(player, pid, rc): return False
        pool[pid] -= 1
        self.state.turn = Player.RED if player==Player.BLUE else Player.BLUE
        if (not self.can_deploy_any(Player.RED)) and (not self.can_deploy_any(Player.BLUE)):
            self.state.phase = 'play'; self.state.turn = Player.RED
        return True
    def legal_moves(self, player: Player):
        if self.state.phase != 'play': return []
        b=self.state.board; moves=[]
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                p=b.get((r,c))
                if p is None or p.owner!=player or not p.can_move(): continue
                for nb in b.neighbors((r,c)):
                    q=b.get(nb)
                    if q is None or q.owner!=player: moves.append(((r,c),nb))
        return moves
    def step(self, src: Coord, dst: Coord) -> Dict:
        assert self.state.phase=='play','not in play phase'
        player=self.state.turn
        opponent = Player.RED if player == Player.BLUE else Player.BLUE
        
        ev=self.state.board.move(player, src, dst)
        if not ev.get('ok'): return ev
        if ev['type']=='move': self.state.no_battle_counter+=1
        else:
            self.state.no_battle_counter=0
            if ev.get('flag_captured'):
                self.state.winner=player; self.state.end_reason='flag_captured'
        if self.state.winner is None and hasattr(self.state.board, 'has_legal_move'):
            if not self.state.board.has_legal_move(opponent):
                self.state.winner = player
                self.state.end_reason = 'no_moves_opponent'
            elif: not self.state.board.has_legal_move(player):
                self.state.winner = opponent
                self.state.end_reason = 'no_moves_self'
        
        if self.state.winner is None and self.state.no_battle_counter>=self.cfg.no_battle_draw_steps:
            self.state.end_reason='draw'
        if self.state.winner is None and self.state.end_reason is None:
            self.state.turn = Player.RED if player==Player.BLUE else Player.BLUE
        return ev
    def is_over(self) -> bool:
        return self.state.winner is not None or self.state.end_reason is not None
    def resign(self, player: Player):
        self.state.winner = Player.RED if player==Player.BLUE else Player.BLUE
        self.state.end_reason = 'resign'
