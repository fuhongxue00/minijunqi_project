
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from .constants import PieceID, Player

@dataclass
class Piece:
    pid: PieceID
    owner: Player
    revealed: bool = True
    def can_move(self) -> bool:
        return self.pid != PieceID.FLAG
    def __str__(self) -> str:
        return f"{self.owner.name}:{self.pid.name}"
