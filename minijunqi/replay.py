
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import json, time, os, datetime
from .constants import BOARD_H, BOARD_W, PieceID, Player

class ReplayLogger:
    def __init__(self):
        self.data: Dict[str, Any] = {
            'version': 1,
            'board_size': [BOARD_H, BOARD_W],
            'piece_mapping': {int(k): k.name for k in PieceID},
            'players': {},
            'initial_deployments': {'RED': [], 'BLUE': []},
            'moves': [],
            'outcome': None,
            'meta': {'created': datetime.datetime.utcnow().isoformat()+'Z'}
        }
    def set_players(self, red_id: str='AI', blue_id: str='AI'):
        self.data['players'] = {'RED': {'id': red_id}, 'BLUE': {'id': blue_id}}
    def log_deploy(self, player: Player, pid: PieceID, rc: Tuple[int,int]):
        key = 'RED' if player==Player.RED else 'BLUE'
        self.data['initial_deployments'][key].append({'piece': pid.name, 'pos': list(rc)})
    def log_move(self, turn_idx: int, player: Player, src, dst, event: Dict):
        self.data['moves'].append({
            'turn': turn_idx,
            'player': 'RED' if player==Player.RED else 'BLUE',
            'from': list(src),
            'to': list(dst),
            'event': event,
            'ts': time.time()
        })
    def set_outcome(self, winner: Optional[Player], reason: Optional[str]):
        if winner is None:
            o = {'winner': None, 'reason': reason}
        else:
            o = {'winner': 'RED' if winner==Player.RED else 'BLUE', 'reason': reason}
        self.data['outcome'] = o
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    @staticmethod
    def load(path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
