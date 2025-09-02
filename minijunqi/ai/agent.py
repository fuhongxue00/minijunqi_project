
# # -*- coding: utf-8 -*-
# from __future__ import annotations
# from ..constants import Player
# from ..game import Game
# from .policy import SharedPolicy

# class Agent:
#     """神经网络Agent封装。"""
#     def __init__(self, device='cpu', temperature: float=1.0):
#         self.policy = SharedPolicy(device=device)
#         self.temperature = temperature
#     def load(self, ckpt: str): self.policy.load(ckpt)
#     def _next_piece(self, game: Game, player: Player):
#         from ..constants import DEPLOY_SEQUENCE
#         pool = game.state.pools[player]
#         for pid in DEPLOY_SEQUENCE:
#             if pool.get(pid,0) > 0: return pid
#         return None
#     def select_deploy(self, game: Game, player: Player):
#         piece = self._next_piece(game, player)
#         rc = self.policy.select_deploy(game.state.board, player, piece, game.state.no_battle_counter, game.cfg.no_battle_draw_steps, temperature=self.temperature)
#         return piece, rc
#     def select_move(self, game: Game, player: Player):
#         return self.policy.select_move(game.state.board, player, side_to_move=game.state.turn, no_battle_counter=game.state.no_battle_counter, no_battle_limit=game.cfg.no_battle_draw_steps, temperature=self.temperature)

# class RandomAgent:
#     def select_deploy(self, game: Game, player: Player):
#         import random
#         from ..constants import BOARD_W, PieceID
#         home = [(r,c) for r in game.state.board.home_rows(player) for c in range(BOARD_W)]
#         # 旗子优先找合法位置
#         for (r,c) in home:
#             if game.state.board.can_place(player, PieceID.FLAG, (r,c)):
#                 return PieceID.FLAG,(r,c)
#         # 其余随机
#         # 找到任意剩余棋子
#         pool = game.state.pools[player]
#         for pid,cnt in pool.items():
#             if cnt>0:
#                 # 找一个合法位置
#                 random.shuffle(home)
#                 for (r,c) in home:
#                     if game.state.board.can_place(player, pid, (r,c)):
#                         return pid,(r,c)
#         return None,(0,0)
#     def select_move(self, game: Game, player: Player):
#         import random
#         legal = game.legal_moves(player)
#         return random.choice(legal) if legal else ((0,0),(0,0))


# -*- coding: utf-8 -*-
from __future__ import annotations
try:
    from ..constants import Player
except Exception:
    class Player:
        RED = 0
        BLUE = 1
try:
    from ..game import Game
except Exception:
    Game = object

from .policy import SharedPolicy

class Agent:
    """Neural agent wrapper with perspective canonicalization and 8-step history."""
    def __init__(self, device='cpu', temperature: float=1.0,net=None):
        
        if net is not None:
            self.policy = SharedPolicy(net=net,device=device)
        else:
            self.policy = SharedPolicy(device=device)
        self.temperature = temperature
    def load(self, ckpt: str): self.policy.load(ckpt)
    def reset(self): self.policy.reset_history()

    def _next_piece(self, game: Game, player: Player):
        try:
            from ..constants import DEPLOY_SEQUENCE
            pool = game.state.pools[player]
            for pid in DEPLOY_SEQUENCE:
                if pool.get(pid,0)>0:
                    return pid
            for pid,cnt in pool.items():
                if cnt>0: return pid
        except Exception:
            return None
        return None

    def select_deploy(self, game: Game, player: Player):
        pid = self._next_piece(game, player)
        if pid is None:
            return None, (-1,-1)
        rc,pc = self.policy.select_deploy(
            game.state.board, player, pid, no_battle_ratio=0.0, temperature=self.temperature
        )
        # pc为概率
        return pid, rc, pc

    def select_move(self, game: Game, player: Player):
        state = game.state
        try:
            ratio = float(state.no_battle_counter) / max(1, game.cfg.no_battle_draw_steps)
        except Exception:
            ratio = 0.0
        src, dst,pc,pt = self.policy.select_move(
            state.board, player, side_to_move=state.turn, no_battle_ratio=ratio, temperature=self.temperature
        )
        return src, dst,pc,pt
