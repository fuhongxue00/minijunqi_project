
# -*- coding: utf-8 -*-
from enum import IntEnum

BOARD_H = 6
BOARD_W = 6

class Player(IntEnum):
    RED = 0
    BLUE = 1

class PieceID(IntEnum):
    EMPTY = 0
    FLAG = 1
    COMMANDER = 2
    ARMY = 3
    DIVISION = 4
    BRIGADE = 5
    REGIMENT = 6
    BATTALION = 7
    BOMB = 8
    UNKNOWN_ENEMY = 9  # 用于观测中的未知敌子（棋盘内部不放这个）

INITIAL_POOL = {
    PieceID.FLAG: 1,
    PieceID.COMMANDER: 1,
    PieceID.ARMY: 1,
    PieceID.DIVISION: 2,
    PieceID.BRIGADE: 2,
    PieceID.REGIMENT: 2,
    PieceID.BATTALION: 2,
    PieceID.BOMB: 1,
}

STRENGTH = {
    PieceID.FLAG: -1,
    PieceID.COMMANDER: 7,
    PieceID.ARMY: 6,
    PieceID.DIVISION: 5,
    PieceID.BRIGADE: 4,
    PieceID.REGIMENT: 3,
    PieceID.BATTALION: 2,
    PieceID.BOMB: 1,
}

FLAG_ALLOWED_COLS = (1, 4)
DEFAULT_NO_BATTLE_DRAW = 40

# 预定义部署顺序（旗子先放，其余按固定顺序展开多份）
DEPLOY_SEQUENCE = (
    [PieceID.FLAG] +
    [PieceID.COMMANDER] +
    [PieceID.ARMY] +
    [PieceID.DIVISION]*2 +
    [PieceID.BRIGADE]*2 +
    [PieceID.REGIMENT]*2 +
    [PieceID.BATTALION]*2 +
    [PieceID.BOMB]
)
