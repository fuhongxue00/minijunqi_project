
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import os
from .constants import Player, PieceID, BOARD_H, BOARD_W
from .board import Board

def _load_font(size=28):
    # 直接用 WQY Micro Hei 的文件路径（Ubuntu）
    path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        # 兜底：再试环境变量；最后退回默认字体
        env_font = os.environ.get("MINIJUNQI_FONT")
        if env_font:
            return ImageFont.truetype(env_font, size)
        return ImageFont.load_default()

PIECE_SHORT = {
    PieceID.FLAG: '旗',
    PieceID.COMMANDER: '司',
    PieceID.ARMY: '军',
    PieceID.DIVISION: '师',
    PieceID.BRIGADE: '旅',
    PieceID.REGIMENT: '团',
    PieceID.BATTALION: '营',
    PieceID.BOMB: '炸',
    PieceID.UNKNOWN_ENEMY: '?'
}

def ascii_board(board: Board, viewer: Player, reveal_all: bool=False) -> str:
    obs = board.observe(viewer, reveal_all=reveal_all)
    rows = []
    for r in range(BOARD_H):
        row = []
        for c in range(BOARD_W):
            v = PieceID(obs[r][c])
            if v == PieceID.EMPTY:
                row.append('  .')
            elif v == PieceID.UNKNOWN_ENEMY:
                row.append('  ?')
            else:
                ch = PIECE_SHORT.get(v, '??')
                p = board.get((r,c))
                row.append(('R' if (p and p.owner==Player.RED) else 'B') + ch)
        rows.append(' '.join(row))
    header = f"视角: {viewer.name}  (reveal_all={reveal_all})"
    return header + "\n" + '\n'.join(rows)

def save_image(board: Board, path: str, viewer: Player, reveal_all: bool=False, title: Optional[str]=None):
    cell = 64
    pad = 20
    w = BOARD_W*cell + pad*2
    h = BOARD_H*cell + pad*2 + 30
    img = Image.new('RGB', (w,h), (240,240,240))
    draw = ImageDraw.Draw(img)
    # Grid
    for i in range(BOARD_H+1):
        y = pad + i*cell
        draw.line((pad, y, pad+BOARD_W*cell, y), fill=(0,0,0), width=2)
    for j in range(BOARD_W+1):
        x = pad + j*cell
        draw.line((x, pad, x, pad+BOARD_H*cell), fill=(0,0,0), width=2)
    # Pieces
    obs = board.observe(viewer, reveal_all=reveal_all)
    for r in range(BOARD_H):
        for c in range(BOARD_W):
            v = PieceID(obs[r][c])
            if v == PieceID.EMPTY: continue
            x0 = pad + c*cell
            y0 = pad + r*cell
            x1 = x0 + cell
            y1 = y0 + cell
            p = board.get((r,c))
            own = (p is not None and (reveal_all or p.owner == viewer) and p.owner==viewer)
            color = (220,80,80) if p.owner==Player.RED else (80,120,220)
            draw.rectangle((x0+4,y0+4,x1-4,y1-4), outline=(0,0,0), width=2, fill=color)
            ch = PIECE_SHORT.get(v, '?')
            try:
                # font = ImageFont.load_default()
                font = ImageFont.truetype("WenQuanYi Micro Hei",28)
            except:
                font = _load_font(28)
            bbox = draw.textbbox((0, 0), ch, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x0 + (cell-tw)//2, y0 + (cell-th)//2), ch, fill=(255,255,255), font=font)
    # Title
    t = title or f"{viewer.name} view (reveal_all={reveal_all})"
    draw.text((pad, pad + BOARD_H*cell + 6), t, fill=(0,0,0))
    img.save(path)
    return path

def save_triple_latest(board: Board, out_dir: str = '.', stem: str = 'board_latest'):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    paths['all'] = save_image(board, os.path.join(out_dir, f"{stem}_all.png"), viewer=Player.RED, reveal_all=True, title='All Info')
    paths['red'] = save_image(board, os.path.join(out_dir, f"{stem}_red.png"), viewer=Player.RED, reveal_all=False, title='RED View')
    paths['blue'] = save_image(board, os.path.join(out_dir, f"{stem}_blue.png"), viewer=Player.BLUE, reveal_all=False, title='BLUE View')
    return paths
