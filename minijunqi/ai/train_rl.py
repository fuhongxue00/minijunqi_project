
# -*- coding: utf-8 -*-
# import sys, os
# sys.path.append(os.path.dirname(__file__))

import argparse, random
import torch, torch.nn as nn, torch.optim as optim
from tqdm import trange
from ..constants import Player, PieceID, BOARD_H, BOARD_W, DEFAULT_NO_BATTLE_DRAW, INITIAL_POOL
from ..game import Game, GameConfig
from .net import TinyUNet
from .policy import encode_obs

DIRS=[(-1,0),(1,0),(0,-1),(0,1)]

def random_deploy(g: Game, player: Player):
    cells=[(r,c) for r in g.state.board.home_rows(player) for c in range(BOARD_W)]
    random.shuffle(cells)
    # flag first
    flag=None
    for rc in cells:
        if g.state.board.can_place(player, PieceID.FLAG, rc): flag=rc; break
    g.deploy(player, PieceID.FLAG, flag)
    pool=dict(INITIAL_POOL); pool[PieceID.FLAG]-=1
    for pid,cnt in pool.items():
        for _ in range(cnt):
            for rc in cells:
                if g.state.board.can_place(player, pid, rc):
                    g.deploy(player, pid, rc); break

def play_episode(net: TinyUNet):
    g=Game(GameConfig())
    random_deploy(g, Player.RED); random_deploy(g, Player.BLUE)
    g.state.phase='play'; g.state.turn=Player.RED
    traj=[]  # (logp, player)
    while not g.is_over():
        player=g.state.turn
        x=encode_obs(g.state.board, player, side_to_move=player, no_battle_counter=g.state.no_battle_counter, no_battle_limit=DEFAULT_NO_BATTLE_DRAW, reveal_all=True).unsqueeze(0)
        out=net(x)
        # select piece
        ls=out['select_piece_logits'].squeeze(0)
        mask=torch.zeros(BOARD_H*BOARD_W)
        legal=g.legal_moves(player); idxs=set([mv[0][0]*BOARD_W+mv[0][1] for mv in legal])
        if not idxs: break
        for i in idxs: mask[i]=1
        prob_s=torch.softmax(ls.masked_fill(mask==0,-1e9), dim=0)
        src_idx=torch.distributions.Categorical(prob_s).sample()
        src=(int(src_idx)//BOARD_W, int(src_idx)%BOARD_W)
        # dir
        ld=out['move_dir_logits'].squeeze(0)
        dmask=torch.zeros(4)
        for k,(dr,dc) in enumerate(DIRS):
            rr,cc=src[0]+dr,src[1]+dc
            if 0<=rr<BOARD_H and 0<=cc<BOARD_W:
                q=g.state.board.get((rr,cc))
                if q is None or q.owner != player: dmask[k]=1
        prob_d=torch.softmax(ld.masked_fill(dmask==0,-1e9), dim=0)
        dir_idx=torch.distributions.Categorical(prob_d).sample()
        dr,dc=DIRS[int(dir_idx)]
        dst=(src[0]+dr, src[1]+dc)
        logp=torch.log(prob_s[src_idx]+1e-9)+torch.log(prob_d[dir_idx]+1e-9)
        traj.append((logp, player))
        ev=g.step(src,dst)
    if g.state.end_reason=='flag_captured':
        r=2.0 if g.state.winner==Player.RED else -1.0
    else:
        r=0.0
    returns=[]
    for logp,player in traj:
        gain=r if player==Player.RED else -r
        returns.append((logp,gain))
    return returns

def train(episodes, out, from_ckpt=None,lr_step=5, lr_gamma=0.9):
    net = TinyUNet() 
    if from_ckpt :
        net.load_state_dict(torch.load(from_ckpt)) 
    opt=optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)
    for _ in trange(episodes):
        traj=play_episode(net)
        if not traj: continue
        loss=0.0
        for logp,R in traj: loss=loss - logp*R
        opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
    import os
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    torch.save(net.state_dict(), out); print('saved:', out)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--out', type=str, default='checkpoints/rl.pt')
    ap.add_argument('--from_ckpt',type=str,default=None)
    args=ap.parse_args()
    train(args.episodes, args.out,from_ckpt=args.from_ckpt)
