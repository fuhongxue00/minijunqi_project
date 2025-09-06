
# -*- coding: utf-8 -*-
# import sys, os
# sys.path.append(os.path.dirname(__file__))

import argparse, random
import torch, torch.nn as nn, torch.optim as optim
from tqdm import trange
from ..constants import Player, PieceID, BOARD_H, BOARD_W, DEFAULT_NO_BATTLE_DRAW, INITIAL_POOL
from ..game import Game, GameConfig
from ..render import ascii_board
# from .net import TinyUNet
from .net import PolicyNet
from .policy import SharedPolicy
from .agent import Agent
# from .policy import encode_obs

DIRS=[(-1,0),(1,0),(0,-1),(0,1)]

# def random_deploy(g: Game, player: Player):
#     cells=[(r,c) for r in g.state.board.home_rows(player) for c in range(BOARD_W)]
#     random.shuffle(cells)
#     # flag first
#     flag=None
#     for rc in cells:
#         if g.state.board.can_place(player, PieceID.FLAG, rc): flag=rc; break
#     g.deploy(player, PieceID.FLAG, flag)
#     pool=dict(INITIAL_POOL); pool[PieceID.FLAG]-=1
#     for pid,cnt in pool.items():
#         for _ in range(cnt):
#             for rc in cells:
#                 if g.state.board.can_place(player, pid, rc):
#                     g.deploy(player, pid, rc); break

def play_episode(net:PolicyNet):
    traj=[]  # (logp, player)
    g=Game(GameConfig())
    red_agent = Agent(net=net)
    blue_agent = Agent(net=net)
    red_agent.reset()
    blue_agent.reset()
    # random_deploy(g, Player.RED); random_deploy(g, Player.BLUE)
    # g.state.phase='play'; g.state.turn=Player.RED
    
    while g.state.phase == "deploy":
        player = g.state.turn
        play_agent = red_agent if player == Player.RED else blue_agent
        pid,rc,pc = play_agent.select_deploy(g,player)
        r,c=rc
        idx = r*BOARD_W+c
        logp=torch.log(pc[idx]+1e-9)
        result = 'deploy'
        traj.append((logp, player,result))
        ev=g.deploy(player,pid,rc)
        # print(ev)
        # print('deploy',pid,rc)
        # print('pc',pc)
        # print(ascii_board(g.state.board, viewer=Player.RED, reveal_all=True))
    red_agent.reset()
    blue_agent.reset()
    while not g.is_over():
        player = g.state.turn
        play_agent = red_agent if player == Player.RED else blue_agent
        # 起点，终点，起点概率张量，重点概率张量
        src,dst,ps,pt = play_agent.select_move(g, player)
        s_r,s_c = src
        src_idx = s_r*BOARD_W+s_c
        t_r,t_c = dst
        dir_idx = t_r*BOARD_W+t_c
        logp = torch.log(ps[src_idx]+1e-9)+torch.log(pt[dir_idx]+1e-9)
        
        ev=g.step(src,dst)
        result = ev.get('result','default_justmove')
        # print(ev)
        traj.append((logp, player,result))
        # print(ascii_board(g.state.board, viewer=Player.RED, reveal_all=True))
    if g.state.winner is not None:
        # r=10.0 if g.state.winner==Player.RED else -10.0
        pass
    elif g.state.end_reason=='draw':
        # r=0
        pass
    else:
        print(f"end_reason:{g.end_reason}")
        raise ValueError("出现了未知的结束原因")
    returns=[]
    for index,(logp,player,result) in enumerate(traj):
        gain = 0
        if player == g.state.winner:
            gain+=10
        else:
            oppo = Player.RED if player == Player.BLUE else Player.BLUE
            if g.state.winner == oppo:
                gain-=10
        if g.state.end_reason=='draw':
            # gain -= 1
            pass
        if result == 'attacker':
            pass
            # gain += 10 * index/len(traj)
            # print('主动吃子 ')
        elif result == 'defender':
            pass
            # gain -= 2
            # print('送子，r-=2')
        returns.append((logp,gain))
    return returns

def train(episodes, out, from_ckpt=None,lr_step=2, lr_gamma=0.9):
    net = PolicyNet()
    if from_ckpt :
        net.load_state_dict(torch.load(from_ckpt)) 
    opt=optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)
    for _ in trange(episodes):
        traj=play_episode(net)
        if not traj: continue
        loss=0.0
        for logp,R in traj:
            # print('logp:',logp.item(),'R:',R)
            loss=loss - logp*R
        opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
        # print('loss:',loss.item())
        # print('lr:',scheduler.get_last_lr()[0])
    import os
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    torch.save(net.state_dict(), out); print('saved:', out)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=30)
    ap.add_argument('--out', type=str, default='checkpoints/rl.pt')
    ap.add_argument('--from_ckpt',type=str,default=None)
    args=ap.parse_args()
    train(args.episodes, args.out,from_ckpt=args.from_ckpt)
