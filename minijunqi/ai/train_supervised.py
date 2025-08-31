
# -*- coding: utf-8 -*-
import argparse, glob, json, random
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from ..constants import Player, PieceID, BOARD_H, BOARD_W, DEFAULT_NO_BATTLE_DRAW
from ..game import Game, GameConfig
from .net import TinyUNet
from .policy import encode_obs

def load_samples(paths):
    samples = []
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        g = Game(GameConfig())
        for side in ['RED','BLUE']:
            for item in data['initial_deployments'][side]:
                pid = PieceID[item['piece']]; r,c = item['pos']
                g.state.board.place(Player.RED if side=='RED' else Player.BLUE, pid, (r,c))
        g.state.phase='play'; g.state.turn=Player.RED
        no_battle=0
        for mv in data['moves']:
            player = Player.RED if mv['player']=='RED' else Player.BLUE
            src = tuple(mv['from']); dst = tuple(mv['to'])
            x = encode_obs(g.state.board, player, side_to_move=player, no_battle_counter=no_battle, no_battle_limit=DEFAULT_NO_BATTLE_DRAW, reveal_all=True)
            src_idx = src[0]*BOARD_W+src[1]
            dr = dst[0]-src[0]; dc = dst[1]-src[1]
            dmap = {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}
            if (dr,dc) not in dmap: continue
            dir_idx = dmap[(dr,dc)]
            samples.append((x, src_idx, dir_idx))
            ev = g.state.board.move(player, src, dst)
            if ev['type']=='move': no_battle+=1
            else: no_battle=0
    return samples

def train(paths, epochs, out):
    device='cpu'
    net = TinyUNet().to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    samples = load_samples(paths)
    random.shuffle(samples)
    for ep in range(epochs):
        total=0.0
        for x, src_idx, dir_idx in tqdm(samples, desc=f'epoch {ep+1}'):
            x=x.unsqueeze(0).to(device)
            out=net(x)
            loss = ce(out['select_piece_logits'], torch.tensor([src_idx]).to(device)) +                        ce(out['move_dir_logits'], torch.tensor([dir_idx]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f'epoch {ep+1} avg loss: {total/len(samples):.4f}')
    torch.save(net.state_dict(), out); print('saved:', out)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--replays', nargs='+', required=True)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--out', type=str, default='checkpoints/sup.pt')
    args = ap.parse_args()
    import glob
    paths=[]; [paths.extend(glob.glob(pat)) for pat in args.replays]
    train(paths, args.epochs, args.out)
