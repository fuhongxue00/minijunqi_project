
#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse, json, time
from minijunqi.constants import Player, PieceID
from minijunqi.board import Board
from minijunqi.render import ascii_board, save_triple_latest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', type=str, required=True)
    ap.add_argument('--renders', type=str, default='renders')
    ap.add_argument('--step', action='store_true')
    ap.add_argument('--sleep', type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.renders, exist_ok=True)
    with open(args.replay, 'r', encoding='utf-8') as f:
        data = json.load(f)
    b = Board()
    for side in ['RED','BLUE']:
        from minijunqi.constants import Player as P
        for item in data['initial_deployments'][side]:
            pid = PieceID[item['piece']]; r,c = item['pos']
            b.place(P.RED if side=='RED' else P.BLUE, pid, (r,c))
    print('部署完成。'); print(ascii_board(b, Player.RED, reveal_all=True))
    save_triple_latest(b, out_dir=args.renders, stem='replay_step_0')
    t=1
    for mv in data['moves']:
        if args.step: input('回车下一手...')
        time.sleep(args.sleep)
        player = Player.RED if mv['player']=='RED' else Player.BLUE
        src = tuple(mv['from']); dst = tuple(mv['to'])
        ev = mv['event']
        if ev['type']=='move':
            p = b.get(src); b.set(dst,p); b.set(src,None)
        else:
            res = ev.get('result','both')
            if res=='attacker':
                p=b.get(src); b.set(dst,p); b.set(src,None)
            elif res=='defender':
                b.set(src,None)
            else:
                b.set(src,None); b.set(dst,None)
        print(ascii_board(b, Player.RED, reveal_all=True))
        save_triple_latest(b, out_dir=args.renders, stem=f'replay_step_latest')
        t+=1
    print('复盘结束。')
if __name__ == '__main__': main()
