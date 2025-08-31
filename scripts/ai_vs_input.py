
#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
from minijunqi.constants import Player, PieceID, BOARD_W
from minijunqi.game import Game, GameConfig
from minijunqi.render import ascii_board, save_triple_latest
from minijunqi.replay import ReplayLogger
from minijunqi.ai.agent import Agent

def parse_coord(s: str):
    r,c = s.strip().split(','); return (int(r), int(c))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ai_side', choices=['RED','BLUE'], default='RED')
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--renders', type=str, default='renders')
    ap.add_argument('--replay_out', type=str, default='replays/ai_vs_input.json')
    args = ap.parse_args()
    os.makedirs(args.renders, exist_ok=True)
    os.makedirs(os.path.dirname(args.replay_out) or '.', exist_ok=True)

    ai_side = Player.RED if args.ai_side=='RED' else Player.BLUE
    game = Game(GameConfig())
    agent = Agent()
    if args.ckpt: agent.load(args.ckpt)
    logger = ReplayLogger(); logger.set_players('AI', 'INPUT')

    print('--- 部署阶段 ---')
    while game.state.phase == 'deploy':
        cur = game.state.turn
        if cur == ai_side:
            pid, rc = agent.select_deploy(game, cur)
            game.deploy(cur, pid, rc); logger.log_deploy(cur, pid, rc)
            print(f'AI 部署: {pid.name} @ {rc}')
        else:
            print('为“对手”部署。输入: piece_name_or_? r,c  （?表示未知，占位）')
            s = input('> ').strip()
            if not s: continue
            name, coord = s.split(); rc = parse_coord(coord)
            # 简化处理：未知用 BRIGADE 占位
            pid = getattr(PieceID, name, PieceID.BRIGADE) if name!='?' else PieceID.BRIGADE
            if not game.deploy(cur, pid, rc): print('非法位置，请重试'); continue
            logger.log_deploy(cur, pid, rc)
        print(ascii_board(game.state.board, viewer=ai_side, reveal_all=False))
        save_triple_latest(game.state.board, out_dir=args.renders, stem='board_latest')

    print('--- 行动阶段 ---')
    turn_idx=0
    while not game.is_over():
        save_triple_latest(game.state.board, out_dir=args.renders, stem='board_latest')
        if game.state.turn != ai_side:
            print('输入对手走子与结果： from_r,from_c to_r,to_c  结果: none/win/lose/draw')
            s = input('> ').strip()
            if not s: continue
            a,b,*rest = s.split()
            src = parse_coord(a); dst = parse_coord(b)
            ev = game.state.board.move(game.state.turn, src, dst)
            if not ev.get('ok'):
                p = game.state.board.get(src)
                game.state.board.set(dst, p); game.state.board.set(src, None)
                ev = {'ok':True, 'type':'move', 'forced':True}
            game.state.turn = ai_side
            logger.log_move(turn_idx, Player.BLUE if ai_side==Player.RED else Player.RED, src, dst, ev); turn_idx += 1
        else:
            src,dst = agent.select_move(game, ai_side)
            print(f'AI 建议: {src}->{dst}，在外部对局后输入结果：win/lose/draw/move')
            res = input('> ').strip()
            if res=='move':
                ev={'ok':True,'type':'move'}; _=game.state.board.move(ai_side, src, dst)
            elif res in ('win','lose','draw'):
                if res=='win':
                    p=game.state.board.get(src); game.state.board.set(dst,p); game.state.board.set(src,None)
                    ev={'ok':True,'type':'capture','result':'attacker'}
                elif res=='lose':
                    game.state.board.set(src,None); ev={'ok':True,'type':'capture','result':'defender'}
                else:
                    game.state.board.set(src,None); game.state.board.set(dst,None); ev={'ok':True,'type':'capture','result':'both'}
            else:
                ev={'ok':True,'type':'move'}; _=game.state.board.move(ai_side, src, dst)
            logger.log_move(turn_idx, ai_side, src, dst, ev); turn_idx += 1
            game.state.turn = Player.BLUE if ai_side==Player.RED else Player.RED
        print(ascii_board(game.state.board, viewer=ai_side, reveal_all=False))
    logger.set_outcome(game.state.winner, game.state.end_reason); logger.save(args.replay_out)
    print('结束：', game.state.end_reason, 'winner=', game.state.winner)
if __name__ == '__main__': main()
