
#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
from minijunqi.constants import Player, PieceID, BOARD_W
from minijunqi.game import Game, GameConfig
from minijunqi.render import ascii_board, save_triple_latest
from minijunqi.replay import ReplayLogger
from minijunqi.ai.agent import Agent

NAME2PID = {name: getattr(PieceID, name) for name in PieceID.__members__}
def parse_coord(s: str):
    r,c = s.strip().split(','); return (int(r), int(c))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human_side', choices=['RED','BLUE'], default='RED')
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--renders', type=str, default='renders')
    ap.add_argument('--replay_out', type=str, default='replays/ai_vs_human.json')
    ap.add_argument('--human-random-deploy',
                    action=argparse.BooleanOptionalAction,
                    default=True,
                    help='人类玩家使用随机布局（默认开启）。如需手动布局，使用 --no-human-random-deploy')
    args = ap.parse_args()
    os.makedirs(args.renders, exist_ok=True)
    os.makedirs(os.path.dirname(args.replay_out) or '.', exist_ok=True)

    human = Player.RED if args.human_side=='RED' else Player.BLUE
    ai_side = Player.BLUE if human==Player.RED else Player.RED
    game = Game(GameConfig())
    agent = Agent(); 
    if args.ckpt: agent.load(args.ckpt)
    logger = ReplayLogger(); logger.set_players('HUMAN', 'AI')
    while game.state.phase == 'deploy':
        cur = game.state.turn
        if cur == human and args.human_random_deploy:
            # 连续自动下子直到轮到 AI 或进入行棋阶段
            did_any = False
            while game.state.phase == 'deploy' and game.state.turn == human:
                pid, rc = agent.select_deploy(game, human)
                ok = game.deploy(human, pid, rc)
                if ok:
                    logger.log_deploy(human, pid, rc)
                    did_any = True
                else:
                    break
            if did_any:
                print('（已为人类玩家完成随机布局）')
        elif cur == human:
            print('你的回合-部署。输入: PIECE_NAME r,c  例如: COMMANDER 5,1')
            print('可用棋子：', {k.name:v for k,v in game.state.pools[cur].items() if v>0})
            s = input('> ').strip(); 
            if not s: continue
            name, coord = s.split(); pid = NAME2PID[name]; rc = parse_coord(coord)
            if game.deploy(cur, pid, rc): logger.log_deploy(cur, pid, rc)
            else: print('非法位置或该棋子已用完')
        else:
            pid, rc = agent.select_deploy(game, cur)
            game.deploy(cur, pid, rc); logger.log_deploy(cur, pid, rc)
            print(f'AI 部署: {pid.name} @ {rc}')
        save_triple_latest(game.state.board, out_dir=args.renders, stem='board_latest')
        print(ascii_board(game.state.board, viewer=human, reveal_all=False))
    print('部署完毕，开始对局。')
    turn_idx=0
    while not game.is_over():
        print('（分析用-全信息）\n' + ascii_board(game.state.board, viewer=human, reveal_all=True))
        print('（对弈用-局部视角）\n' + ascii_board(game.state.board, viewer=human, reveal_all=False))
        save_triple_latest(game.state.board, out_dir=args.renders, stem='board_latest')
        if game.state.turn == human:
            print('你的回合-行棋。输入: r1,c1 r2,c2')
            s = input('> ').strip()
            if not s: continue
            a,b = s.split(); src = parse_coord(a); dst = parse_coord(b)
            ev = game.step(src,dst)
            if not ev.get('ok'): print('非法走法')
            else: logger.log_move(turn_idx, human, src, dst, ev); turn_idx+=1
        else:
            src,dst = agent.select_move(game, game.state.turn)
            ev = game.step(src,dst); logger.log_move(turn_idx, ai_side, src, dst, ev); turn_idx+=1
            print(f'AI: {src}->{dst}  事件={ev}')
    logger.set_outcome(game.state.winner, game.state.end_reason); logger.save(args.replay_out)
    print('对局结束：', game.state.end_reason, 'winner=', game.state.winner)
if __name__ == '__main__': main()
