
#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse, time, random
from minijunqi.constants import Player, PieceID, BOARD_W
from minijunqi.game import Game, GameConfig
from minijunqi.render import ascii_board, save_triple_latest
from minijunqi.replay import ReplayLogger
from minijunqi.ai.agent import Agent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_red', type=str, default=None)
    ap.add_argument('--ckpt_blue', type=str, default=None)
    ap.add_argument('--step', action='store_true')
    ap.add_argument('--sleep', type=float, default=0.1)
    ap.add_argument('--replay_out', type=str, default='replays/ai_vs_ai.json')
    ap.add_argument('--renders', type=str, default='renders')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.replay_out) or '.', exist_ok=True)
    os.makedirs(args.renders, exist_ok=True)

    game = Game(GameConfig())
    logger = ReplayLogger()
    logger.set_players('AI_R', 'AI_B')

    red = Agent(); blue = Agent()
    if args.ckpt_red: red.load(args.ckpt_red)
    if args.ckpt_blue: blue.load(args.ckpt_blue)
    cur = Player.RED
    while game.state.phase == 'deploy':
        agent = red if cur==Player.RED else blue
        piece, rc = agent.select_deploy(game, cur)
        ok = game.deploy(cur, piece, rc)
        if ok: logger.log_deploy(cur, piece, rc)
        cur = game.state.turn

        # 在部署阶段也加上可视化
        time.sleep(args.sleep)
        print(ascii_board(game.state.board, viewer=Player.RED, reveal_all=True,is_deploy=True))
        save_triple_latest(game.state.board, out_dir=args.renders, stem='board_latest',is_deploy=True)
    print('部署完毕，开始对局。')
    turn_idx=0
    while not game.is_over():
        if args.step: input('回车走一步...')
        time.sleep(args.sleep)
        player = game.state.turn
        agent = red if player==Player.RED else blue
        src,dst = agent.select_move(game, player)
        ev = game.step(src,dst)
        logger.log_move(turn_idx, player, src, dst, ev); turn_idx += 1
        print(ascii_board(game.state.board, viewer=Player.RED, reveal_all=True))
        save_triple_latest(game.state.board, out_dir=args.renders, stem='board_latest')
    logger.set_outcome(game.state.winner, game.state.end_reason)
    logger.save(args.replay_out)
    print('对局结束：', game.state.end_reason, 'winner=', game.state.winner)
if __name__ == '__main__': main()
