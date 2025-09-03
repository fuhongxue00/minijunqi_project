
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
    ap.add_argument('--count',type=int, default=1)
    ap.add_argument('--ckpt_red', type=str, default=None)
    ap.add_argument('--ckpt_blue', type=str, default=None)
    ap.add_argument('--step', action='store_true')
    ap.add_argument('--sleep', type=float, default=0)
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
    
    # 一些统计信息
    global_red_win = 0 #红色赢的次数
    global_blue_win = 0
    global_draw = 0 #平局次数
    global_sum_stepnum = 0 #总移动数，最终要除以对局场次得到平均每局步数
    global_red_attack = 0 #红色主动吃子次数
    global_red_beat = 0 #红色成功吃子次数
    global_red_song = 0 #红色送子次数
    global_blue_attack = 0
    global_blue_beat = 0
    global_blue_song = 0
    
    for i in range(args.count):
        game = Game(GameConfig())
        red.reset();blue.reset()
        cur = Player.RED
        while game.state.phase == 'deploy':
            agent = red if cur==Player.RED else blue
            piece, rc ,_= agent.select_deploy(game, cur)
            ok = game.deploy(cur, piece, rc)
            if ok: logger.log_deploy(cur, piece, rc)
            cur = game.state.turn

            # 在部署阶段也加上可视化
            time.sleep(args.sleep)
            if args.count==1:
                print(ascii_board(game.state.board, viewer=Player.RED, reveal_all=True,is_deploy=True))
                save_triple_latest(game.state.board, out_dir=args.renders, stem='board_latest',is_deploy=True)
        print('部署完毕，开始对局。')
        red.reset();blue.reset()
        turn_idx=0
        while not game.is_over():
            if args.step: input('回车走一步...')
            time.sleep(args.sleep)
            player = game.state.turn
            agent = red if player==Player.RED else blue
            src,dst,_,_ = agent.select_move(game, player)
            ev = game.step(src,dst)
            if ev['type'] == 'capture':
                # 计入统计信息
                if player == Player.RED:
                    global_red_attack += 1
                    if ev['result'] == 'attacker':
                        global_red_beat += 1
                    elif ev['result'] == 'defender':
                        global_red_song += 1
                else:
                    global_blue_attack += 1
                    if ev['result'] == 'attacker':
                        global_blue_beat += 1
                    elif ev['result'] == 'defender':
                        global_blue_song += 1
            logger.log_move(turn_idx, player, src, dst, ev); turn_idx += 1
            if args.count==1:   
                print(ascii_board(game.state.board, viewer=Player.RED, reveal_all=True))
                save_triple_latest(game.state.board, out_dir=args.renders, stem='board_latest')
        logger.set_outcome(game.state.winner, game.state.end_reason)
        logger.save(args.replay_out)
        print('对局结束：', game.state.end_reason, 'winner=', game.state.winner,'总步数:',turn_idx)
        if game.state.winner == Player.RED:
            global_red_win +=1
        elif game.state.winner == Player.BLUE:
            global_blue_win += 1
        else:
            global_draw += 1 #平局次数
        global_sum_stepnum += turn_idx
    



    print('===========\n')
    print('红方获胜场次',global_red_win)
    print('蓝方获胜场次',global_blue_win)
    print('平局场次',global_draw)
    print('场均总步数',global_sum_stepnum/args.count)
    print('场均红色主动试图吃子次数',global_red_attack/args.count)
    print('场均红色吃子成功',global_red_beat/args.count)
    print('场均红色送子次数',global_red_song/args.count)
    print('场均蓝色主动试图吃子次数',global_blue_attack/args.count)
    print('场均蓝色吃子成功',global_blue_beat/args.count)
    print('场均蓝色送子',global_blue_song/args.count)
if __name__ == '__main__': main()
