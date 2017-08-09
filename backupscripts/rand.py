from time import localtime
from collections import Counter
import numpy as np
import tensorflow as tf
import pickle as p

from common.game import Game
from common.q import QNetwork ,QPlayerTrainer
from common.benchmark import GameRate


def test():

    #saver = tf.train.Saver()
    #saver.restore(sess, tf.train.latest_checkpoint('chk/dqvrand'))
    e = 0
    epilog =[]
    logs = []
    wins = []
    game = Game(verbose=False)
    while e <=1000:
        e+=1
        if not game.game_over:

            action = game.random_space()
            game.move(action,1)
            game.step(1)

        if game.game_over:
            wins.append(game.game_over)
            log = game.setup()
            logs.append(log)
            if e % 100 == 0:
                win_p1, comp1, bloc1, win_p2, comp2, bloc2 = 0, 0, 0, 0, 0, 0

                c = Counter(wins)
                r = GameRate(verbose=False, list=logs, player=1, opponent=2)
                r2 = GameRate(verbose=False, list=logs, player=2, opponent=1)

                r.check_games()
                r2.check_games()
                win_p1 = c[1] / len(wins)
                print("1win percentage", win_p1)
                if r.completions + r.missed_completions > 0:
                    comp1 = r.completions / (r.completions + r.missed_completions)
                else:
                    comp1 = 0
                print("1immediate completions", comp1)
                if r.blocks + r.missed_blocks > 0:
                    bloc1 = r.blocks / (r.blocks + r.missed_blocks)
                win_p2 = c[2] / len(wins)
                print("2win percentage", win_p2)
                if r2.completions + r2.missed_completions > 0:

                    comp2 = r2.completions / (r2.completions + r2.missed_completions)
                else:
                    comp2 = 0
                print("2immediate completions", comp2)
                if r2.blocks + r2.missed_blocks > 0:
                    bloc2 = r2.blocks / (r2.blocks + r2.missed_blocks)
                epilog.append([e,win_p1,comp1,bloc1,win_p2,comp2,bloc2])
            continue
        move = game.random_space()
        game.move(move,2)
        game.step(2)

    return epilog







def main(_):

    epi_log = test()


    with open('data/rand2', 'wb') as f:
        p.dump(epi_log, f)
if __name__ == '__main__':
    tf.app.run()