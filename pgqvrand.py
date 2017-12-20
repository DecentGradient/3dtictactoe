import warnings
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import tensorflow as tf
from common.benchmark import GameRate
import pickle as p
from common.game import Game
from common.dnn import ActorNetwork, CriticNetwork, PlayerTrainer

warnings.filterwarnings("ignore")
'''
This file contains the gameplay, passing state to the neural network, and receiving actions.
See common/game.py for the game environment.
See common/dnn.py for the neural network and training objects
 
'''
# ==========================
#   Training Parameters
# ==========================

# Number of test episodes
TEST_EPISODES = 500
# Max training steps
MAX_EPISODES = 20000
# Max episode length
MAX_EP_STEPS = 100
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.000001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.00001
# Discount factor
GAMMA = 0.4
# Soft target update param
TAU = 0.5

# ===========================
#   Utility Parameters
# ===========================

RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 1000
MINIBATCH_SIZE = 128


def train(sess, a1, c1, scaler):
    game = Game(verbose=False)
    player1 = PlayerTrainer(actor=a1, critic=c1, buffersize=BUFFER_SIZE, game=game, player=1, batch_size=MINIBATCH_SIZE,
                            gamma=GAMMA)

    sess.run(tf.global_variables_initializer())

    # Initialize target network weights
    a1.update_target_network()
    c1.update_target_network()

    episode = 0
    all_wins = []
    all_logs = []
    win_p1, comp1, bloc1 = 0, 0, 0
    win_p2, comp2, bloc2 = 0, 0, 0
    stat = []
    for i in range(MAX_EPISODES):

        episode += 1
        game.setup()

        ep_reward = 0
        ep_reward2 = 0
        reward2 = 0
        terminal = False
        for j in range(MAX_EP_STEPS):

            if not terminal:
                if episode < 7500:

                    move = game.random_space()
                    game.move(move, 1)
                    state, reward = game.step(player=1)
                else:
                    state, reward = player1.noisyMaxQMove()
                _, reward2 = game.step(player=2)
                ep_reward += reward
                ep_reward2 += reward2
                terminal = game.game_over

            if terminal:

                all_wins.append(game.game_over)
                log = game.setup()
                s = game.space
                all_logs.append(log)
                print(scaler, win_p1, comp1, bloc1, win_p2, comp2, bloc2, " | Episode", i, ep_reward, ep_reward2)

                if episode % 1000 == 0:
                    win_p1, comp1, bloc1, win_p2, comp2, bloc2 = test(sess, a1)
                    stat.append([episode, win_p1, comp1, bloc1, win_p2, comp2, bloc2])
                    df = pd.DataFrame(stat)
                    print(df)
                    plt.close('all')
                    xwinp = plt.plot(df[0], df[1], label="P1wins")
                    xcomp = plt.plot(df[0], df[2], label="P1Imm Compl")
                    xbloc = plt.plot(df[0], df[3], label="p1immbloc")
                    xwinp2 = plt.plot(df[0], df[4], label="P2wins")
                    xcomp2 = plt.plot(df[0], df[5], label="P2Imm Compl")
                    xbloc2 = plt.plot(df[0], df[6], label="p2immbloc")
                    plt.legend()
                    plt.ylim(0, 1)
                    plt.ylabel('percent')
                    plt.show(block=False)
                break
            else:
                move = game.random_space()
                game.move(move, 2)
                _, reward = game.step(player=1)
                terminal = game.game_over
                ep_reward2 += reward2
                ep_reward += reward

    return stat


def test(sess, actor1):
    game = Game(verbose=False)
    logs = []
    wins = []
    for i in range(TEST_EPISODES):
        game.setup()
        s = game.space
        terminal = False

        for j in range(MAX_EP_STEPS):
            if not terminal:
                a = actor1.predict(np.reshape(game.space, (1, *s.shape)))
                avail = game.avail()
                # noinspection PyPep8Naming
                availQ = {}

                for x in avail:
                    availQ[x] = a[0][x]
                action = max(availQ, key=availQ.get)

                game.move(action, 1)
                s2, r = game.step(1)
                terminal = game.game_over
                info = None
            if terminal:
                wins.append(game.game_over)
                log = game.setup()
                logs.append(log)
                s = game.space
                break
            else:
                action = game.random_space()

                game.move(action, 2)
                s2, r = game.step(1)
                terminal = game.game_over
                info = None

    c = Counter(wins)
    r = GameRate(verbose=False, list=logs, player=1, opponent=2)
    r2 = GameRate(verbose=False, list=logs, player=2, opponent=1)
    bloc1, bloc2 = 0, 0
    r.check_games()
    r2.check_games()
    win_p1 = c[1] / (TEST_EPISODES - 1)
    print("1win percentage", win_p1)
    if r.completions + r.missed_completions > 0:
        comp1 = r.completions / (r.completions + r.missed_completions)
    else:
        comp1 = 0
    print("1immediate completions", comp1)
    if r.blocks + r.missed_blocks > 0:
        bloc1 = r.blocks / (r.blocks + r.missed_blocks)
    win_p2 = c[2] / (TEST_EPISODES - 1)
    print("2win percentage", win_p2)
    if r2.completions + r2.missed_completions > 0:

        comp2 = r2.completions / (r2.completions + r2.missed_completions)
    else:
        comp2 = 0
    print("2immediate completions", comp2)
    if r2.blocks + r2.missed_blocks > 0:
        bloc2 = r2.blocks / (r2.blocks + r2.missed_blocks)
    return win_p1, comp1, bloc1, win_p2, comp2, bloc2


def test_log(wins, log):
    c = Counter(wins)
    r = GameRate(verbose=False, list=log)

    r.check_games()
    win_p = c[1] / (TEST_EPISODES - 1)
    print("win percentage", win_p)
    comp = r.completions / (r.completions + r.missed_completions)
    print("immediate completions", comp)
    return win_p, comp


def main(_):
    with tf.Session() as sess:
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        state_dim = 27
        action_dim = 27
        action_bound = 27
        # Ensure action bound is symmetric

        results = []
        tot_stat = []
        for i in range(10):
            p1 = dict()
            p1["a"] = ActorNetwork(sess, state_dim, action_dim, action_bound,
                                   ACTOR_LEARNING_RATE, TAU, vscope="p1a")
            p1["c"] = CriticNetwork(sess, state_dim, action_dim,
                                    CRITIC_LEARNING_RATE, TAU, p1["a"].get_num_trainable_vars(), vscope="p1c")

            # p2 = {}
            # p2["a"] = ActorNetwork(sess, state_dim, action_dim, action_bound,
            #                        ACTOR_LEARNING_RATE, TAU, vscope="p2a")
            # p2["c"] = CriticNetwork(sess, state_dim, action_dim,
            #                         CRITIC_LEARNING_RATE, TAU, p2["a"].get_num_trainable_vars(), vscope="p2c")

            stat = train(sess, p1["a"], p1["c"], i)  # , p2["a"], p2["c"])
            tot_stat.append(stat)
            # results.append([win_p,comp,epi])

    with open('data/pgvrdd7500', 'wb') as f:
        p.dump(tot_stat, f)
        # print('done',results)
        # print('stats',tot_stat)


if __name__ == '__main__':
    tf.app.run()
