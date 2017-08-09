from time import localtime
from collections import Counter
import numpy as np
import tensorflow as tf
import pickle as p

from common.game import Game
from common.q import QNetwork ,QPlayerTrainer
from common.benchmark import GameRate


train_episodes = 2000000          # max number of episodes to learn from
max_steps = 100                # max steps in an episode
gamma = 0.7

TEST_EPISODES = 5000

# future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.00001            # exponential decay rate for exploration prob

# Network parameters
hidden_size = 200               # number of units in each Q-network hidden layer
learning_rate = 0.01         # Q-network learning rate

# Memory parameters
memory_size = 1000            # memory capacity
batch_size = 50                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory


def train(mainQN,sess):
    winp, comp, blocp = 0, 0, 0
    saver = tf.train.Saver()
    game = Game(verbose=False)
    wins=[]
    logs=[]
    epi_log = []
    #memory = Memory(max_size=memory_size)

    trainer = QPlayerTrainer(qnet=mainQN,buffersize=memory_size,game=game,player=1,batch_size=batch_size,gamma=gamma,sess=sess)

    # Now train with experiences

    rewards_list = []
    loss = False
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        step = 0
        for ep in range(1, train_episodes+1):
            total_reward = 0
            t = 0
            explore_p=0
            while t < max_steps:
                if not game.game_over:
                    step += 1
                    #explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * step)
                    #if explore_p > np.random.rand():
                        # Make a random action
                    #    next_state, reward, loss = trainer.randomMove()
                    #else:
                    next_state, reward,loss = trainer.noisyMaxQMove()
                    total_reward += reward
                if game.game_over:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
                    t = max_steps

                    if loss:
                        print(winp,comp,blocp,'Episode: {}'.format(ep),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),explore_p)

                    rewards_list.append((ep, total_reward))

                    # Add experience to memory

                    wins.append(game.game_over)

                    log = game.setup()
                    logs.append(log)

                    if ep % 10000 == 0:
                       #print(wins[-100:],logs[-100:])
                       #exit(0)

                        time = str(localtime())
                        saver.save(sess, "chk/dqvrand/" + time + ".ckpt")
                        winp, comp, blocp = test(mainQN,sess)
                        epi_log.append([ep,winp,comp,blocp ])



                    state = game.space
                else:
                    state = next_state
                    t += 1
                space = game.random_space()
                game.move(space,2)
                _,reward = game.step(player=2)
                total_reward += reward
        time = str(localtime())
        saver.save(sess, "chk/dqvrand/"+time+".ckpt")
        with open('data/epi2', 'wb') as f:
            p.dump(epi_log, f)

def test(mainQN,sess):

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('chk/dqvrand'))
    e = 0
    logs = []
    wins = []
    game = Game(verbose=False)
    while e <=1000:
        e+=1
        if not game.game_over:
            state = game.space
            feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
            As = sess.run(mainQN.output, feed_dict=feed)
            avail = game.avail()

            availQ = {}
            for k in avail:
                availQ[k]=As[0][k]
            action = max(availQ,key=availQ.get)
            game.move(action,1)
            game.step(1)

        if game.game_over:
            wins.append(game.game_over)
            log = game.setup()
            logs.append(log)
            continue
        move = game.random_space()
        game.move(move,2)
        game.step(2)

    win, comp, bloc = 0, 0, 0
    c = Counter(wins)
    r = GameRate(verbose=False, list=logs,player=1,opponent=2)

    r.check_games()
    #print(r,c)

    win= c[1] / len(wins)
    print("win percentage",win)
    if (r.completions + r.missed_completions)>0:
        comp =  r.completions / (r.completions + r.missed_completions)
    print("immediate completions",comp)
    if (r.blocks + r.missed_blocks)>0:
        bloc = r.blocks / (r.blocks + r.missed_blocks)
    print("blocks",bloc)
    #exit(1)
    if win ==0.0:
        print(wins)
        exit(1)
    return win,comp,bloc







def main(_):
    with tf.Session() as sess:

        mainQN = QNetwork(name='player1q', hidden_size=hidden_size, learning_rate=learning_rate)
        # p2QN = QNetwork(name='p2', hidden_size=hidden_size, learning_rate=learning_rate)

        train(mainQN=mainQN,sess=sess)
        wins,comp,bloc               = test(mainQN=mainQN,sess=sess)



if __name__ == '__main__':
    tf.app.run()