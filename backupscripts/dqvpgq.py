from time import localtime

import numpy as np
import tensorflow as tf

from common.game import Game
from common.q import QNetwork, Memory

train_episodes = 1000000          # max number of episodes to learn from
max_steps = 100                # max steps in an episode
gamma = 0.2                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.000001            # exponential decay rate for exploration prob

# Network parameters
hidden_size = 200               # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 50                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory
tf.reset_default_graph()
mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate)
#p2QN = QNetwork(name='p2', hidden_size=hidden_size, learning_rate=learning_rate)
game = Game(verbose=False)
memory = Memory(max_size=memory_size)
saver = tf.train.Saver()
action = game.random_space()
game.move(action, 1)
state , reward = game.step()

space = game.random_space()
game.move(space, 2)

for ii in range(pretrain_length):

    action = game.random_space()
    game.move(action,1)
    next_state , reward = game.step()

    if game.game_over:
        next_state=np.zeros(state.shape)
        memory.add((state, action, reward, next_state))

        game.setup()

        action = game.random_space()
        game.move(action,1)
        state , reward = game.step()
    else:
        memory.add((state,action,reward,next_state))
        state = next_state
    inv_action = game.random_space()
    game.move(inv_action,2)

# Now train with experiences

rewards_list = []
loss = False
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    step = 0
    for ep in range(1, train_episodes):
        total_reward = 0
        t = 0
        while t < max_steps:
            if not game.game_over:
                step += 1
                explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * step)
                if explore_p > np.random.rand():
                    # Make a random action
                    action = game.random_space()
                else:
                    # Get action from Q-network
                    feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    #print(Qs)
                    avail = game.avail()
                    availQ = {}

                    for i in avail:
                        availQ[i]=Qs[0][i]
                    action = max(availQ,key=availQ.get)
                game.move(action, 1)
                next_state, reward = game.step()
                total_reward += reward
            if game.game_over:
                # the episode ends so no next state
                next_state = np.zeros(state.shape)
                t = max_steps
                if loss:
                    print('Episode: {}'.format(ep),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_p))
                rewards_list.append((ep, total_reward))

                # Add experience to memory
                memory.add((game.space, action, reward, next_state))

                game.setup()
                action = game.random_space()
                game.move(action, 1)
                state, reward = game.step()
            else:
                memory.add((game.space, action, reward, next_state))
                state = next_state
                t += 1
            space = game.random_space()
            game.move(space,2)
            _,reward = game.step()
            total_reward += reward

            batch = memory.sample(batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])

            # Train network
            target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})

            # Set target_Qs to 0 for states where episode ends
            #episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
            #target_Qs[episode_ends] = (0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

            targets = rewards + gamma * np.max(target_Qs, axis=1)

            loss, _ = sess.run([mainQN.loss, mainQN.opt],
                               feed_dict={mainQN.inputs_: states,
                                          mainQN.targetQs_: targets,
                                          mainQN.actions_: actions})
    time = str(localtime())
    saver.save(sess, "checkpoints/"+time+".ckpt")