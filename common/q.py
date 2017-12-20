import tensorflow as tf
import numpy as np

'''
Adapted from Reinforment Learning in Udacity Deep Learning Foundations Nanodegree
'''
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=27,
                 action_size=27, hidden_size=100,
                 name='QNetwork'):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, hidden_size)


            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc3, action_size,
                                                            activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


from collections import deque


class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]
    def size(self):
        return len(self.buffer)

class QPlayerTrainer(object):
    def __init__(self,qnet,buffersize,game,player,batch_size,gamma,sess):

        self.qnet =qnet
        self.sess = sess
        self.memory = Memory(buffersize)
        self.game =game
        self.player = player
        self.batch_size = batch_size
        self.gamma = gamma
        self.sess.run(tf.global_variables_initializer())


    def noisyMaxQMove(self):

        loss=0
        state = self.game.space
        feed = {self.qnet.inputs_: state.reshape((1, *state.shape))}
        As = self.sess.run(self.qnet.output, feed_dict=feed)
        avail = self.game.avail()
        availQ = {}
        availP = []
        for k in avail:
            availQ[k] = As[0][k]
            availP.append(As[0][k])
        # if sum(availP)> 0:
        availP = np.array(availP)

        availP = [round(i, 5) if i >= 0 else (-.001 * round(i, 5)) for i in availP]
        availNorm = [i / sum(availP) for i in availP]

        action = np.random.choice(avail, p=availNorm)

        self.game.move(action,self.player)
        next_state, reward = self.game.step(self.player)

        self.memory.add((state, action, reward, next_state))
        if self.memory.size() > self.batch_size:
            batch = self.memory.sample(self.batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])

            # Train network
            target_Qs = self.sess.run(self.qnet.output, feed_dict={self.qnet.inputs_: next_states})

            # Set target_Qs to 0 for states where episode ends
            # episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
            # target_Qs[episode_ends] = (0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

            targets = rewards + self.gamma * np.max(target_Qs, axis=1)

            loss, _ = self.sess.run([self.qnet.loss, self.qnet.opt],
                               feed_dict={self.qnet.inputs_: states,
                                          self.qnet.targetQs_: targets,
                                          self.qnet.actions_: actions})
        return self.game.space , reward,loss

    def randomMove(self):
        loss=0

        state = self.game.space

        action = self.game.random_space()

        self.game.move(action,self.player)
        next_state, reward = self.game.step(self.player)

        self.memory.add((state, action, reward, next_state))
        if self.memory.size() > self.batch_size:
            batch = self.memory.sample(self.batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])

            # Train network
            target_Qs = self.sess.run(self.qnet.output, feed_dict={self.qnet.inputs_: next_states})

            # xSet target_Qs to 0 for states where episode ends
            # xepisode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
            # xtarget_Qs[episode_ends] = (0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

            targets = rewards + self.gamma * np.max(target_Qs, axis=1)

            loss, _ = self.sess.run([self.qnet.loss, self.qnet.opt],
                               feed_dict={self.qnet.inputs_: states,
                                          self.qnet.targetQs_: targets,
                                          self.qnet.actions_: actions})
        return self.game.space , reward,loss

    def maxQMove(self):

        loss=0
        state = self.game.space
        feed = {self.qnet.inputs_: state.reshape((1, *state.shape))}
        As = self.sess.run(self.qnet.output, feed_dict=feed)
        avail = self.game.avail()
        availQ = {}

        for k in avail:
            availQ[k] = As[0][k]

        action = max(availQ, key=availQ.get)

        self.game.move(action,self.player)
        next_state, reward = self.game.step(self.player)

        self.memory.add((state, action, reward, next_state))
        if self.memory.size() > self.batch_size:
            batch = self.memory.sample(self.batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])

            # Train network
            target_Qs = self.sess.run(self.qnet.output, feed_dict={self.qnet.inputs_: next_states})

            # Set target_Qs to 0 for states where episode ends
            # episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
            # target_Qs[episode_ends] = (0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

            targets = rewards + self.gamma * np.max(target_Qs, axis=1)

            loss, _ = self.sess.run([self.qnet.loss, self.qnet.opt],
                               feed_dict={self.qnet.inputs_: states,
                                          self.qnet.targetQs_: targets,
                                          self.qnet.actions_: actions})
        return self.game.space , reward,loss
