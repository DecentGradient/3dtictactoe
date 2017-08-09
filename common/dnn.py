import numpy as np
import tensorflow as tf
import tflearn

from common.replay_buffer import ReplayBuffer


class ActorNetwork(object):

    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau,vscope="act"):
        self.sess = sess
        self.vscope = vscope
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):

        with tf.variable_scope(self.vscope):
            inputs = tf.placeholder(dtype=tf.float32,shape=[None,self.s_dim],name="a_inputs")
            # w1 = tf.placeholder(dtype=tf.float32,shape=[self.s_dim,400],name="w1")
            # #b1 = tf.placeholder(dtype=tf.float32)
            #
            # w2 = tf.placeholder(dtype=tf.float32,shape=[400,300],name="w2")
            # w3 = tf.placeholder(dtype=tf.float32,shape=[300,self.a_dim],name="w3")

            #inputs = tf.Variable(tf.truncated_normal(shape=[None,self.s_dim],name="a_inputs")
            w1 = tf.Variable(tf.truncated_normal(shape=[self.s_dim,100],name="aw1"))
            b1 = tf.Variable(tf.zeros([100],name="ab1"))
            w2 = tf.Variable(tf.truncated_normal(shape=[100,300],name="aw2"))
            b2 = tf.Variable(tf.zeros([300],name="ab2"))
            w3 = tf.Variable(tf.truncated_normal(shape=[300,400],name="aw3"))
            b3 = tf.Variable(tf.zeros([400],name="ab3"))
            w4 = tf.Variable(tf.truncated_normal(shape=[400,400],name="aw4"))
            b4 = tf.Variable(tf.zeros([400],name="ab4"))
            w5 = tf.Variable(tf.truncated_normal(shape=[400,300],name="aw5"))
            b5 = tf.Variable(tf.zeros([300],name="ab5"))
            w6 = tf.Variable(tf.truncated_normal(shape=[300,300],name="aw6"))
            b6 = tf.Variable(tf.zeros([300],name="ab6"))
            w7 = tf.Variable(tf.truncated_normal(shape=[300,200],name="aw7"))
            b7 = tf.Variable(tf.zeros([200],name="ab7"))
            w8 = tf.Variable(tf.truncated_normal(shape=[200,200],name="aw8"))
            b8 = tf.Variable(tf.zeros([200],name="ab8"))
            w9 = tf.Variable(tf.truncated_normal(shape=[200,100],name="aw9"))
            b9 = tf.Variable(tf.zeros([100],name="ab9"))
            w10 = tf.Variable(tf.truncated_normal(shape=[100,self.a_dim],name="aw10"))
            b10 = tf.Variable(tf.zeros([self.a_dim],name="ab10"))


            net1 = tf.nn.relu(tf.matmul(inputs, w1)+b1,name="anet1")
            net2 = tf.nn.relu(tf.matmul(net1,w2)+b2,name="anet2")
            net3 = tf.nn.relu(tf.matmul(net2, w3)+b3,name="anet3")
            net4 = tf.nn.relu(tf.matmul(net3,w4)+b4,name="anet4")
            net5 = tf.nn.relu(tf.matmul(net4, w5) + b5, name="anet5")
            net6 = tf.nn.relu(tf.matmul(net5, w6) + b6, name="anet6")
            net7 = tf.nn.relu(tf.matmul(net6, w7) + b7, name="anet7")
            net8 = tf.nn.relu(tf.matmul(net7, w8) + b8, name="anet8")
            net9 = tf.nn.relu(tf.matmul(net8, w9) + b9, name="anet9")

            out = tf.nn.tanh(tf.matmul(net9, w10) + b10, name="aout")


            # net = tf.layers.dense(inputs,400,activation=tf.nn.relu,name="al1")
            # net = tf.layers.dense(net, 300, activatsion=tf.nn.relu,name="al2")
            # # Final layer weights are init to Uniform[-3e-3, 3e-3]
            # # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            # # #print(w_init)
            # # #tf.identity(w_init,"weights")
            # out = tf.layers.dense(
            #     net, self.a_dim, activation=tf.nn.tanh)
            # # Scale output to -action_bound to action_bound

        return inputs, out , out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars,vscope="crit"):
        self.sess = sess
        self.vscope = vscope
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):

        with tf.variable_scope(self.vscope):
            inputs = tf.placeholder(shape=[None,self.s_dim],name="cinstate",dtype=tf.float32)
            action = tf.placeholder(shape=[None,self.a_dim],name="cact",dtype=tf.float32)
            w1 = tf.Variable(tf.truncated_normal(shape=[self.s_dim, 100], name="cw1"))
            b1 = tf.Variable(tf.zeros([100], name="cb1"))
            w2 = tf.Variable(tf.truncated_normal(shape=[100, 200], name="cw2"))
            b2 = tf.Variable(tf.zeros([200], name="cb2"))
            w3 = tf.Variable(tf.truncated_normal(shape=[200,300], name="cw3"))
            b3 = tf.Variable(tf.zeros([300], name="cb3"))
            w4 = tf.Variable(tf.truncated_normal(shape=[300, 400], name="cw4"))
            b4 = tf.Variable(tf.zeros([400], name="cb4"))
            w5 = tf.Variable(tf.truncated_normal(shape=[400, 300], name="cw5"))
            b5 = tf.Variable(tf.zeros([300], name="cb5"))
            w6 = tf.Variable(tf.truncated_normal(shape=[300, 200], name="cw6"))
            b6 = tf.Variable(tf.zeros([200], name="cb6"))
            w7 = tf.Variable(tf.truncated_normal(shape=[200, 100], name="cw7"))
            b7 = tf.Variable(tf.zeros([100], name="cb7"))
            w8 = tf.Variable(tf.truncated_normal(shape=[100, 50], name="cw8"))
            b8 = tf.Variable(tf.zeros([50], name="cb8"))
            w9 = tf.Variable(tf.truncated_normal(shape=[50, 50], name="cw9"))

            #b5 = tf.Variable(tf.zeros([300], name="ab2"))
            w10 = tf.Variable(tf.truncated_normal(shape=[ self.a_dim,50], name="cw10"))
            b10 = tf.Variable(tf.zeros([50], name="cb6"))
            w11 = tf.Variable(tf.random_uniform([50,1],minval=-0.003, maxval=0.003))
            net1 = tf.nn.relu(tf.matmul(inputs, w1) + b1, name="cnet1")
            net2 = tf.nn.relu(tf.matmul(net1, w2) + b2, name="cnet2")
            net3 = tf.nn.relu(tf.matmul(net2, w3) + b3, name="cnet3")
            net4 = tf.nn.relu(tf.matmul(net3, w4) + b4, name="cnet4")
            net5 = tf.nn.relu(tf.matmul(net4, w5) + b5, name="cnet5")
            net6 = tf.nn.relu(tf.matmul(net5, w6) + b6, name="cnet6")
            net7 = tf.nn.relu(tf.matmul(net6, w7) + b7, name="cnet7")
            net8 = tf.nn.relu(tf.matmul(net7, w8) + b8, name="cnet8")

            net9 = tf.nn.relu(tf.matmul(net8, w9) + tf.matmul(action,w10)+ b10, name="cnet9")
            out = tf.matmul(net9, w11)

            # net = tf.nn.relu()
            # #inputs = tflearn.input_data(shape=[None, self.s_dim])
            # #action = tflearn.input_data(shape=[None, self.a_dim])
            # net = tflearn.fully_connected( inputs, 400, activation='relu')
            #
            # # Add the action tensor in the 2nd hidden layer
            # # Use two temp layers to get the corresponding weights and biases
            # t1 = tflearn.fully_connected(net, 300)
            # t2 = tflearn.fully_connected(action, 300)
            #
            # net = tflearn.activation(
            #     tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
            #
            # # linear layer connected to 1 output representing Q(s,a)
            # # Weights are init to Uniform[-3e-3, 3e-3]
            # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            # out = tflearn.fully_connected(net, 1, weights_init=w_init)
            return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


class PlayerTrainer(object):
    def __init__(self,actor,critic,buffersize,game,player,batch_size,gamma):
        self.actor = actor
        self.critic = critic
        self.replay = ReplayBuffer(buffersize)
        self.game =game
        self.player = player
        self.batch_size = batch_size
        self.gamma = gamma


    def noisyMaxQMove(self):
        state = self.game.space
        As = self.actor.predict(np.reshape(state, (1, *state.shape)))
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

        a = np.random.choice(avail, p=availNorm)

        self.game.move(a,self.player)
        next_state, reward = self.game.step(self.player)

        self.bufferAdd(state,As,reward,self.game.game_over,next_state)
        if self.replay.size() > self.batch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay.sample_batch(self.batch_size)
            target_q = self.critic.predict_target(s2_batch,self.actor.predict_target(s2_batch))
            y_i = []
            for k in range(self.batch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.gamma * target_q[k])

            predicted_q_value, _ = self.critic.train(
                s_batch, a_batch, np.reshape(y_i, (self.batch_size, 1)))

            #ep_ave_max_q += np.amax(predicted_q_value)

            # Update the actor policy using the sampled gradient
            a_outs = self.actor.predict(s_batch)
            grads = self.critic.action_gradients(s_batch, a_outs)
            self.actor.train(s_batch, grads[0])

            # Update target networks
            self.actor.update_target_network()
            self.actor.update_target_network()
        return self.game.space , reward

    def bufferAdd(self,state,Qs,reward,terminal,next_state):
        self.replay.add(np.reshape(state,(self.actor.s_dim,)),np.reshape(Qs,(self.actor.a_dim,)),reward,terminal,np.reshape(next_state,(self.actor.s_dim,)))

