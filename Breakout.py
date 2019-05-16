import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import sys
from collections import namedtuple

class StateProcessor:
    def __init__(self):
        with tf.variable_scope('process'):
            self.input_state = tf.placeholder(shape = [210, 160, 3], dtype = tf.uint8, name = 'input_state')
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160) # offset height, offset width, x, y
            self.output = tf.image.resize_images(self.output, [84, 84], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output) # remove demension wich values = 1
    def process(self, sess, state):
        return sess.run(self.output, {self.input_state : state})

env = gym.make('Breakout-v0')
env = gym.wrappers.Monitor(env, "./drive/My Drive/dqn/video", video_callable=lambda episode_id: episode_id%50==0,force=True)
env.reset()

VALID_ACTIONS = [0, 1, 2, 3]

class DQN:
    def __init__(self, scope):
        self.scope = scope
        with tf.variable_scope(self.scope):
            self.build_model()
    def build_model(self):
        self.x = tf.placeholder(shape = [None, 84, 84, 4], dtype = tf.uint8, name = 'x')
        self.y = tf.placeholder(shape = [None], dtype = tf.float32, name = 'y')
        self.a = tf.placeholder(shape = [None], dtype = tf.int32, name = 'actions')

        x = tf.to_float(self.x) / 255.0
        batch_size = tf.shape(self.x)[0]

        conv1 = tf.layers.conv2d(x, 32, 8, 4, activation = tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, 4, 2, activation = tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, 3, 1, activation = tf.nn.relu)

        flatten = tf.contrib.layers.flatten(conv3)
        fc1 = tf.layers.dense(flatten, 512, activation = tf.nn.relu)
        self.predictions = tf.layers.dense(fc1, len(VALID_ACTIONS))

        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.a
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        self.losses = tf.squared_difference(self.y, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train = self.optimizer.minimize(self.loss)

    def predict(self, sess, s):
        return sess.run(self.predictions, {self.x : s})

    def update(self, sess, s, a, y):
        _, loss = sess.run([self.train, self.loss], {self.x : s, self.y : y, self.a : a})
        return loss

def copy_model_parameters(sess, DQN1, DQN2):
    n1 = [t for t in tf.trainable_variables() if t.name.startswith(DQN1.scope)]
    n1 = sorted(n1, key = lambda v: v.name)
    n2 = [t for t in tf.trainable_variables() if t.name.startswith(DQN2.scope)]
    n2 = sorted(n2, key = lambda v: v.name)

    update_ops = []
    for n1_v, n2_v in zip(n1, n2):
        op = n2_v.assign(n1_v)
        update_ops.append(op)
    sess.run(update_ops)

tf.reset_default_graph()

dqn = DQN(scope = 'dqn')
target_dqn = DQN(scope = 'target_dqn')
state_processor = StateProcessor()

#hyperparameters
num_episodes = 10000

replay_memory_size = 250000
replay_memory_init_size = 50000

update_target_dqn = 10000

epsilon_start = 1.0
epsilon_end = 0.1

epsilon_decay_steps = 50000
discount_factor = 0.99
batch_size = 32

def make_epsilon_greedy_policy(dqn, nA):
    def policy_fn(sess, state, epsilon):
        A = np.ones(nA, dtype = float) * epsilon/nA
        q_values = dqn.predict(sess, np.expand_dims(state, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

start_i_episode = 0
opti_step = -1

replay_memory = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

    saver = tf.train.Saver()

    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    policy = make_epsilon_greedy_policy(dqn, len(VALID_ACTIONS))

    epi_reward = []
    best_epi_reward = 0

    for i_episode in range(start_i_episode, num_episodes):
        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis = 2)
        loss = None
        done = False
        r_sum = 0
        mean_epi_reward = np.mean(epi_reward)
        if best_epi_reward < mean_epi_reward:
            best_epi_reward = mean_epi_reward
        if i_episode%1000 == 0:
            saver.save(tf.get_default_session(), './drive/My Drive/dqn/')
            np.save('./drive/My Drive/dqn/epi_reward.npy', epi_reward)
            np.save('./drive/My Drive/dqn/replay_memory.npy', replay_memory)
        len_replay_memory = len(replay_memory)
        while not done:
            epsilon = epsilons[min(opti_step+1, epsilon_decay_steps-1)]

            if opti_step % update_target_dqn == 0:
                copy_model_parameters(sess, dqn, target_dqn)
            if i_episode%100 ==0:
              print("\r Epsilon ({}) ReplayMemorySize : ({}) rSum: ({}) best_epi_reward: ({}) OptiStep ({}) @ Episode {}/{}, loss: {}".format(epsilon, len_replay_memory, mean_epi_reward, best_epi_reward, opti_step, i_episode + 1, num_episodes, loss), end="")
              sys.stdout.flush()

            actions_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(actions_probs)), p = actions_probs)

            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            r_sum += reward

            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis = 2)

            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            replay_memory.append(Transition(state, action, reward, next_state, done))

            if len_replay_memory > replay_memory_init_size:
                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                q_values_next_target = target_dqn.predict(sess, next_states_batch)
                best_action_target = np.argmax(q_values_next_target, axis = 1)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * q_values_next_target[np.arange(batch_size), best_action_target]

                states_batch = np.array(states_batch)
                loss = dqn.update(sess, states_batch, action_batch, targets_batch)

                opti_step +=1

            state = next_state
            if done:
                break
        epi_reward.append(r_sum)
        if len(epi_reward) > 100:
            epi_reward = epi_reward[1:]
