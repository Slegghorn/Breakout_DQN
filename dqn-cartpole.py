import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import numpy as np
import random
from collections import deque
import gym

class ReplayBuffer:
    def __init__(self, max_size):
        self.M = []
        self.max_size = max_size

    def store(self, state, action, reward, next_state, done):
        if self.M == self.max_size:
            self.M.pop(0)
        self.M.append((state, action, reward, next_state, done))

    def batch_sample(self, batch_size):
        batch = random.sample(self.M, batch_size)
        return batch

class Model:
    def __init__(self):
        self.NN = self.build_model(24, 24, 2, 4, 0.0005)

    def build_model(self, fc1, fc2, n_actions, input_dims, lr):
        model = Sequential()
        model.add(Dense(fc1, input_shape = (input_dims,), activation = 'relu'))
        model.add(Dense(fc2, activation = 'relu'))
        model.add(Dense(n_actions, activation = 'linear'))
        model.compile(optimizer = Adam(lr = lr), loss = 'mse')
        return model

class Agent:
    def __init__(self):
        self.model = Model()
        self.memory = ReplayBuffer(200000)
        self.action_space = [0, 1]
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.996

    def choose_action(self, state):
        state = np.array(state)[np.newaxis, :]
        if random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            actions = self.model.NN.predict(state)
            return np.argmax(actions[0])

    def learn(self):
        if len(self.memory.M) > self.batch_size:
            batch = self.memory.batch_sample(32)
            for state, action, reward, next_state, done in batch:
                state = np.array(state)[np.newaxis, :]
                next_state = np.array(next_state)[np.newaxis, :]

                if done:
                    target = reward
                if not done:
                    target = reward + self.gamma * np.max(self.model.NN.predict(next_state)[0])

                q = self.model.NN.predict(state)
                q[0][action] = target

                self.model.NN.fit(state, q, epochs = 1, verbose = 0)
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay


agent = Agent()
env = gym.make('CartPole-v0')
#env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: episode_id%50==0,force=True)
env.reset()

n_games = 500

scores = []
for n in range(n_games):
    done = False
    score = 0
    state = env.reset()
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        score += reward
        agent.memory.store(state, action, reward, next_state, done)
        state = next_state
        agent.learn()
    scores.append(score)

    avg_score = np.mean(scores[-10:-1])

    print('episode {}   score {}   average score {}'.format(n, score, avg_score))
