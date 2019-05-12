import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import sys
import random
from gym.wrappers import Monitor
from collections import namedtuple

env = gym.make('Breakout-v0')
env = Monitor(env, ./video)
env.reset()
VALID_ACTIONS = [0, 1, 2, 3]

class StateProcessor:
    def __init__(self):
        self.input_state = tf.placeholder(shape = [210, 160, 3], dtype = tf.uint8)
        self.output_state = tf.image.rgb_to_grayscale(self.input_state)
        self.output_state = tf.image.crop_to_bounding_box(self.output_state, 34, 0, 160, 160)
        self.output_state = tf.image.resize_images(self.ouput_state, [84, 84], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.output_state = tf.image.squeeze(self.output_state)
    def process(self, sess, state):
        return sess.run(self.ouput_state, {self.input_state : state})
class DQN:
    def __init__(self):
        self.build_model()
    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, 8, 4, input_shape = [84, 84, 4], activation = tf.nn.relu))
        self.model.add(Conv2D(64, 4, 2, activation = tf.nn.relu))
        self.model.add(Conv2D(64, 3, 1, activation = tf.nn.relu))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation = tf.nn.relu))
        self.model.add(Dense(len(VALID_ACTIONS)))
        model.compile(
            loss = 'mse'
            optimizer = RMSprop(0.00025, 0.99, 0.0, 1e-6)
        )
    def predict(self, state):
        self.model.predict(state)
    def update(self, state, action, target_state):
        target = model.predict(state)
        target[0][action] = target_state
        model.fit(state, target, epochs = 1, verbose = 2)
