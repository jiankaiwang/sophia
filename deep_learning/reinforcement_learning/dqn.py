#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
@Note:
  install `pip install gym`
  install `pip install atari-py` first.
"""

import tensorflow as tf
import gym
import tensorflow.contrib.slim as slim
import cv2
import numpy as np
import random
import tqdm
import copy

# In[]

def epsilon_greedy_action_annealed(action_distribution, training_percentage,
                                   epsilon_start=1.0, epsilon_end=1e-2):
  """Explore and Exploit."""
  annealed_epsilon = epsilon_start * (1-training_percentage) + epsilon_end * training_percentage
  if random.random() < annealed_epsilon:
    # take random action
    return np.argmax(np.random.random(action_distribution.shape))
  else:
    # take the recommended action
    return np.argmax(action_distribution)

# In[]

class DQNAgent(object):

  def __init__(self, session, num_actions,
               learning_rate=1e-3, history_length=4,
               screen_height=84, screen_width=84, gamma=0.98):
    self.session = session
    self.num_actions = num_actions
    self.learning_rate = learning_rate
    self.history_length = history_length
    self.screen_height = screen_height
    self.screen_width = screen_width
    self.gamma = gamma

    self.build_prediction_network()
    self.build_target_network()
    self.build_training()

  def build_prediction_network(self):
    with tf.variable_scope("pred_network"):
      self.s_t = tf.placeholder('float32',
                                shape=[None, self.history_length,
                                       self.screen_height, self.screen_width],
                                name="state")
      self.conv_0 = slim.conv2d(self.s_t, 32, 8, 4, scope='conv_0')
      self.conv_1 = slim.conv2d(self.conv_0, 64, 4, 2, scope='conv_1')
      self.conv_2 = slim.conv2d(self.conv_1, 64, 3, 1, scope='conv_2')

      shape = self.conv_2.get_shape().as_list()

      self.flattened = tf.reshape(self.conv_2, [-1, shape[1]*shape[2]*shape[3]])
      self.fc_0 = slim.fully_connected(self.flattened, 512, scope='fc_0')
      self.q_t = slim.fully_connected(self.fc_0, self.num_actions,
                                      activation_fn=None, scope='q_values')

      #self.q_action = tf.argmax(self.q_t, dimension=1)

  def build_target_network(self):
    with tf.variable_scope("target_network"):
      self.target_s_t = tf.placeholder('float32',
                                       shape=[None, self.history_length,
                                              self.screen_height, self.screen_width],
                                       name="state")
      self.target_conv_0 = slim.conv2d(self.target_s_t, 32, 8, 4, scope='conv_0')
      self.target_conv_1 = slim.conv2d(self.target_conv_0, 64, 4, 2, scope='conv_1')
      self.target_conv_2 = slim.conv2d(self.target_conv_1, 64, 3, 1, scope='conv_2')

      shape = self.target_conv_2.get_shape().as_list()

      self.target_flattened = tf.reshape(self.target_conv_2, [-1, shape[1]*shape[2]*shape[3]])
      self.target_fc_0 = slim.fully_connected(self.target_flattened, 512, scope="fc_0")
      self.target_q = slim.fully_connected(self.target_fc_0, self.num_actions,
                                           activation_fn=None, scope='q_values')

      #self.target_q_action = tf.argmax(self.target_q, dimension=1)


  def update_target_q_weights(self):
    """
    update target q weights which is based on predicted q weights
    """
    pred_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pred_network')
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network')

    for target_var, pred_var in zip(target_vars, pred_vars):
      weight_input = tf.placeholder('float32', name='weight')
      target_var.assign(weight_input).eval({weight_input: pred_var.eval()})

  def sample_and_train_pred(self, replay_table, batch_size):
    s_t, action, reward, s_t_plus_1, terminal = replay_table.sample_batch(batch_size)
    q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

    terminal = np.array(terminal) + 0.
    max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
    target_q_t = (1. - terminal) * self.gamma * max_q_t_plus_1 + reward
    _, q_t, loss = self.session.run([self.train_step, self.q_t, self.loss],
                                    {self.target_q_t: target_q_t,
                                     self.action: action, self.s_t: s_t})
    return q_t

  def build_training(self):
    self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
    self.action = tf.placeholder('int64', [None], name='action')

    action_one_hot = tf.one_hot(self.action, self.num_actions, 1.0, 0.0, name='action_one_hot')
    q_of_action = tf.reduce_sum(self.q_t * action_one_hot, reduction_indices=1, name='q_of_action')

    self.delta = tf.square((self.target_q_t - q_of_action))
    self.loss = tf.reduce_mean(self.clip_error(self.delta), name='loss')

    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.train_step = self.optimizer.minimize(self.loss)

  def sample_action_from_distribution(self, action_distribution,
                                      epsilon_percentage):
    action = epsilon_greedy_action_annealed(action_distribution, epsilon_percentage)
    return action

  def predict_action(self, state, epsilon_percentage):
    action_distribution, = self.session.run([self.q_t], feed_dict={self.s_t: [state]})
    action = self.sample_action_from_distribution(action_distribution, epsilon_percentage)
    return action

  def process_state_into_stacked_frames(self, frame,
                                        past_frames, past_state=None):

    # the shape is [..., screen width, screen height]
    full_state = np.zeros((self.history_length, self.screen_width, self.screen_height))

    if past_state is not None:
      for i in range(len(past_state)-1):
        full_state[i,:,:] = past_state[i+1,:,:]
      full_state[-1,:,:] = self.preprocess_frame(frame, (self.screen_width, self.screen_height))
    else:
      all_frames = past_frames + [frame]
      for i, frame_f in enumerate(all_frames):
        full_state[i,:,:] = self.preprocess_frame(frame_f, (self.screen_width, self.screen_height))
    return full_state

  def to_grayscale(self, x):
    return np.dot(x[...,:3], [0.299, 0.587, 0.114])

  def clip_error(self, x):
    try:
      return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
      return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

  def preprocess_frame(self, im, shape):
    cropped = im[16:201, :]
    grayscaled = self.to_grayscale(cropped)
    #resized = imresize(grayscaled, shape, 'nearest').astype('float32')
    resized = cv2.resize(grayscaled, shape)
    mean, std = 40.45, 64.15
    frame = (resized-mean) / std
    return frame


# In[]

class EpisodeHistory(object):

  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.state_primes = []
    self.terminals = []

  def add_to_history(self, state, action, reward, state_prime, terminal):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)
    self.state_primes.append(state_prime)
    self.terminals.append(terminal)

# In[]

class ExperienceReplayTable(object):

  def __init__(self, table_size=5000):
    self.states = []
    self.actions = []
    self.rewards = []
    self.state_primes = []
    self.terminals = []

    self.table_size = table_size

  def add_episode(self, episode):
    self.states += episode.states
    self.actions += episode.actions
    self.rewards += episode.rewards
    self.state_primes += episode.state_primes
    self.terminals += episode.terminals

    self.purge_old_experiences()

  def purge_old_experiences(self):
    while len(self.states) > self.table_size:
      self.states.pop(0)
      self.actions.pop(0)
      self.rewards.pop(0)
      self.state_primes.pop(0)
      self.terminals.pop(0)

  def sample_batch(self, batch_size):
    s_t, action, reward, s_t_plus_1, terminal = [], [], [], [], []

    rands = np.arange(len(self.states))
    np.random.shuffle(rands)
    rands = rands[:batch_size]

    for r_i in rands:
      s_t.append(self.states[r_i])
      action.append(self.actions[r_i])
      reward.append(self.rewards[r_i])
      s_t_plus_1.append(self.state_primes[r_i])
      terminal.append(self.terminals[r_i])

    return np.array(s_t), np.array(action), np.array(reward), \
          np.array(s_t_plus_1), np.array(terminal)

# In[]

def main(argv):
  # configuration
  scale = 10
  total_episodes = 500 * scale
  learn_start = total_episodes // 2
  epsilon_stop = 200 * scale
  train_frequency = 4
  target_frequency = 1000
  batch_size = 32
  max_episode_length = 100000

  render_start = 10   # start to render the frame
  should_render = True

  env = gym.make('Breakout-v4')
  num_actions = env.action_space.n

  solved = False
  with tf.Session() as sess:
    agent = DQNAgent(session=sess, num_actions=num_actions, learning_rate=1e-3,
                     history_length=4, gamma=0.98)
    sess.run(tf.global_variables_initializer())

    episode_rewards = []
    q_t_list = []

    replay_table = ExperienceReplayTable()
    global_step_counter = 0

    for i in tqdm.tqdm(range(total_episodes)):
      frame = env.reset()
      past_frames = [copy.deepcopy(frame) for _ in range(agent.history_length-1)]
      state = agent.process_state_into_stacked_frames(frame, past_frames, past_state=None)

      episode_reward = 0.0
      episode_history = EpisodeHistory()
      epsilon_percentage = float(min(i / float(epsilon_stop),1.0))

      for j in range(max_episode_length):
        action = agent.predict_action(state, epsilon_percentage)

        if global_step_counter < learn_start:
          action = np.argmax(np.random.random(agent.num_actions))

        reward = 0

        frame_prime, reward, terminal, _ = env.step(action)
        if terminal:
          reward -= 1

        state_prime = agent.process_state_into_stacked_frames(frame_prime, past_frames, past_state=state)

        past_frames.append(frame_prime)
        past_frames = past_frames[len(past_frames)-agent.history_length:]

        if (i > render_start) and should_render and solved:
          env.render()

        episode_history.add_to_history(state, action, reward, state_prime, terminal)
        state = state_prime
        episode_reward += reward
        global_step_counter += 1

        if global_step_counter > learn_start and global_step_counter % train_frequency == 0:
          q_t = agent.sample_and_train_pred(replay_table, batch_size)
          q_t_list.append(q_t)

          if global_step_counter % target_frequency == 0:
            agent.update_target_q_weights()

        if j == (max_episode_length - 1):
          terminal = True

        if terminal:
          replay_table.add_episode(episode_history)
          episode_rewards.append(episode_reward)
          break

      if i % 50 == 0:
        ave_reward = np.mean(episode_rewards[-100:])

        print("Reward stats (min, max, median, mean): ",
              np.min(episode_rewards[-100:]),
              np.max(episode_rewards[-100:]),
              np.median(episode_rewards[-100:]),
              ave_reward)

        print("Global stats (ep_percentage, global_step_counter)",
              str(epsilon_percentage),
              global_step_counter)

        if q_t_list:
          print("QT stats (min, max, median, mean)",
                np.min(q_t_list[-1000:]),
                np.max(q_t_list[-1000:]),
                np.median(q_t_list[-1000:]),
                np.mean(q_t_list[-1000:]))

        if ave_reward > 100.0:
          solved = True
          print("Solved.")
        else:
          solved = False
          print(ave_reward)


# In[]

if __name__ == "__main__":
  main('')






























