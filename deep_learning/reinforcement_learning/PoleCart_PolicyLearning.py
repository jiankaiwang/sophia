# -*- coding: utf-8 -*-
"""
author: jiankaiwang
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import tqdm
import gym
import os

# In[]

def discount_rewards(rewards, gamma=0.98):
    discounted_returns = [0 for _ in rewards]
    discounted_returns[-1] = rewards[-1]
    for t in range(len(rewards)-2, -1, -1):
        discounted_returns[t] = rewards[t] + discounted_returns[t+1]*gamma
    return discounted_returns

def epsilon_greedy_action(action_distribution, epsilon=1e-1):
  if random.random() < epsilon:
    return np.argmax(np.random.random(action_distribution.shape))
  else:
    return np.argmax(action_distribution)
      
def epsilon_greedy_annealed(action_distribution, training_percentage, 
                            epsilon_start=1.0, epsilon_end=1e-2):
  annealed_epsilon = epsilon_start * (1-training_percentage) + epsilon_end * training_percentage
  if random.random() < annealed_epsilon:
    # take random action
    return np.argmax(np.random.random(action_distribution.shape))
  else:
    # take the recommended action
    return np.argmax(action_distribution)

# In[]

# # Create an Agent

class PGAgent(object):
  def __init__(self, session, state_size, num_actions, 
               hidden_size, learning_rate=1e-3,
               explore_exploit_setting='epsilon_greedy_annealed_1.0->0.001'):
    
    self.session = session
    self.state_size = state_size
    self.num_actions = num_actions
    self.hidden_size = hidden_size
    self.leanring_rate = learning_rate
    self.explore_exploit_setting = explore_exploit_setting
    
    self.build_model()
    #self.build_training()
  
  def build_model(self):
    with tf.variable_scope("pg-model") as scope:
      self.state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
      self.h0 = slim.fully_connected(self.state, self.hidden_size)
      self.h1 = slim.fully_connected(self.h0, self.hidden_size)
      
      # output: (None, 2)
      self.output = slim.fully_connected(self.h1,
                                         self.num_actions,
                                         activation_fn=tf.nn.softmax)
      
      scope.reuse_variables()
  
  def build_training(self, global_step):
    self.action_input = tf.placeholder(shape=[None], dtype=tf.int32)
    self.reward_input = tf.placeholder(shape=[None], dtype=tf.float32)
    
    # select the corresponding action
    # * tf.shape(self.output)[1] is because we are going to reshape output 
    # into one-dimensional array, so we have to multiply column's number first
    # second, we have to plus action_input to make choice
    self.output_index_for_actions = (tf.range(0, tf.shape(self.output)[0]) * 
                                     tf.shape(self.output)[1]) + self.action_input
                                     
    # output (None, 2) reshape into (None * 2) by row first
    self.logits_for_actions = tf.gather(tf.reshape(self.output, [-1]), 
                                        self.output_index_for_actions)
    
    # summarizing the above
    # time-0's action might select output's first two value [9.89, 1.05, 7.6, 9.99, ...]
    # and 9.89 stands for time-0 with action 0
    # and 1.05 stands for time-0 with action 1
    # and 7.6 stands for time-1 with action 0
    # and 9.99 stands for time-1 with action 1    
    # ...
    
    self.loss = - tf.reduce_mean(tf.log(self.logits_for_actions) * self.reward_input)
    
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.leanring_rate)
    self.train_step = self.optimizer.minimize(self.loss, global_step=global_step)
  
  def sample_action_from_distribution(self, action_distribution, epsilon_percentage):
    """
    Parameters:
      epsilon_percentage is the same with training_percentage
      
    Return:
      the action id
    """
    if self.explore_exploit_setting == "epsilon_greedy_0.5":
      action = epsilon_greedy_action(action_distribution, 0.5)
    elif self.explore_exploit_setting == "epsilon_greedy_annealed_1.0->0.001":
      action = epsilon_greedy_annealed(action_distribution, epsilon_percentage,
                                       1.0, 1e-3)
    
    return action
  
  def predict_action(self, state, epsilon_percentage):
    action_distribution, = self.session.run(self.output, 
                                            feed_dict={self.state: [state]})
    action = self.sample_action_from_distribution(action_distribution, 
                                                  epsilon_percentage)
    return action

# In[]

class EpisodeHistory(object):
  
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.state_primes = []
    self.discounted_returns = []

  def add_to_history(self, state, action, reward, state_prime):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)
    self.state_primes.append(state_prime)

# In[]

class Memory(object):
  
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.state_primes = []
    self.discounted_returns = []
    
  def reset_memory(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.state_primes = []
    self.discounted_returns = []

  def add_episode(self, episode):    
    self.states += episode.states
    self.actions += episode.actions
    self.rewards += episode.rewards
    self.discounted_returns += episode.discounted_returns

# In[]

def main(full_training):
  
  # configurate the setting
  total_episodes = 7001 if full_training else 10  # total episode (actions)
  epsilon_stop = 3000 if full_training else 100  # for training percentage, e.g. 1 / 3000
  
  max_episode_length = 500 if full_training else 97  # explore or exploit times in each episodes
  train_frequency = 8  # collect 8 units max_episode_length
  
  should_render = True  # show game on the screen
  
  explore_exploit_setting = "epsilon_greedy_annealed_1.0->0.001"
  
  env = gym.make('CartPole-v0')
  state_size = env.observation_space.shape[0]  # here is 4
  num_actions = env.action_space.n  # here is 2
  
  solved = False
  
  episode_rewards = []
  batch_losses = []
  
  global_memory = Memory()
  global_step = tf.Variable(0, name="gloabl_step", trainable=False)
  
  with tf.Session() as sess:

    agent = PGAgent(session=sess, state_size=state_size,
                    num_actions=num_actions, hidden_size=16,
                    explore_exploit_setting=explore_exploit_setting)
    
    agent.build_training(global_step)
    
    saver = tf.train.Saver()
    
    latest_indexing = 0
    if os.path.exists(os.path.join(".","checkpoint")):
      latest_checkpoint = tf.train.latest_checkpoint("./")
      latest_indexing = int(latest_checkpoint.split("-")[-1])
      saver.restore(sess, latest_checkpoint)
      print("Model was restored from {}.".format(latest_checkpoint))
    else:
      sess.run(tf.global_variables_initializer())
    
    # here we introduce a new variable latest_indexing
    # because epsilon_percentage (training progress) is based on it
    for i in tqdm.tqdm(range(latest_indexing, total_episodes)):
      state = env.reset()
      
      episode_reward = 0.0
      episode_history = EpisodeHistory()
      epsilon_percentage = float(min(float(i) / epsilon_stop, 1.0))
      
      for j in range(max_episode_length):
        # the block is mainly used to fetch action and reward data
        
        action = agent.predict_action(state, epsilon_percentage)
        # state_prime: as Object, here is a list
        # reward: as float in shape (1, )
        # terminal: as bool in shape(1, )
        state_prime, reward, terminal, _ = env.step(action)
        
        if solved and should_render:
            env.render()
        
        episode_history.add_to_history(state, action, reward, state_prime)
        state = state_prime
        episode_reward += reward
        
        if terminal:
          # being True indicates the episode has terminated
          episode_history.discounted_returns = discount_rewards(episode_history.rewards)
          global_memory.add_episode(episode_history)
          
          if np.mod(i, train_frequency) == 0:
            # start the training
            feed_dict = {agent.action_input: np.array(global_memory.actions), 
                         agent.reward_input: np.array(global_memory.discounted_returns),
                         agent.state: np.array(global_memory.states)}
            _, batch_loss = sess.run([agent.train_step, agent.loss], feed_dict=feed_dict)
            batch_losses.append(batch_loss)
            global_memory.reset_memory()
          
          episode_rewards.append(episode_reward)
          break
       
      if i % 10 == 0:
        if np.mean(episode_rewards[:-100]) > 140.0:
          solved = True  
        else:
          solved = False
          
      if i % 2000 == 0 and i > 0:
        # save checkpoint
        saver.save(sess, "./cp-v0-checkpoint", global_step=global_step)
        print("Model was stored.")
            
      if i % 500 == 0 and i > 0:
        print('Solved: {}, Mean Reward: {}'.format(solved, np.mean(episode_rewards[:-100])))
    

# In[]
    
if __name__ == "__main__":
  full_training = True
  main(full_training)



