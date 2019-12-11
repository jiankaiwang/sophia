import gym
import sys
env = gym.make('CartPole-v0')

# In[]

# list all environments
from gym import envs
print(envs.registry.all())

# In[]

# https://gym.openai.com/assets/docs/aeloop-138c89d44114492fd02822303e6b4b07213010bb14ca5856d2d49d6b62d88e53.svg

print("action space")
# Discrete(2)
# The Discrete space allows a fixed range of non-negative numbers
print(env.action_space)
print(env.action_space.n)

print("observation space")
# Box(4,)
# The Box space represents an n-dimensional boxhttps://www.youtube.com/watch?list=LLaEn6TtKrd6gPEYQsnfOM_A&v=SdzLl-XpJt0
print(env.observation_space) 
print(env.observation_space.shape) 
print(env.observation_space.high)
print(env.observation_space.low)

# In[]

from gym import spaces

# Box and Discrete are common spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}

# You can sample from a space.
x = space.sample()

# You can also check whether the space contains the value.
assert space.contains(x)
assert space.n == 8

# In[]

for i_episode in range(20):
    # initialize the environment
    observation = env.reset()
    
    for t in range(100):
        env.render()
        #print(observation)
        
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        if done:
            print("Episode finished after {} timesteps, reward is {}".format(
                t+1, reward))
            break