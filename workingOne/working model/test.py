import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

env = gym.make('BipedalWalkerHardcore-v2')

model = SAC.load("sac_walker500000")

obs = env.reset()
for i in range(0,1500):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #print(rewards)
    env.render()
    if rewards == -100:
        break
env.close()