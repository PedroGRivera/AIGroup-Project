import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC


tStep = 100000

env = gym.make('BipedalWalkerHardcore-v2')

model = SAC(MlpPolicy, 
            env, 
            learning_rate=0.001,
            batch_size=125,
            buffer_size=50000,
            verbose=1)
model.save("sac_walker_init")

for i in range(0,100):
    pos = (i+1)*tStep

    model.learn(total_timesteps=tStep, log_interval=10)
   model.save("sac_walker" + str(pos) )

   del model # remove to demonstrate saving and loading

   model = SAC.load("sac_walker" + str(pos))

   obs = env.reset()
   for i in range(0,1000):
       action, _states = model.predict(obs)
       obs, rewards, dones, info = env.step(action)
       env.render()
   env.close()



# model = SAC.load("sac_walker500000")

# obs = env.reset()
# for i in range(0,1500):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
# env.close()