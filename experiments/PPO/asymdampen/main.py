from stable_baselines3.common.callbacks import CheckpointCallback
import warnings
from src.env import AvellanedaStoikovEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from src.market import ASMarket
from stable_baselines3 import PPO
from config import config
import os
import numpy as np
import torch
import torch.nn as nn


warnings.filterwarnings("ignore")

class CustomASEnv(AvellanedaStoikovEnv):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    # # mean variance utility
    # def reward_function(self,db, da, dnb, dna, q, delta_s, t, dampened_factor=0.7):
    #     dw = self.Market.get_w(t) - self.Market.get_w(t-1)
    #     self.Market.set_dw(dw, t)
    #     return dw - self.gamma * (dw - self.Market.dw.mean())**2

    # # inventory penalization
    # def reward_function(self,db, da, dnb, dna, q, delta_s, t, dampened_factor=0.7):
    #     # print('reward:', db*dnb + da*dna - self.invpen_lambda*abs(q))
    #     reward = db*dnb + da*dna - self.invpen_lambda*abs(q)
        
    #     if np.isnan(reward) or reward is None:
    #         reward=0
    #     return reward
    
    # asymmetrically dampened
    def reward_function(self,db, da, dnb, dna, q, delta_s, t, dampened_factor=0.7):
        return self.nondamp_pnl(db, da, dnb, dna, q, delta_s) - max(0, dampened_factor*(q*delta_s))
        
env = CustomASEnv(config,logger_enabled=True)

model_path = config.model_path

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=500000,
  save_path=model_path,
  name_prefix='PPO',
  save_replay_buffer=True,
  save_vecnormalize=True,
)
# create model_path
if not os.path.exists(model_path):
    os.makedirs(model_path)

model = PPO("MlpPolicy", env=env)
# model = PPO.load(model_path +'baseline', env=env)
model.learn(total_timesteps=2000000, callback=checkpoint_callback, reset_num_timesteps=False) 
model.save(model_path+'baseline')


# ================================================================================================================


