from stable_baselines3.common.callbacks import CheckpointCallback
import warnings
from src.env import AvellanedaStoikovEnv
from src.market import ASMarket
from stable_baselines3 import A2C
from config import config
import os
import numpy as np

warnings.filterwarnings("ignore")

class CustomASEnv(AvellanedaStoikovEnv):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.logger_dir = '/Users/jadonng/Desktop/MarketMaking/log/Simulation/A2C/utility'
    
    def reward_function(self,db, da, dnb, dna, q, delta_s, t, dampened_factor=0.7):
        dw = self.Market.get_w(t) - self.Market.get_w(t-1)
        self.Market.set_dw(dw, t)
        return dw - self.gamma * (dw - self.Market.dw.mean())**2


model_path = config.model_path
env = CustomASEnv(config, logger_enabled=True)
agent = A2C.load(model_path+'baseline', env=env)
for i in range(5000):
    terminated = False
    truncated = False
    n_state, info = env.reset()
    while not terminated and not truncated:
        action, _ = agent.predict(n_state, deterministic=True)
        n_state, reward, terminated, truncated, info = env.step(action)
    
