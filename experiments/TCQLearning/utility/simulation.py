from stable_baselines3.common.callbacks import CheckpointCallback
import warnings
from src.env import AvellanedaStoikovEnv
from src.market import ASMarket
from config import config
import os
import numpy as np
from algorithms.TemporalDifference import SARSA, QTable, create_tile_coding_table, TileCodedQTable, QLearning

warnings.filterwarnings("ignore")

class CustomASEnv(AvellanedaStoikovEnv):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.logger_dir = '/Users/jadonng/Desktop/MarketMaking/log/Simulation/TCQLearning/utility'
    
     # mean variance utility
    def reward_function(self,db, da, dnb, dna, q, delta_s, t, dampened_factor=0.7):
        dw = self.Market.get_w(t) - self.Market.get_w(t-1)
        self.Market.set_dw(dw, t)
        return dw - self.gamma * (dw - self.Market.dw.mean())**2
    

low, high, tiling = create_tile_coding_table(config.low, config.high, config.bins, config.N_TILES)
    
# training
tq = TileCodedQTable(low, high, tiling, config.N_ACTIONS, alpha=config.alpha)

env = CustomASEnv(config, logger_enabled=True)
agent = QLearning(env, tq, config)
agent.load(config.model_path)
for i in range(5000):
    terminated = False
    truncated = False
    n_state, info = env.reset()
    while not terminated and not truncated:
        action, _ = agent.predict(n_state)
        n_state, reward, terminated, truncated, info = env.step(action)
    
