from stable_baselines3.common.callbacks import CheckpointCallback
import warnings
from src.env import AvellanedaStoikovEnv
from src.market import ASMarket
from config import config
import os
import numpy as np
from algorithms.TemporalDifference import SARSA, QTable

warnings.filterwarnings("ignore")

class CustomASEnv(AvellanedaStoikovEnv):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.logger_dir = '/Users/jadonng/Desktop/MarketMaking/log/Simulation/SARSA/asymdamp'
    
     # asymmetrically dampened
    def reward_function(self,db, da, dnb, dna, q, delta_s, t, dampened_factor=0.7):
        return self.nondamp_pnl(db, da, dnb, dna, q, delta_s) - max(0, dampened_factor*(q*delta_s))


grid = [np.linspace(config.low[i], config.high[i], config.bins[i]) for i in range(len(config.low))]
q_table = QTable(tuple(config.bins), config.N_ACTIONS, grid)

env = CustomASEnv(config, logger_enabled=True)
agent = SARSA(env, q_table, config)
agent.load(config.model_path)
for i in range(5000):
    terminated = False
    truncated = False
    n_state, info = env.reset()
    while not terminated and not truncated:
        action, _ = agent.predict(n_state)
        n_state, reward, terminated, truncated, info = env.step(action)
    
