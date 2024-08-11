from config import config
import pandas as pd
import numpy as np
import os
from src.env import AvellanedaStoikovEnv
from algorithms.TemporalDifference import TileCodedQTable, QLearning, SARSA


def create_tile_coding_table(low, high, bins, N_TILES):
    # create tile coded Qtable    
    tiling_spec = []
    for j in range(N_TILES):
        spec = []
        spec.append(tuple(bins))
        offset = []
        for i in range(len(low)):
            inc = (high[i] - low[i])/bins[i]/ N_TILES
            offset.append(inc*j)
        spec.append(tuple(offset))
        
        tiling_spec.append(tuple(spec))
        
    return low, high, tiling_spec

low, high, tiling = create_tile_coding_table(config.low, config.high, config.bins, config.N_TILES)


class CustomASEnv(AvellanedaStoikovEnv):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
    
    def reward_function(self,db, da, dnb, dna, q, delta_s, t, dampened_factor=0.7):
        dw = self.Market.get_w(t) - self.Market.get_w(t-1)
        self.Market.set_dw(dw, t)
        return dw - self.gamma * (dw - self.Market.dw.mean())**2

path = '/'.join(config.model_path.split('/')[:-1])
if not os.path.exists(path):
    os.makedirs(path)
    
# training
tq = TileCodedQTable(low, high, tiling, config.N_ACTIONS, alpha=config.alpha)

env = CustomASEnv(config, logger_enabled=True)
agent = SARSA(env, tq, config)
agent.load(config.model_path)
agent.learn(total_timesteps=200000)
agent.save()
