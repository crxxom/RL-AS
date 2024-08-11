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

# training
tq = TileCodedQTable(low, high, tiling, config.N_ACTIONS, alpha=config.alpha)

env = AvellanedaStoikovEnv(config, logger_enabled=True)
agent = SARSA(env, tq, config)
# agent.load('/mnt/wuzixi/Paper_Implementation/tspooner/model/SARSA/SARSA_consolidation/agent_baseline_750000')

agent.learn(total_timesteps=1000000)
agent.save()
