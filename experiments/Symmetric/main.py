from config import config
import pandas as pd
import numpy as np
import os
from src.env import AvellanedaStoikovEnv
from algorithms.SymmetricAgent import SymmetricAgent

displacements = [1.4907704227476832/2]

for displacement in displacements:
    config.logger_dir = f'/Users/jadonng/Desktop/MarketMaking/log/Simulation/Symmetric'
    config.experiment_name = f'Symmetric_ASspread'
    for i in range(5000):
        env = AvellanedaStoikovEnv(config, logger_enabled=True)
        agent = SymmetricAgent(env, config, displacement=displacement)
        agent.run()