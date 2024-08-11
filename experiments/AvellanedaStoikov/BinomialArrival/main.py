from config import config
import pandas as pd
import numpy as np
import os
from src.env import AvellanedaStoikovEnv
from algorithms.AvellanedaStoikov import AvellanedaStoikovAgent

for i in range(5000):
    env = AvellanedaStoikovEnv(config, logger_enabled=True)
    agent = AvellanedaStoikovAgent(env, config)
    agent.run()