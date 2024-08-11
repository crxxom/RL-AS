from config import config
import pandas as pd
import numpy as np
import os
from src.env import AvellanedaStoikovEnv
from algorithms.MonteCarlo import MonteCarloControl

class CustomASEnv(AvellanedaStoikovEnv):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
    
    # mean variance utility
    def reward_function(self,db, da, dnb, dna, q, delta_s, t, dampened_factor=0.7):
        dw = self.Market.get_w(t) - self.Market.get_w(t-1)
        self.Market.set_dw(dw, t)
        return dw - self.gamma * (dw - self.Market.dw.mean())**2

    # # inventory penalization
    # def reward_function(self,db, da, dnb, dna, q, delta_s, t, dampened_factor=0.7):
    #     # print('reward:', db*dnb + da*dna - self.invpen_lambda*abs(q))
    #     reward = db*dnb + da*dna - self.invpen_lambda*abs(q)
        
    #     if np.isnan(reward) or reward is None:
    #         reward=0
    #     return reward
    
    # # asymmetrically dampened
    # def reward_function(self,db, da, dnb, dna, q, delta_s, t, dampened_factor=0.7):
    #     return self.nondamp_pnl(db, da, dnb, dna, q, delta_s) - max(0, dampened_factor*(q*delta_s))

env = CustomASEnv(config, logger_enabled=True)
agent = MonteCarloControl(env, config)
agent.learn(episodes=10000)
agent.save()
