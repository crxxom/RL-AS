import numpy as np
import math

class config:
    # Configurations for the original AS model paper
    T = 1
    dt = 0.005
    A = 140
    k = 1.5
    s0 = 100
    sigma = 2
    gamma = 0.1
    
    dampened_factor = 0.7
    max_inv = 20

    logger_dir = '/Users/jadonng/Desktop/MarketMaking/log/AvellanedaStoikov'
    experiment_name = 'AS_k1p5_sigma2'

    
    # RL config
    N_ACTIONS=9
    low = [math.floor(s0-sigma*np.sqrt(dt)/dt),-max_inv,0]
    high = [math.ceil(s0+sigma*np.sqrt(dt)/dt),max_inv,1]


    # Tile Coding config
    N_TILES=16
    bins = []

    
    # TD Learning config
    RLgamma=0.97
    epsilon=0.7
    min_epsilon=0.0001
    epsilon_decay_rate=0.996
    alpha=0.001
    
    model_save_freq=250000






