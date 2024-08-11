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

    price_tick = 0.01
    lambda_neg = 1
    lambda_pos = 1
    
    dampened_factor = 0.7
    max_inv = 20

    logger_dir = '/Users/jadonng/Desktop/MarketMaking/log/SARSA/TCSARSA_asympnl'
    experiment_name = 'TCSARSA_k1p5_sigma2'
    
    # RL config
    N_ACTIONS=9
    # low = [math.floor(s0-sigma*np.sqrt(dt)/dt),-max_inv,0]
    # high = [math.ceil(s0+sigma*np.sqrt(dt)/dt),max_inv,1]
    low = [-max_inv,0]
    high = [max_inv,1]

    # Tile Coding config
    N_TILES=8
    # bins = [50,40,10]
    bins = [40,10]
    
    # TD Learning config
    model_path = '/Users/jadonng/Desktop/MarketMaking/models/SARSA/TCSARSA_asympnl'
    RLgamma=0.97
    epsilon=0.7
    min_epsilon=0.0001
    epsilon_decay_rate=0.9995
    alpha=0.001
    
    model_save_freq=250000
    initial_timesteps=0





