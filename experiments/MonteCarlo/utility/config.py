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
    invpen_lambda = 0.1
    
    dampened_factor = 0.7
    max_inv = 20

    logger_dir = '/Users/jadonng/Desktop/MarketMaking/log/Training/MonteCarlo/utility'
    
    experiment_name = 'MC_g97_a001'
    
    # RL config
    N_ACTIONS=25
    low = [-max_inv,0]
    high = [max_inv,1]
    continuous_action = False
    quote_max_displacement = 1.5


    # Tile Coding config
    N_TILES=8
    bins = [20,5]
    
    # TD Learning config
    model_path = '/Users/jadonng/Desktop/MarketMaking/models/MonteCarlo/utility'
    RLgamma=0.97
    epsilon=0.7
    min_epsilon=0.0001
    epsilon_decay_rate=0.95
    alpha=0.001
    
    model_save_freq=250000
    initial_timesteps=0






