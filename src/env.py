from .market import ASMarket
import numpy as np
from gymnasium import Env, spaces
import math

class AvellanedaStoikovEnv(Env):
    def __init__(self, config, logger_enabled=False):
        self.config = config
        self.T = config.T
        self.dt = config.dt
        self.t = 0
        
        self.gamma = config.gamma
        self.s0 = config.s0
        self.A = config.A
        self.k = config.k
        self.sigma = config.sigma 
        self.dampened_factor = config.dampened_factor
        self.lambda_pos = config.lambda_pos
        self.lambda_neg = config.lambda_neg

        self.invpen_lambda = config.invpen_lambda
        
        self.q = 0
        self.c = 0
        self.w = 0

        self.max_inv = config.max_inv
        self.price_tick = config.price_tick
        self.Market = ASMarket(config)
        self.logger_enabled = logger_enabled
        self.logger_dir = config.logger_dir
        self.experiment_name = config.experiment_name

        self.quote_max_displacement = config.quote_max_displacement
        self.continuous_action = config.continuous_action
        self.N_ACTIONS = config.N_ACTIONS

        # RL settings
        self.observation_space = spaces.Box(low=np.array(config.low),
                                         high=np.array(config.high),
                                         dtype=np.float32)

        if self.continuous_action:
            self.action_space = spaces.Box(low=np.array([0.001,0.001]),
                                          high=np.array([self.quote_max_displacement,self.quote_max_displacement]),
                                          dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(config.N_ACTIONS)
        self.reward_range = (-math.inf,math.inf)
    
    def reset(self, seed=42):
        self.t = 0
        self.q = 0
        self.c = 0
        self.w = 0
        self.Market.reset()
        n_state = self.Market.get_state(0)
        
        return n_state, {}
        
    def step(self, action):
        s = self.Market.get_midprice(self.t)

        # calculate limit order
        bid, ask = self.get_bidask(action, s)
        
        db = s - bid
        da = ask - s

        # inventory constraints
        if self.q >= self.max_inv:
            db = 9999
        elif self.q <= -self.max_inv:
            da = 9999

        # calculate limit order execution probability
        lb = self.l(self.A, self.k, db)
        la = self.l(self.A, self.k, da)
        dnb = 1 if np.random.uniform() <= lb * self.dt else 0
        dna = 1 if np.random.uniform() <= la * self.dt else 0

        # # The number of orders arrival
        # n_MO_buy = np.random.poisson(self.lambda_neg)
        # n_MO_sell = np.random.poisson(self.lambda_pos)
        # # Sample the number of orders executed
        # dnb = np.random.binomial(n_MO_buy, lb*self.dt)
        # dna = np.random.binomial(n_MO_sell, la*self.dt)

        # update inventory, cash
        self.q += dnb - dna
        self.c += -dnb * bid + dna * ask # cash
        
        # next state
        self.t += 1
        self.Market.q[self.t] = self.q
        self.Market.step(self.t)
        n_state = self.Market.get_state(self.t)

        # update wealth
        new_s = self.Market.get_midprice(self.t)
        self.w = self.c + self.q * self.Market.get_midprice(self.t)
        self.Market.set_w(self.w, self.t)
        
        # compute reward
        delta_s = new_s - s
        reward = self.reward_function(db, da, dnb, dna, self.q, delta_s, self.t, dampened_factor=self.dampened_factor)
        
        # log agentmetadata for analysis purpose
        self.Market.log_agentmetadata(bid, ask, dnb, dna, self.q, self.c, self.w, reward, self.t-1)
        # termination conditions
        info = {}
        terminated = False
        truncated = False
        if self.t >= self.T//self.dt:
            terminated = True
            if self.logger_enabled:
                self.Market.flush(self.logger_dir, self.experiment_name)
            
        return n_state, reward, terminated, truncated, info

    # =====================================
    # helper functions
    
    def l(self, A, k, d):
        return A*np.exp(-k*d) 

    def get_bidask(self, action, midprice):

        if self.continuous_action:
            p_b = self.round_to_tick(midprice - action[0])
            p_a = self.round_to_tick(midprice + action[1])
            return p_b, p_a
        
        elif isinstance(action, list): # AS  agents
            return action[0], action[1]

        else: # RL agent
            n = (np.sqrt(self.N_ACTIONS))
            dist_ask = self.quote_max_displacement/n * (action//n+1)
            dist_bid = self.quote_max_displacement/n * (action%n+1)
            p_a = self.round_to_tick(midprice + dist_ask)
            p_b = self.round_to_tick(midprice - dist_bid)
            return p_b, p_a
            
            
    def reward_function(self,db, da, dnb, dna, q, delta_s, t, dampened_factor=0.7):
        return self.nondamp_pnl(db, da, dnb, dna, q, delta_s) - max(0, dampened_factor*(q*delta_s))
        
    def nondamp_pnl(self,db, da, dnb, dna, q, delta_s):
        return dna*(da) + dnb*(db) + q*delta_s

    def round_to_tick(self, p):
        dp = len(str(self.price_tick)) - 2
        return round(p, dp)