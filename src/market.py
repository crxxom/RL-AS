import numpy as np
import os
import pandas as pd

class ASMarket:
    def __init__(self, config):
        self.config = config
        
        self.s0 = config.s0
        self.sigma = config.sigma
        self.T = config.T
        self.dt = config.dt
        self.price_tick = config.price_tick

        self.s = np.zeros(int(self.T/self.dt))

        # for logging purposes
        self.q = np.zeros(int(self.T/self.dt))
        self.c = np.zeros(int(self.T/self.dt))
        self.w = np.zeros(int(self.T/self.dt))
        self.bid = np.zeros(int(self.T/self.dt))
        self.ask = np.zeros(int(self.T/self.dt))
        self.dnb = np.zeros(int(self.T/self.dt))
        self.dna = np.zeros(int(self.T/self.dt))
        self.reward = np.zeros(int(self.T/self.dt))
        self.spread = np.zeros(int(self.T/self.dt))
        self.dw = np.zeros(int(self.T/self.dt))

        self.s[0] = self.s0
        self.q[0] = 0
        self.c[0] = 0
        self.w[0] = 0
        self.bid[0] = None
        self.ask[0] = None
        self.dnb[0] = 0
        self.dna[0] = 0
        self.reward[0] = 0
        self.dw[0] = 0

        self.low = config.low
        self.high = config.high

    def reset(self):
        self.s = np.zeros(int(self.T/self.dt))
        self.c = np.zeros(int(self.T/self.dt))
        self.w = np.zeros(int(self.T/self.dt))
        self.bid = np.zeros(int(self.T/self.dt))
        self.ask = np.zeros(int(self.T/self.dt))
        self.dnb = np.zeros(int(self.T/self.dt))
        self.dna = np.zeros(int(self.T/self.dt))
        self.reward = np.zeros(int(self.T/self.dt))
        self.dw = np.zeros(int(self.T/self.dt))

        self.s[0] = self.s0
        self.q[0] = 0
        self.c[0] = 0
        self.w[0] = 0
        self.bid[0] = None
        self.ask[0] = None
        self.dnb[0] = 0
        self.dna[0] = 0
        self.reward[0] = 0
        self.dw[0] = 0
    
    def step(self, t):
        # get new midprice
        inc = (1 if np.random.uniform() < 0.5 else -1)
        s = self.round_to_tick(self.s[t-1] + self.sigma*np.sqrt(self.dt)*inc)
        self.s[t] = s

    def get_state(self, t):
        # return np.array([self.preprocess(self.s[t], self.low[0], self.high[0]), 
        #         self.preprocess(self.q[t], self.low[1], self.high[1]),
        #         self.preprocess((self.T//self.dt - t)*self.dt, self.low[2], self.high[2])], dtype=np.float32)

        return np.array([ 
                 self.preprocess(self.q[t], self.low[0], self.high[0]),
                 self.preprocess((self.T//self.dt - t)*self.dt, self.low[1], self.high[1])], dtype=np.float32)
        
    def get_midprice(self, t):
        return self.s[t]

    def get_w(self, t):
        return self.w[t]

    def set_w(self, w, t):
        self.w[t] = w

    def set_dw(self, dw, t):
        self.dw[t] = dw
        
    def log_agentmetadata(self,bid, ask, dnb, dna, q, c, w, reward, t):
        self.bid[t] = bid
        self.ask[t] = ask
        self.dnb[t] = dnb
        self.dna[t] = dna
        self.q[t] = q
        self.c[t] = c
        self.w[t] = w
        self.reward[t] = reward
        
    def flush(self, logger_dir, experiment_name):
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
       
        data = {
            'midprice': self.s,
            'inv': self.q,
            'cash': self.c,
            'wealth': self.w,
            'bid_order': self.bid,
            'ask_order': self.ask,
            'dnb': self.dnb,
            'dna': self.dna,
            'reward': self.reward
        }
        
        df = pd.DataFrame(data).iloc[:-1]
        
        files = [file for file in os.listdir(logger_dir) if os.path.isfile(os.path.join(logger_dir, file)) and file.endswith('.csv')]
        index = len(files)
        df.to_csv(f"{logger_dir}/{index}_{experiment_name}.csv")


    def ulb(self, val, lb, ub):
        return max(min(val, ub), lb)

    def minmax_norm(self, val, lb, ub):
        return (val-lb)/(ub-lb)

    def preprocess(self, val, lb, ub):
        # return self.minmax_norm(self.ulb(val, lb, ub), lb, ub)
        return self.ulb(val, lb, ub)

    def round_to_tick(self, p):
        dp = len(str(self.price_tick)) - 2
        return round(p, dp)