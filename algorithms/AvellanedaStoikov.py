import numpy as np

class AvellanedaStoikovAgent:
    def __init__(self,env, config):
        
        self.env = env
        self.config = config
        
        self.gamma = config.gamma
        self.sigma = config.sigma
        self.k = config.k
        
    def choose_action(self, state):
        s = self.env.Market.get_midprice(self.env.t)
        q = state[0]
        T_t = state[1]
        # reservation price
        r = s - q*self.gamma*self.sigma**2*(T_t) 
        # optimal spread
        delta = self.gamma*self.sigma**2*(T_t) + 2/self.gamma*np.log(1+self.gamma/self.k)
        return [r-delta/2, r+delta/2]

    def run(self):
        n_state, info = self.env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = self.choose_action(n_state)
            n_state, reward, terminated, truncated, info = self.env.step(action)