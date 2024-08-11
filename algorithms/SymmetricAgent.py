import numpy as np

class SymmetricAgent:
    def __init__(self,env, config, displacement=0.5):
        
        self.env = env
        self.config = config
        self.displacement = displacement
        
    def choose_action(self, state):
        s = self.env.Market.get_midprice(self.env.t)
        return [s-self.displacement, s+self.displacement]

    def run(self):
        n_state, info = self.env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = self.choose_action(n_state)
            n_state, reward, terminated, truncated, info = self.env.step(action)