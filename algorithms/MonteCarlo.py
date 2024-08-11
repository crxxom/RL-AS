import numpy as np
import random
from .TemporalDifference import QTable
import os
import pickle

class MonteCarloControl:
    def __init__(self, env, config):             

        self.env = env
        self.state_size = tuple(config.bins)
        self.action_size = config.N_ACTIONS  # 1-dimensional discrete action space
        
        self.low = config.low
        self.high = config.high
        self.grid = [np.linspace(config.low[i], config.high[i], config.bins[i]) for i in range(len(config.low))]
        
        # initalize random policy
        self.policy = QTable(self.state_size, self.action_size, self.grid)
        self.policy.q_table[:] = 1/self.action_size

        self.QTable = QTable(self.state_size, self.action_size, self.grid)

        # Learning parameters
        self.gamma = config.RLgamma
        self.alpha = config.alpha  # learning rate

        # configs
        self.model_save_freq = config.model_save_freq
        self.model_path = config.model_path
        
        self.initial_timesteps = config.initial_timesteps

        if self.initial_timesteps != 0:
            self.epsilon*=self.epsilon_decay_rate*(self.initial_timesteps/3400)
    
    def reset_episode(self):
        state, info = self.env.reset()
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)
        
        return state
    
    def reset_exploration(self, epsilon=None):
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def load(self, path=None):
        if path is None:
            path = self.model_path
            
        path += '/baseline'

        try:
            self.initial_timesteps = int(path.split('/')[-1].split('_')[-1])
        except:
            self.initial_timesteps = 0
            
        with open(path, 'rb') as file:
            self.policy.q_table = pickle.load(file)

        with open(path+'_qtable', 'rb') as file:
            self.QTable.q_table = pickle.load(file)

        
    def save(self, path=None, timesteps=0):
        
        if path is None:
            path = self.model_path

        if not os.path.exists(path):
            os.makedirs(path)

        path += '/baseline'
        
        if timesteps !=0 or self.initial_timesteps!=0:
            path += '_' + str(timesteps + self.initial_timesteps)
        
        with open(path, 'wb') as file:
            pickle.dump(self.policy.q_table, file)
            
        with open(path+'_qtable', 'wb') as file:
            pickle.dump(self.QTable.q_table, file)
    
    def choose_action(self, state):
        n = random.uniform(0, sum(self.policy.get_state_values(state)))
        top_range = 0
        for i, prob in enumerate(self.policy.get_state_values(state)):
            top_range += prob
            if n < top_range:
                action = i
                break 
        return action

    def run_episode(self):
        episode = []
        terminated = False
        truncated = False
        n_state, info = self.env.reset()

        while not terminated or truncated:
            timestep = []
            timestep.append(self.QTable.encode_state(n_state))
            action = self.choose_action(n_state)
            n_state, reward, terminated, truncated, info = self.env.step(action)
            timestep.append(action)
            timestep.append(reward)
            episode.append(timestep)
        
        return episode

    
    def learn(self, epsilon=0.01, episodes=10000):
        returns = {} 
        
        for _ in range(episodes): 
            G = 0 
            episode = self.run_episode()
            for i in reversed(range(0, len(episode))):   
                s_t, a_t, r_t = episode[i] 
                state_action = (s_t, a_t)
                G = self.gamma*G + r_t 
                
                if not state_action in [(x[0], x[1]) for x in episode[0:i]]: # first occurance of state action pair in the episode
                    if returns.get(state_action):
                        returns[state_action].append(G)
                    else:
                        returns[state_action] = [G]   

                    new_qval = sum(returns[state_action]) / len(returns[state_action]) 
                    self.QTable.update(s_t, a_t, new_qval, encoded=True)

                    Q_list = self.QTable.get_state_values(s_t, encoded=True)
                    indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                    max_Q = random.choice(indices)
                    
                    A_star = max_Q 
                    
                    for i, q_a in enumerate(self.policy.get_state_values(s_t, encoded=True)):
                        if i == A_star:
                            new_prob = 1 - epsilon + (epsilon / abs(sum(self.policy.get_state_values(s_t, encoded=True))))
                            self.policy.update(s_t, i, new_prob,  encoded=True)
                        else:
                            new_prob = (epsilon / abs(sum(self.policy.get_state_values(s_t, encoded=True))))
                            self.policy.update(s_t, i, new_prob, encoded=True)

    def predict(self, state):
        encoded_state = self.policy.encode_state(state)
        return self.choose_action(encoded_state), {}


