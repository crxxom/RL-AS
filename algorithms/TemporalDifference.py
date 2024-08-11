import numpy as np
import pickle
import os

def create_tile_coding_table(low, high, bins, N_TILES):
    # create tile coded Qtable    
    tiling_spec = []
    for j in range(N_TILES):
        spec = []
        spec.append(tuple(bins))
        offset = []
        for i in range(len(low)):
            inc = (high[i] - low[i])/bins[i]/ N_TILES
            offset.append(inc*j)
        spec.append(tuple(offset))
        
        tiling_spec.append(tuple(spec))
        
    return low, high, tiling_spec
    
def create_tilings(low, high, tiling_specs):
    return [create_tiling_grid(low, high, bins, offsets) for bins, offsets in tiling_specs]

def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] + offsets[dim] for dim in range(len(bins))]
    return grid

def discretize(sample, grid):
    return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))

def tile_encode(sample, tilings, flatten=False):
    encoded_sample = [discretize(sample, grid) for grid in tilings]
    return np.concatenate(encoded_sample) if flatten else encoded_sample

class QTable:
    def __init__(self, state_size, action_size, grid, alpha=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        self.grid = grid
        self.alpha = alpha
        
    def encode_state(self, state):
        encode_state = [np.digitize(state[i],self.grid[i])-1 for i in range(len(state))]
        return tuple(encode_state)

    def get_state_values(self, state, encoded=False):
        old_state = state
        if not encoded:
            state = self.encode_state(state)
        # try:
        #     a = self.q_table[tuple(state)]
        # except: 
        #     print('old_state:',old_state)
        #     print('new_state:', state)
        return self.q_table[tuple(state)]

    def get(self, state, action, encoded=False):
        if not encoded:
            state = self.encode_state(state)
        
        return self.q_table[tuple(state + (action,))]
        
    def update(self, state, action, value, encoded=False):
        if not encoded:
            state = self.encode_state(state)
        value_ = self.q_table[tuple(state + (action,))]
        value = self.alpha * value + (1.0 - self.alpha) * value_
        self.q_table[tuple(state + (action,))] = value
        
            
class TileCodedQTable:
    def __init__(self, low, high, tiling_specs, action_size, alpha=0.1):
        
        self.tilings = create_tilings(low, high, tiling_specs)
        self.state_size = [tuple(len(splits)+1 for splits in tiling_grid) for tiling_grid in self.tilings]
        self.action_size = action_size
        self.q_table = [QTable(state_size, self.action_size, self.tilings[i]) for i, state_size in enumerate(self.state_size)]
        self.alpha = alpha

    def encode_state(self,state):
        return tile_encode(state, self.tilings)
        
    def get(self, state, action):
        encoded_state = tile_encode(state, self.tilings)
        # encoded states looks like this [(1,2,3), (0,1,2), (3,2,1)]
        # len(encoded_state) = N_TILES, len(encoded_state[0]) = state dimension

        # get mean of all tiles -> you can have destinated weightings if needed
        value = 0.0
        for idx, q_table in zip(encoded_state, self.q_table):
            value += q_table.q_table[tuple(idx + (action,))]
        value /= len(self.q_table)
        return value
    
    def update(self, state, action, value):
        
        encoded_state = tile_encode(state, self.tilings)
        # update only by a small step-size hyperparameter
        for idx, q_table in zip(encoded_state, self.q_table):
            value_ = q_table.q_table[tuple(idx + (action,))]  # current value
            q_table.q_table[tuple(idx + (action,))] = self.alpha * value + (1.0 - self.alpha) * value_

# ========================================================================
# ALGORITHMS

class QLearning:
    def __init__(self, env, QTable, config):
        
        self.env = env
        self.QTable = QTable
        self.state_size = QTable.state_size           # list of state sizes for each tiling
        self.action_size = self.env.action_space.n  # 1-dimensional discrete action space

        # Learning parameters
        self.gamma = config.RLgamma
        self.alpha = config.alpha  # learning rate
        self.epsilon = self.initial_epsilon = config.epsilon  # initial exploration rate
        self.epsilon_decay_rate = config.epsilon_decay_rate   # how quickly should we decrease epsilon
        self.min_epsilon = config.min_epsilon

        # configs
        self.mode = 'test'
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
            self.QTable.q_table = pickle.load(file)

        if self.initial_timesteps != 0 and self.epsilon == self.initial_epsilon:
            self.epsilon*=self.epsilon_decay_rate*(self.initial_timesteps/3400)
        
    def save(self, path=None, timesteps=0):
        
        if path is None:
            path = self.model_path

        if not os.path.exists(path):
            os.makedirs(path)

        path += '/baseline'
        
        if timesteps !=0 or self.initial_timesteps!=0:
            path += '_' + str(timesteps + self.initial_timesteps)
        
        with open(path, 'wb') as file:
            pickle.dump(self.QTable.q_table, file)
        
    def get_action(self, state):
        encoded_state = self.QTable.encode_state(state)
        Q_s = np.array([self.QTable.get(state, action) for action in range(self.action_size)])
        greedy_action = Q_s.argmax()
        action = 0
        if self.mode == 'train':
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                action = np.random.randint(0, self.action_size)
            else:
                action = greedy_action

        return action

    def predict(self, state):
        encoded_state = self.QTable.encode_state(state)
        Q_s = np.array([self.QTable.get(state, action) for action in range(self.action_size)])
        greedy_action = Q_s.argmax()
        return greedy_action, {}
            
    def learn(self, total_timesteps=100000):
        self.mode = 'train'
        self.run(total_timesteps=total_timesteps)
        
    def run(self, total_timesteps=100000):
        
        steps = 0
        truncation = True
        termination = True
        state=None
        while steps < total_timesteps:
            if truncation or termination: # reset env when terminated
                state = self.reset_episode()

            action = self.get_action(state)
            state, reward, termination, truncation, info = self.env.step(action)
            
            if self.mode=='train':
                Q_s = [self.QTable.get(state, action) for action in range(self.action_size)]
                value = self.gamma * max(Q_s) + reward
                self.QTable.update(state, action, value)
                
            steps += 1
            # save table periodically
            if steps % self.model_save_freq == 0 and self.mode=='train':
                self.save(timesteps=steps)
        

class SARSA(QLearning):
    def __init__(self, env, QTable, config,**kwargs):
        super().__init__(env, QTable, config,**kwargs)
        
    def run(self, total_timesteps=100000):
        
        steps = 0
        truncation = True
        termination = True
        state=None
        prev_state=None
        while steps < total_timesteps:
            if truncation or termination: # reset env when terminated
                state = self.reset_episode()
                prev_state=None

            action = self.get_action(state)
            state, reward, termination, truncation, info = self.env.step(action)
            
            if self.mode=='train' and prev_state is not None:
                value = self.gamma * self.QTable.get(state, action) + reward
                self.QTable.update(prev_state, prev_action, value)
                
            steps += 1
            prev_state = state
            prev_action = action
            # save table periodically
            if steps % self.model_save_freq == 0 and self.mode=='train':
                self.save(timesteps=steps)
