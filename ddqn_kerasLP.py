from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size # Index of last memory item
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape)) 
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype) 
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32) # expected feature award for terminal stateis 0

    def store_transition(self, state, action, reward, state_, done): # state_ is the new state
        index = self.mem_cntr % self.mem_size # Keeps memory finite
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size): # sample some subset of the NN
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
        Dense(fc1_dims, input_shape=(input_dims,)),
        Activation('relu'),
        Dense(fc2_dims),
        Activation('relu'),
        Dense(n_actions)
    ])

    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')

    return model

class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.999995,  epsilon_end=0.01,
                 mem_size=1000000, fname='ddqn_LPmodel.h5', replace_target=100,
                 fc1_dims=256, fc2_dims=256):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, fc1_dims, fc2_dims)
        self.q_next = build_dqn(alpha, n_actions, input_dims, fc1_dims, fc2_dims)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state) # feed forward from defined architecture
            action = np.argmax(actions)
        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_next.predicton(new_state)
            q_eval = self.q_eval.predict(new_state)

            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma*q_next[batch_index, max_actions]*done  

            _ = self.q_eval.fit(state, q_target, verbose=0) # keras dumps a lot of output, so suppress it.

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
            
            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()

        def update_network_parameters(self):
            self.q_next.set_weights(self.q_eval.model.get_weights())

        def save_model(self):
            self.q_eval.save(self.model_file)

        def load_model(self):
            self.q_eval = load_model(self.model_file) 

            if self.epsilon <= self.epsilon_min:
                self.epsilon = self.epsilon_min