# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:25:12 2023

@author: hanne
"""

import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T 
import gym 
import matplotlib.pyplot as plt
import os
import pickle


class Memory():
    """
    Handles the storation of transitions (state, action, reward, state_, done)
    and the sampling of memories for the learning of the DDQN
    """
    def __init__(self, memory, input_shape):
        self.memory = memory #number of maximal transitions stored
        self.idx = 0
        shape = (memory,*input_shape) #input_shape is the shape of the observation
        
        self.states = np.zeros(shape, dtype = np.float32)
        self.states_ = np.zeros(shape,dtype = np.float32) #new states after taking an action
        self.actions = np.zeros([memory],dtype = np.int64)
        self.rewards = np.zeros([memory], dtype = np.float32)
        self.terminal = np.zeros([memory],dtype = np.uint8) #information about when an environment is done
        
    def sample_memories(self,batch_size):
        max_mem = min(self.idx, self.memory)
        sample_idx = np.random.choice(max_mem, size=batch_size, replace=False) #for choosing random transitions
        
        sample_states = self.states[sample_idx]
        sample_actions = self.actions[sample_idx]
        sample_rewards = self.rewards[sample_idx]
        sample_states_ = self.states_[sample_idx]
        sample_dones = self.terminal[sample_idx]
        
        return sample_states, sample_actions, sample_rewards, sample_states_,\
            sample_dones
    
    def store_transition(self,state, action, reward, state_, done):
        idx = self.idx%self.memory
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.states_[idx] = state_
        self.terminal[idx] = done
        
        self.idx += 1
        

class DDQN(nn.Module):
    """
    This class implement the basic functionality of the Double-Deep-Q-Network 
    """
    
    def __init__(self, learning_rate, input_dim, n_actions, name, checkpoint_dir):
        super(DDQN,self).__init__()
        
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name)
        
        
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.n_actions = n_actions
        
        #Network architecture
        self.layer1 = nn.Linear(*input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x):
        """
        Calculates values of output nodes for given observation x
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        
        return x
    
    def save(self):
        #print(f"---   saving model: {self.name}   ---")
        T.save(self.state_dict(),self.checkpoint_file)
        
    def load(self):
        #print(f"---   loading model {self.name}   ---")
        self.load_state_dict(T.load(self.checkpoint_file))
        
class Agent():
    def __init__(self, learning_rate, input_dim, n_actions, mem_size, batch_size,\
                 name, checkpoint_dir, gamma = 0.99, eps_max = 1, eps_min = 0.01,\
                     eps_dec = 5e-6, replace = 1000):
        
        self.learning_rate = learning_rate
        
        self.n_actions = n_actions
        self.actions = [i for i in range(n_actions)]
        self.input_dim = input_dim
        
        self.mem_size = mem_size
        self.batch_size = batch_size
        
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        
        self.gamma = gamma
        
        self.epsilon = eps_max
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        
        self.replace = replace #after how many learning steps should the target-network be updated
        self.learning_counter = 0
        
        self.online_q = DDQN(learning_rate,input_dim,n_actions,name,checkpoint_dir) #two networks for deciding which action to take
        self.target_q = DDQN(learning_rate,input_dim,n_actions,name,checkpoint_dir) # for calculating the target_value
        
        self.memory = Memory(mem_size,input_dim)
    
    def get_action(self, state):
        if np.random.random() < self.epsilon: #select random action with regards to epsilon
            action = np.random.choice(self.actions)
        else:
            state = T.tensor(state, dtype=T.float).to(self.online_q.device)
            action = T.argmax(self.online_q.forward(state)).item()
        return action
    
    def store_transition(self,s,a,r,s_,d):
        self.memory.store_transition(s, a, r, s_, d)
    
    
    def sample_memory(self):
        s, a, r, s_, d = self.memory.sample_memories(self.batch_size)
        
        #convert numpy array to Torch tensor
        states = T.tensor(s).to(self.online_q.device)
        actions = T.tensor(a).to(self.online_q.device)
        rewards = T.tensor(r).to(self.online_q.device)
        states_ = T.tensor(s_).to(self.online_q.device)
        dones = T.tensor(d, dtype = T.bool).to(self.online_q.device)
        
        return states, actions, rewards, states_, dones
    
    def save_model(self):
        print(f"---   saving model: {self.name}   ---")
        self.online_q.save()
        self.target_q.save()
       
        #  saving epsilon
        checkpoint_epsilon = os.path.join(self.checkpoint_dir,"epsilon")
        with open (checkpoint_epsilon, "wb") as file:
            pickle.dump(self.epsilon, file)
        
        
    def load_model(self):
        print(f"---   loading model: {self.name}   ---")
        self.online_q.load()
        self.target_q.load()
        
        #loading epsilon
        checkpoint_epsilon = os.path.join(self.checkpoint_dir,"epsilon")
        with open(checkpoint_epsilon, 'rb') as file:
            self.epsilon = pickle.load(file)
        
        
    def update_target_network(self):
        if self.learning_counter % self.replace == 0: #update target network only every replace-steps
            state_dict = self.online_q.state_dict()
            self.target_q.load_state_dict(state_dict)
      
            
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon - self.eps_dec > self.eps_min\
            else self.eps_min
   
    def learn(self,s,a,r,s_,d):
        
        self.store_transition(s, a, r, s_, d)
        
        if self.memory.idx < self.batch_size:
            return
        
        
        self.online_q.optimizer.zero_grad()
        
        self.update_target_network()
        
        
        states, actions, rewards, states_, dones = self.sample_memory()
        
        indices = np.arange(self.batch_size)
        
        q_pred = self.online_q.forward(states)[indices, actions]
        
        q_next = self.target_q.forward(states_)[indices]
        
        q_eval = self.online_q.forward(states_)[indices]
        
        
        eval_actions = q_eval.argmax(dim=1)
        
        q_next[dones] = 0.0
        
        test = q_next[indices, eval_actions]
        
        
        q_target = rewards + self.gamma * q_next[indices,eval_actions]
        
        loss = self.online_q.loss(q_pred,q_target).to(self.online_q.device)
        
        loss.backward()
        
        self.online_q.optimizer.step()
        self.learning_counter += 1
        self.decrement_epsilon()
        
        
        
    
        
        