import math
import numpy as np
import torch
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        
class RmaxAgent:
    def __init__(self, env, R_max, meta_gamma, inner_gamma, radius, epsilon, rmax_error):
        self.meta_gamma = meta_gamma
        self.inner_gamma = inner_gamma
        self.epsilon = epsilon
        self.rmax_error = rmax_error   #rmax_error for discretization error
        self.radius = radius               #added decimal place for Q & R matrix dimension
        
        self.m = int(math.ceil(math.log(1 / (self.rmax_error * (1-self.meta_gamma))) / (1-self.meta_gamma)))   #calculate m number
        self.Rmax = R_max * self.m
        self.Q0 = round(self.Rmax  / (1 - self.meta_gamma), 2)
        
        #no of possible combinations for an inner Q value
        self.poss_combo = math.ceil((1//(1-inner_gamma)) / radius) +1
        self.meta_size = self.poss_combo ** (env.d * env.num_actions)
        
        #Q = [meta_state, meta_action], **2 for 2 player  
        self.Q = torch.ones(self.meta_size ** 2, self.meta_size).mul(self.Q0).to(device) 
        self.R = torch.zeros(self.meta_size ** 2, self.meta_size).to(device)
        self.nSA = torch.zeros(self.meta_size ** 2, self.meta_size).to(device)
        self.nSAS = torch.zeros(self.meta_size ** 2, self.meta_size, self.meta_size ** 2).to(device)
    
        
    def select_action(self, env, state, epsilon = None):
        if epsilon == None:
            epsilon = self.epsilon
        #set epsilon=-1 if we just want to get the max Q value without epsilon-greedy
        if np.random.random() < epsilon:
            action = self.index_to_table(env, random.randint(0, self.meta_size-1), 1)
        else:
            #find maximum action index, given state, makes sure if indices have same Q value, randomise
            lis = self.Q[self.find_meta_index(torch.flatten(state)),:]
            rand_from_poss_max = random.choice(torch.argwhere(lis == torch.max(lis)).to(device)) 
            action = self.index_to_table(env, rand_from_poss_max.item(), 1) 
        return action     #returns action index
    
    def find_meta_index(self, meta):
        index = int(0) #initialise index

        #for every digit in metastate/ metaaction:
        for i in range(list(meta.size())[0]):
            index += (meta[i]//self.radius) * (self.poss_combo ** i)
        return int(index)
                
    def index_to_table(self, env, index, agent_size):
        #returns a table of size [agent_size, num_states, num_actions], given index
        #agent_size either 1/2, 1 for action table, 2 for state table
        Q_size = agent_size * env.d * env.num_actions 
        reconstruct = torch.empty(Q_size).to(device)
        for i in reversed(range(Q_size)):
            if index >= self.poss_combo**i:
                q, mod = divmod(index, self.poss_combo**i)
                reconstruct[i] = q * self.radius     #recover original discretized value
                index = mod
            else:
                q = 0
                mod = 0
                reconstruct[i] = 0
        
        return torch.reshape(reconstruct, (agent_size, env.d, env.num_actions)).to(device)
                                
    def update(self, env, memory, state, action, next_state):
        action_mapped = self.find_meta_index( torch.flatten(action))
        state_mapped = self.find_meta_index( torch.flatten(state))
        next_state_mapped = self.find_meta_index( torch.flatten(next_state))
        
        if self.nSA[state_mapped][action_mapped] < self.m:
            
            if self.nSA[state_mapped][action_mapped] == 0:   #if the s-a pair hasn't been visited before,
                self.R[state_mapped][action_mapped] = memory.rewards[-1]   #Input R as inner reward
            else:                                           #if visited, R builds on previous R
                self.R[state_mapped][action_mapped] = memory.rewards[-1] + self.meta_gamma * self.R[state_mapped][action_mapped] 
                #try no discount factor
                #self.R[state_mapped][action_mapped] += memory.rewards[-1]

            self.nSA[state_mapped][action_mapped] += 1
            self.nSAS[state_mapped][action_mapped][next_state_mapped] += 1

        #Update Q if it's visited m times
        else:

            for i in range(self.m):

                for s in range(self.meta_size * 2):

                    for a in range(self.meta_size):

                        if self.nSA[s][a] >= self.m:

                            q = (self.R[s][a]/self.nSA[s][a])

                            for next_s in range(env.d * 2):
                                transition = self.nSAS[s][a][next_s]/self.nSA[s][a]
                                q += (transition * torch.max(self.Q[next_s,:]))
                            
                            self.Q[state_mapped][action_mapped] = q

                            
        if memory.rewards[-1]!=1:
            self.R[state_mapped][action_mapped] = -10
            