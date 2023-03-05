import math
import numpy as np
import random

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
    def __init__(self, env, R_max, bs, meta_gamma, inner_gamma, radius, epsilon, rmax_error):
        self.bs = bs
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
        
        #Q = [bs, meta_state, meta_action], **2 for 2 player  
        self.Q = np.ones((self.bs, self.meta_size ** 2, self.meta_size)) * self.Q0
        self.R = np.zeros((self.bs, self.meta_size ** 2, self.meta_size))
        self.nSA = np.zeros((self.bs, self.meta_size ** 2, self.meta_size))
        self.nSAS = np.zeros((self.bs, self.meta_size ** 2, self.meta_size, self.meta_size ** 2))
    
    def find_meta_index(self, meta):
        #meta is of size [bs=1028, 0, action_dimension=9]
        index = np.zeros(self.bs) #initialise index
        
        if len(meta.shape) > 2:
            meta = meta.reshape(self.bs, len(meta[1])+len(meta[2]))       #reshape it so [bs,2,2] --> [bs,4]
    
        #for every digit in meta-action/state:
        for i in range(len(meta[1])):
            index[:] += (meta[:, i]//self.radius) * (self.poss_combo ** i)

        return index
                
    def index_to_table(self, env, index, agent_size):
        #VERY POORLY WRITTEN HARDCODING
        #returns a table of size [bs, agent_size, num_states, num_actions], given index
        #agent_size either 1/2, 1 for action table, 2 for state table
        Q_size = agent_size * env.d * env.num_actions 
        reconstruct = np.zeros((self.bs, Q_size))
        
        q3, mod3 = np.divmod(index, self.poss_combo**(Q_size-1))
        reconstruct[:, (Q_size-1)] = q3*self.radius

        q2, mod2 = np.divmod(mod3, self.poss_combo**(Q_size-2))
        reconstruct[:, (Q_size-2)] = q2*self.radius

        q1, mod1 = np.divmod(mod2, self.poss_combo**(Q_size-3))
        reconstruct[:, (Q_size-3)] = q1*self.radius

        q0, _ = np.divmod(mod1, self.poss_combo**(Q_size-4))
        reconstruct[:, (Q_size-4)] = q0*self.radius
        
        #return np.reshape(reconstruct, (self.bs, agent_size, env.d, env.num_actions)) ignored env.d since =1
        return np.reshape(reconstruct, (self.bs, agent_size, env.num_actions))
    
    def select_action(self, env, state, epsilon = None):
        if epsilon == None:
            epsilon = self.epsilon
        #set epsilon=-1 if we just want to get the max Q value without epsilon-greedy
        
        rand_from_poss_max = np.zeros(self.bs) 
        if np.random.random() < epsilon:
            action = self.index_to_table(env, np.random.randint(self.meta_size-1, size=(self.bs)), 1)
        
        else:
            #find maximum action index, given state, makes sure if indices have same Q value, randomise

            lis = self.Q[range(self.bs), self.find_meta_index(state).astype(int), :]
            for b in range(self.bs):
                rand_from_poss_max[b] = np.random.choice(np.argwhere(lis[b] == np.max(lis[b])).squeeze())
                                      
            action = self.index_to_table(env, rand_from_poss_max, 1) 
        return action     #returns action index
    
                                
    def update(self, env, memory, state, action, next_state):
        action_mapped = self.find_meta_index(action).astype(int)
        state_mapped = self.find_meta_index(state).astype(int)
        next_state_mapped = self.find_meta_index(next_state).astype(int)
        
        
        #FOR nSA<m CASE:
        #filter for nSA<m
        mask00 = self.nSA[np.arange(self.bs), state_mapped, action_mapped] < self.m
        
        #filter for nSA=0
        mask01 = mask00 & (self.nSA[np.arange(self.bs), state_mapped, action_mapped] == 0)
        #filter for 0<nSA<m
        mask02 = mask00 & (self.nSA[np.arange(self.bs), state_mapped, action_mapped] > 0)
        
        self.R[mask01, state_mapped, action_mapped] = memory.rewards[-1]   #Input R as inner reward
        if all(mask02)!= False:
            self.R[mask02, state_mapped, action_mapped] = memory.rewards[-1] + self.meta_gamma * self.R[mask02, state_mapped, action_mapped] 
        self.nSA[mask00, state_mapped, action_mapped] += 1
        self.nSAS[mask00, state_mapped, action_mapped, next_state_mapped] += 1
        
        #FOR nSA>=m CASE:
        mask10 = self.nSA[np.arange(self.bs), state_mapped, action_mapped] >= self.m
        for i in range(self.m):

            for s in range(self.meta_size * 2):

                for a in range(self.meta_size):

                    mask11 = self.nSA[:, s, a] >= self.m
                    q = self.R[mask11, s, a]/self.nSA[mask11, s, a]

                    for next_s in range(env.d * 2):
                        transition = self.nSAS[mask11, s, a, next_s]/self.nSA[mask11, s, a]
                        q += transition * np.amax(self.Q[:,next_s,:], axis=1)

                    self.Q[mask11, s, a] = q
                    
#         if memory.rewards[-1]!=1:
#             self.R[mask10] = 0
#         if self.nSA[range(self.bs), state_mapped, action_mapped] < self.m:
            
#             if self.nSA[range(self.bs), state_mapped, action_mapped] == 0:   #if the s-a pair hasn't been visited before,
#                 self.R[range(self.bs), state_mapped, action_mapped] = memory.rewards[-1]   #Input R as inner reward
#             else:                                           #if visited, R builds on previous R
#                 self.R[range(self.bs), state_mapped, action_mapped] = memory.rewards[-1] + self.meta_gamma * self.R[:, state_mapped, action_mapped] 
#                 #try no discount factor
#                 #self.R[state_mapped][action_mapped] += memory.rewards[-1]

#             self.nSA[range(self.bs), state_mapped, action_mapped] += 1
#             self.nSAS[range(self.bs), state_mapped, action_mapped, next_state_mapped] += 1

        #Update Q if it's visited m times
        #else:
#             for i in range(self.m):

#                 for s in range(self.meta_size * 2):

#                     for a in range(self.meta_size):

#                         if self.nSA[s][a] >= self.m:

#                             q = (self.R[range(self.bs),s,a]/self.nSA[range(self.bs),s,a])

#                             for next_s in range(env.d * 2):
#                                 transition = self.nSAS[range(self.bs),s,a,next_s]/self.nSA[range(self.bs),s,a]
#                                 q += (transition * np.amax(self.Q[range(self.bs),next_s,:], dim=2))
                            
#                             self.Q[range(self.bs), state_mapped, action_mapped] = q
                            
#         if memory.rewards[-1]!=1:
#             self.R[range(self.bs), state_mapped, action_mapped] = 0
            