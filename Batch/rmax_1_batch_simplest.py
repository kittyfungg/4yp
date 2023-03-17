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
    def __init__(self, R_max, bs, meta_steps, meta_gamma, inner_gamma, radius, epsilon, rmax_error):
        self.bs = bs
        self.meta_gamma = meta_gamma
        self.inner_gamma = inner_gamma
        self.epsilon = epsilon
        self.rmax_error = rmax_error   #rmax_error for discretization error
        self.radius = radius               #added decimal place for Q & R matrix dimension

        self.m = int(math.ceil(math.log(1 / (self.rmax_error * (1-self.meta_gamma))) / (1-self.meta_gamma)))   #calculate m number
        self.Rmax = R_max * self.m
        self.Q0 = round(self.Rmax  / (1 - self.meta_gamma), 2)
        
        self.meta_steps= meta_steps
        self.ns = 2 * 2 * meta_steps
        #self.ns = 2 * 2 
        self.na = 2
        
        #Q = [bs, meta_state, meta_action], **2 for 2 player  
        self.Q = np.ones((self.bs, self.ns, self.na)) * self.Q0
        self.R = np.zeros((self.bs, self.ns, self.na))
        self.nSA = np.zeros((self.bs, self.ns, self.na))
        self.nSAS = np.zeros((self.bs, self.ns, self.na, self.ns))
    
    def find_meta_index(self, meta, obj):
        #obj can only be "s" / "a"
        #meta is of size [bs=1028, action_dimension=9]
        index = np.zeros(self.bs) #initialise index
        
        if obj == "s":
            #index[:] = (meta[:, 1]//self.radius) + (meta[:, 0]//self.radius)* (2)
            index[:] = meta[:, 2] + meta[:, 1]* self.meta_steps + meta[:, 0] * (2 * self.meta_steps)

        if obj == "a":
            index = meta

        return index
    
                                
    def update(self, memory, state, action, next_state):
        action_mapped = self.find_meta_index(action, "a").astype(int)
        state_mapped = self.find_meta_index(state, "s").astype(int)
        next_state_mapped = self.find_meta_index(next_state, "s").astype(int)
        
        for i in range(self.bs):
            if self.nSA[i, state_mapped[i], action_mapped[i]] <= self.m:
                self.nSA[range(self.bs), state_mapped , action_mapped] += 1
                self.nSAS[range(self.bs), state_mapped , action_mapped, next_state_mapped] +=1
                self.R[i, state_mapped[i], action_mapped[i]] = memory.rewards[-1][i] + self.meta_gamma *self.R[i, state_mapped[i], action_mapped[i]]
                
            if self.nSA[i, state_mapped[i], action_mapped[i]] == self.m:
                for m in range(self.m):
                    for s in range(self.ns):
                        for a in range(self.na):
                            if self.nSA[i, s, a] >= self.m:
                                q = self.R[i, s, a]/ self.nSA[i,s,a]

                                for next_s in range(self.ns):
                                    transition = self.nSAS[i, s, a, next_s]/ self.nSA[i, s, a]
                                    #q += transition * np.max(self.Q[i, next_s, :])
                                    masked_arr = np.ma.masked_where(self.Q[i, next_s] == self.Q0, self.Q[i, next_s])
                                    if masked_arr.mask.all():     #for the first time when everything is still Q0
                                        q += 0
                                    else:
                                        q += self.meta_gamma * transition * np.ma.masked_where(self.Q[i, next_s] == self.Q0, self.Q[i, next_s]).max()
                                self.Q[i, s, a] = q