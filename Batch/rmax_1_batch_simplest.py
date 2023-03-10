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
        
        #FOR nSA<m CASE:
        #filter for nSA<m
        idx00 = np.argwhere(self.nSA[np.arange(self.bs), state_mapped, action_mapped] < self.m)
        
        #filter for nSA=0
        idx01 = np.argwhere((self.nSA[np.arange(self.bs), state_mapped, action_mapped] < self.m) & (self.nSA[np.arange(self.bs), state_mapped, action_mapped] == 0))
        #filter for 0<nSA<m
        idx02 = np.argwhere((self.nSA[np.arange(self.bs), state_mapped, action_mapped] < self.m) & (self.nSA[np.arange(self.bs), state_mapped, action_mapped] > 0))
        
        if len(idx01) > 0:
            self.R[idx01, state_mapped[idx01] , action_mapped[idx01]] = memory.rewards[-1][idx01]
            
        if len(idx02) > 0:
            self.R[idx02, state_mapped[idx02] , action_mapped[idx02]] = memory.rewards[-1][idx02] +  self.meta_gamma *self.R[idx02, state_mapped[idx02] , action_mapped[idx02]]
        
        if len(idx00) > 0:
            self.nSA[idx00, state_mapped[idx00] , action_mapped[idx00]] += 1
            self.nSAS[idx00, state_mapped[idx00] , action_mapped[idx00], next_state_mapped[idx00]] +=1
                   
        #FOR nSA>=m CASE:
        idx10 = np.argwhere(self.nSA[np.arange(self.bs), state_mapped, action_mapped] >= self.m)
        for i in range(self.m):

            idx11 = np.argwhere(self.nSA[:,:,:] >= self.m)
            if len(idx11) > 0:
                q = self.R[idx11[:, 0], idx11[:, 1], idx11[:, 2]]/ self.nSA[idx11[:, 0], idx11[:, 1], idx11[:, 2]]

                for next_s in range(self.ns):
                    transition = self.nSAS[idx11[:, 0], idx11[:, 1], idx11[:, 2], next_s] / self.nSA[idx11[:, 0], idx11[:, 1], idx11[:, 2]]
                    q += transition * np.amax(self.Q[idx11[:,0], next_s, :], axis=1)

                self.Q[idx11[:, 0], idx11[:, 1], idx11[:, 2]] = q

        idx12 = np.argwhere((self.R[np.arange(self.bs), state_mapped, action_mapped] < 1) & (self.nSA[np.arange(self.bs), state_mapped, action_mapped] >= self.m))
        if len(idx12) > 0:
            self.R[idx12, state_mapped[idx12], action_mapped[idx12]] = -10
