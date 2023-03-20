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
        
class RmaxAgentTraj:
    def __init__(self, R_max, bs, meta_steps, meta_gamma, inner_gamma, epsilon, hist_step):
        self.bs = bs
        self.meta_gamma = meta_gamma
        self.inner_gamma = inner_gamma
        self.epsilon = epsilon     #error bound
        self.delta = 0.2        #with probability >=1-delta, the policy is optimal/ algorithm’s failure probability.
        self.vi = int(np.log(1 / (self.epsilon * (1-self.meta_gamma))) / (1-self.meta_gamma))
        self.Rmax = R_max
        self.Q0 = round(R_max  / (1 - self.meta_gamma), 2)
        
        self.meta_steps= meta_steps
        self.hist_step = hist_step
#        self.ns = (4**hist_step)
        self.ns = (4**(hist_step)) * meta_steps
        self.na = 2
        
        self.m = int(self.ns * math.log(1/self.delta) / (epsilon * (1-meta_gamma))**2)   #calculate m number
        
        #Q = [bs, meta_state, meta_action], **2 for 2 player  
        self.Q = np.ones((self.bs, self.ns, self.na)) * self.Q0
        self.R = np.zeros((self.bs, self.ns, self.na))
        self.nSA = np.zeros((self.bs, self.ns, self.na))
        self.nSAS = np.zeros((self.bs, self.ns, self.na, self.ns))
    
    def find_meta_index(self, meta, obj):
        #obj can only be "s" / "a"
        index = np.zeros(self.bs) #initialise index
        
        if obj == "s":
#            for i in range(len(meta[0])):
#                index[:] += meta[:, -i-1]* ( (2**i))
                
             index[:] = (meta[:, -1]) 
             for i in range(len(meta[0])-1):
                 index[:] += meta[:, -i-2]* ( (2**i) * self.meta_steps)
                
        if obj == "a":
            index = meta

        return index
                                
    def update(self, memory, state, action, next_state):
        action_mapped = self.find_meta_index(action, "a").astype(int)
        state_mapped = self.find_meta_index(state, "s").astype(int)
        next_state_mapped = self.find_meta_index(next_state, "s").astype(int)
        
        for i in range(self.bs):
            if self.nSA[i, state_mapped[i], action_mapped[i]] < self.m:
                self.nSA[range(self.bs), state_mapped , action_mapped] += 1
                self.nSAS[range(self.bs), state_mapped , action_mapped, next_state_mapped] +=1
                self.R[i, state_mapped[i], action_mapped[i]] += memory.rewards[-1][i] 
                
                if self.nSA[i, state_mapped[i], action_mapped[i]] == self.m:
                    for vi in range(self.vi):
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
            else:
                if memory.rewards[-1][i] < 0.6:
                    self.R[i, state_mapped[i], action_mapped[i]] *= 0.75 