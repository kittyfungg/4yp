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
    def __init__(self, R_max, bs, meta_gamma, inner_gamma, step, epsilon):
        self.bs = bs
        self.meta_gamma = meta_gamma
        self.inner_gamma = inner_gamma
        self.epsilon = epsilon

        self.m = int(math.ceil(math.log(1 / ((1-self.meta_gamma))) / (1-self.meta_gamma)))   #calculate m number
        self.Rmax = R_max * (1-meta_gamma**(self.m))/(1-meta_gamma)
        self.interval = int(R_max//step + 2)           #number of intervals for reward
        self.Q0 = round(R_max  / (1 - self.meta_gamma), 2)
        
        #meta-s = [oppo_act, our_act, our_r, t], meta-a = our_act
        ##meta-s = [oppo_act, our_act, our_r], meta-a = our_act
        ##self.ns = 2 * 2 * self.interval * self.interval
        #it's 14 since the best performance yields 0.9 + 0.5 = 1.4 reward
        self.ns = 2 * 2 * (self.interval)
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
            ##index = meta[:, 3] + meta[:, 2]* int(self.interval) + meta[:, 1] * (self.interval * int(self.interval)) + meta[:,0] * (2 * self.interval* int(self.interval))
            index = meta[:, 2] * 10 + meta[:, 1]* int(self.interval) + meta[:, 0] * (2 * int(self.interval))
        if obj == "a":
            index = meta

        return index
    
    def select_action(self, state):
        rand_from_poss_max = np.zeros(self.bs) 
        if np.random.random() < 1-self.epsilon:   
            action = np.random.randint(self.na, size=(self.bs, ))
        
        else:
            #find maximum action index, given state, makes sure if indices have same Q value, randomise
            lis = self.Q[range(self.bs), self.find_meta_index(state, "s").astype(int), :]
            for b in range(self.bs):
                if len(np.argwhere(lis[b] == np.max(lis[b]))) < 2:   #when there's only 1 max value
                    rand_from_poss_max[b] = np.argwhere(lis[b] == np.max(lis[b]))
                else:
                    rand_from_poss_max[b] = np.random.choice(np.argwhere(lis[b] == np.max(lis[b])).squeeze())
                                      
            action = rand_from_poss_max
        return action  #returns action from action index
    
                                
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

