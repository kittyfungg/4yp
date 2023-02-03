#Rmax code, cuda not yet incompatible
import math
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def find_meta_index(meta, radius, poss_combo):
    index = int(0) #initialise index

    #for every digit in metastate/ metaaction:
    for i in range(list(meta.size())[0]):
        index += (meta[i]//radius) * (poss_combo ** i)
        
    return int(index.item())

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
    def __init__(self, env, R_max, meta_gamma, inner_gamma, radius, epsilon = 0.2):
        self.meta_gamma = meta_gamma
        self.inner_gamma = inner_gamma
        self.epsilon = epsilon
        self.radius = radius               #added decimal place for Q & R matrix dimension
        self.b = env.b
       
        #no of possible combinations for a inner Q value
        self.poss_combo = math.ceil((1//(1-inner_gamma)) / radius) +1
        self.meta_size = self.poss_combo ** (env.d * env.num_actions)
        
        #Q = [batch, meta_state, meta_action, *2 for 2 player  
        self.Q = torch.ones(self.b, self.meta_size * 2, self.meta_size).mul(R_max / (1 - self.meta_gamma)).to(device) 
        self.R = torch.ones(self.b, self.meta_size * 2, self.meta_size).to(device)
        self.nSA = torch.zeros(self.b, self.meta_size * 2, self.meta_size).to(device)
        self.nSAS = torch.ones(self.b, self.meta_size * 2, self.meta_size, self.meta_size * 2).to(device)
    
        self.val1 = []
        self.val2 = []  #This is for keeping track of rewards over time and for plotting purposes  
        self.m = int(math.ceil(math.log(1 / (self.epsilon * (1-self.meta_gamma))) / (1-self.meta_gamma)))   #calculate m number
        
    def select_action(self, env, state):
        #if np.random.random() > (1-self.epsilon):
        #    action = env.action_space.sample()
        #else:
        action = torch.argmax(self.Q[:,find_meta_index(torch.flatten(state), self.radius, self.poss_combo),:], 1)    #find maximum from the third dimension(action dimension)
        return action     #returns action of length b
    
    def update(self, env, memory, state, action, next_state):
        #for each batch
        for batch in range(self.b):
            action_mapped = find_meta_index( torch.flatten( action[batch] ), self.radius, self.poss_combo)
            state_mapped = find_meta_index( torch.flatten( state[batch] ), self.radius, self.poss_combo)
            next_state_mapped = find_meta_index( torch.flatten( next_state[batch] ), self.radius, self.poss_combo)

            if self.nSA[batch][state_mapped][action_mapped] < self.m:
                
                self.nSA[batch][state_mapped][action_mapped] += 1
                self.R[batch][state_mapped][action_mapped] += memory.rewards[-1][batch]
                self.nSAS[batch][state_mapped][action_mapped][next_state_mapped] += 1

                if self.nSA[batch][state_mapped][action_mapped] == self.m:

                    for i in range(self.m):

                        for s in range(self.meta_size * 2):
                        
                            for a in range(self.meta_size):

                                if self.nSA[batch][s][a] >= self.m:

                                    #We have already calculated the summation of rewards in line 28
                                    q = (self.R[batch][s][a]/self.nSA[batch][s][a])

                                    for next_s in range(env.d * 2):
                                        transition = self.nSAS[batch][s][a][next_s]/self.nSA[batch][s][a]
                                        q += (transition * torch.max(self.Q[batch,next_s,:]))

                                    self.Q[batch][s][a] = q 
                                    #We have already calculated the summation of rewards in line 28
\

        