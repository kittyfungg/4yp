#Rmax code, cuda not yet incompatible
import math
import numpy as np
import torch
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
    def __init__(self, env, R_max, meta_gamma, inner_gamma, radius, epsilon = 0.2):
        self.meta_gamma = meta_gamma
        self.inner_gamma = inner_gamma
        self.epsilon = epsilon
        self.radius = radius               #added decimal place for Q & R matrix dimension
        #self.action_space = []
       
        #no of possible combinations for an inner Q value
        self.poss_combo = math.ceil((1//(1-inner_gamma)) / radius) +1
        self.meta_size = self.poss_combo ** (env.d * env.num_actions)
        
        #Q = [meta_state, meta_action], **2 for 2 player  
        self.Q = torch.ones(self.meta_size ** 2, self.meta_size).mul(R_max / (1 - self.meta_gamma)).to(device) 
        self.R = torch.ones(self.meta_size ** 2, self.meta_size).to(device)
        self.nSA = torch.zeros(self.meta_size ** 2, self.meta_size).to(device)
        self.nSAS = torch.ones(self.meta_size ** 2, self.meta_size, self.meta_size ** 2).to(device)
    
        self.val1 = []
        self.val2 = []  #This is for keeping track of rewards over time and for plotting purposes  
        self.m = int(math.ceil(math.log(1 / (self.epsilon * (1-self.meta_gamma))) / (1-self.meta_gamma)))   #calculate m number
        
    def select_action(self, env, state):
        if np.random.random() > (1-self.epsilon):
            action = torch.randint(self.meta_size)
        else:
            #find maximum action index, given state
            action = torch.argmax(self.Q[find_meta_index(torch.flatten(state), self.radius, self.poss_combo),:])    
        return action     #returns action index
    
    def find_meta_index(self, meta):
        index = int(0) #initialise index

        #for every digit in metastate/ metaaction:
        for i in range(list(meta.size())[0]):
            index += (meta[i]//self.radius) * (self.poss_combo ** i)
        return int(index)
    
    def index_to_table(self, env, index, agent_size) :
        #returns a table of size [agent, num_actions], given index
        maxi = agent_size * env.num_actions 
        reconstruct = torch.empty(maxi)
        for i in reversed(range(maxi)):
            if index >= self.poss_combo**i:
                q, mod = divmod(index, self.poss_combo**i)
                reconstruct[i] = q * self.radius     #recover original discretized value
                index = mod
            else:
                q = 0
                mod = 0
                reconstruct[i] = 0
        
        return torch.reshape(reconstruct, (agent_size, env.num_actions))

    def update(self, env, memory, state, action, next_state):
        action_mapped = self.find_meta_index( torch.flatten(action))
        state_mapped = self.find_meta_index( torch.flatten(state))
        next_state_mapped = self.find_meta_index( torch.flatten(next_state))

        if self.nSA[state_mapped][action_mapped] < self.m:

            self.nSA[state_mapped][action_mapped] += 1
            self.R[state_mapped][action_mapped] += memory.rewards[-1]
            self.nSAS[state_mapped][action_mapped][next_state_mapped] += 1

            if self.nSA[state_mapped][action_mapped] == self.m:

                for i in range(self.m):

                    for s in range(self.meta_size * 2):

                        for a in range(self.meta_size):

                            if self.nSA[s][a] >= self.m:

                                #We have already calculated the summation of rewards in line 28
                                q = (self.R[s][a]/self.nSA[s][a])

                                for next_s in range(env.d * 2):
                                    transition = self.nSAS[s][a][next_s]/self.nSA[s][a]
                                    q += (transition * torch.max(self.Q[next_s,:]))

                                self.Q[s][a] = q 
                                #We have already calculated the summation of rewards in line 28
\

