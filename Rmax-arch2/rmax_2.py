#Rmax code, cuda not yet incompatible
import math
import random
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
    def __init__(self, env, R_max, gamma, max_inner_epi, max_inner_steps, radius, epsilon, rmax_error):
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_inner_epi = max_inner_epi
        self.max_inner_steps = max_inner_steps
        self.radius = radius
        self.rmax_error = rmax_error
        
        #meta-s size = no of possible combinations for meta-s 
        self.meta_S_size = ((env.num_actions**env.num_agents) * (2**env.num_agents)) ** (self.max_inner_epi * self.max_inner_steps)  #2**2 for [0/1] reward for num_agents , same for [0/1] action value
        
        #meta-a size
        self.meta_A_size = (math.ceil((1// self.radius) + 1)) ** (env.d * env.num_actions)  
        
        #Q for meta-game, *2 for 2 player  
        self.Q = torch.ones(self.meta_S_size , self.meta_A_size).mul(R_max / (1 - self.gamma)).to(device) 
        self.R = torch.ones(self.meta_S_size , self.meta_A_size).to(device)
        self.nSA = torch.zeros(self.meta_S_size , self.meta_A_size).to(device)
        self.nSAS = torch.ones(self.meta_S_size , self.meta_A_size, self.meta_S_size).to(device)
        
        self.val1 = []
        self.val2 = []  #This is for keeping track of rewards over time and for plotting purposes  
        self.m = int(math.ceil(math.log(1 / (self.rmax_error * (1-self.gamma))) / (1-self.gamma)))   #calculate m number
        
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            action = env.action_space.sample()
        else:
            #find maximum action index, given state, makes sure if indices have same Q value, randomise
            lis = self.Q[state,:]
            action = random.choice(torch.argwhere(lis == torch.max(lis)).to(device)) 
            
        return action     #returns action index
    
    def find_meta_index(self, meta):
        index = int(0) #initialise index

        #for every digit in metastate/ metaaction:
        for i in range(list(meta.size())[0]):
            index += meta[i] * ((math.ceil((1// self.radius) + 1)) ** i)

        return int(index.item())

    def update(self, env, memory, state, action, next_state):
        action_mapped = self.find_meta_index( torch.flatten(action))
        state_mapped = self.find_meta_index( torch.flatten(state))
        next_state_mapped = self.find_meta_index( torch.flatten(next_state))

        if self.nSA[state_mapped][action_mapped] < self.m:

            self.nSA[state_mapped][action_mapped] += 1
            self.R[state_mapped][action_mapped] += memory.rewards[-1].item()
            self.nSAS[state_mapped][action_mapped][next_state_mapped] += 1

            if self.nSA[state_mapped][action_mapped] == self.m:

                for i in range(self.m):

                    for s in range(self.meta_S_size):

                        for a in range(self.meta_A_size):

                            if self.nSA[s][a] >= self.m:

                                #We have already calculated the summation of rewards in line 28
                                q = (self.R[s][a]/self.nSA[s][a])

                                for next_s in range(env.d * 2):
                                    transition = self.nSAS[s][a][next_s]/self.nSA[s][a]
                                    q += (transition * torch.max(self.Q[next_s,:]))

                                self.Q[s][a] = q 
