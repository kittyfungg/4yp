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
    def __init__(self, env, R_max, gamma, max_inner_epi, max_inner_steps, radius, epsilon = 0.2):
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_inner_epi = max_inner_epi
        self.max_inner_steps = max_inner_steps
    
        #no of possible combinations for meta-s 
        self.poss_combo_s = (env.d -1) * env.num_agents * (env.num_actions * 4)    #4 for number of rewards
        self.meta_S_size = self.poss_combo_s ** (self.max_inner_epi * self.max_inner_steps)
        
        #no of possible combinations for meta-a 
        self.meta_A_size = env.d ** env.num_actions     #5^2
        
        #Q for meta-game, *2 for 2 player  
        self.Q = torch.ones(self.meta_S_size , self.meta_A_size).mul(R_max / (1 - self.gamma)).to(device) 
        self.R = torch.ones(self.meta_S_size , self.meta_A_size).to(device)
        self.nSA = torch.zeros(self.meta_S_size , self.meta_A_size).to(device)
        self.nSAS = torch.ones(self.meta_S_size , self.meta_A_size, self.meta_S_size).to(device)
        
        self.val1 = []
        self.val2 = []  #This is for keeping track of rewards over time and for plotting purposes  
        self.m = int(math.ceil(math.log(1 / (self.epsilon * (1-self.gamma))) / (1-self.gamma)))   #calculate m number
        
    def select_action(self, state):
        if np.random.random() > (1-self.epsilon):
            action = env.action_space.sample()
        else:
            action = torch.amax(self.Q[:,state,:], 2)
        return action     #returns action of length b
    
    def find_meta_index(meta):
        index = int(0) #initialise index

        #for every digit in metastate/ metaaction:
        for i in range(list(meta.size())[0]):
            index += meta[i] * ((math.ceil((1/(1-self.gamma)) / self.radius) + 1 )**i)

        return int(index.item())

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
