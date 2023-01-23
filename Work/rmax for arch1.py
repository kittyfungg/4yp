#Rmax code, cuda not yet incompatible
import math
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def find_meta_index(meta, radius, gamma):
    index = int(0) #initialise index

    #for every digit in metastate/ metaaction:
    for i in range(list(meta.size())[0]):
        index += meta[i] * ((math.ceil((1/(1-gamma)) / radius) + 1 )**i)
        
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
    def __init__(self, env, R_max, gamma, max_episodes, max_steps, radius, epsilon = 0.2):
        self.gamma = gamma
        self.epsilon = epsilon
        self.radius = radius               #added decimal place for Q & R matrix dimension
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.b = env.b
        #no of possible combinations for a inner Q value
        self.poss_combo = math.ceil((1/(1-gamma)) / radius) + 1
        self.meta_size = self.poss_combo ** (env.d * env.num_actions)
        #Q for meta-game, *2 for 2 player  
        self.Q = torch.ones(self.b, self.meta_size * 2, self.meta_size).mul(R_max / (1 - self.gamma)).to(device) 
        self.R = torch.ones(self.b, self.meta_size * 2, self.meta_size).to(device)
        self.nSA = torch.zeros(self.b, self.meta_size * 2, self.meta_size).to(device)
        self.nSAS = torch.ones(self.b, self.meta_size * 2, self.meta_size, self.meta_size * 2).to(device)
        
        self.val1 = []
        self.val2 = []  #This is for keeping track of rewards over time and for plotting purposes  
        self.m = int(math.ceil(math.log(1 / (self.epsilon * (1-self.gamma))) / (1-self.gamma)))   #calculate m number
        
    def select_action(self, state):
        if np.random.random() > (1-self.epsilon):
            action = env.action_space.sample()
        else:
            action = torch.amax(self.Q[:,state,:], 2)
        return action     #returns action of length b
    
    def update(self, memory, state, action, next_state):
        #for each batch
        for i in range(self.b):
            action_mapped = find_meta_index(torch.flatten(action[i]), self.radius, self.gamma)
            state_mapped = find_meta_index(torch.flatten(state[i]), self.radius, self.gamma)
            next_state_mapped = find_meta_index(torch.flatten(next_state[i]), self.radius, self.gamma)

            if self.nSA[i][state_mapped][action_mapped] < self.m:
                
                self.nSA[i][state_mapped][action_mapped] += 1
                self.R[i][state_mapped][action_mapped] += memory.rewards[-1][i]
                self.nSAS[i][state_mapped][action_mapped][next_state_mapped] += 1

                if self.nSA[i][state_mapped][action_mapped] == self.m:

                    for i in range(mnumber):

                        for s in range(self.poss_combo * 2):

                            for a in range(self.poss_combo):

                                if self.nSA[i][s][a] >= self.m:

                                    #We have already calculated the summation of rewards in line 28
                                    q = (self.R[i][s][a]/self.nSA[i][s][a])

                                    for next_state in range(env.d * 2):
                                        transition = self.nSAS[i][s][a][next_state]/self.nSA[i][s][a]
                                        q += (transition * torch.max(self.Q[i,next_state,:]))

                                    self.Q[i][s][a] = q 
                                    #We have already calculated the summation of rewards in line 28
        