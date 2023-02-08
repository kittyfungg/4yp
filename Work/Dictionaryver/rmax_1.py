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
        self.Rmax = R_max
        self.Q0 = round(self.Rmax / (1 - self.meta_gamma), 2)
        
        #no of possible combinations for an inner Q value
        self.poss_combo = math.ceil((1//(1-inner_gamma)) / radius) +1
        self.meta_size = self.poss_combo ** (env.d * env.num_actions)
        
        #Initialise dictionary with key names & null values
        self.Q = {"Qval": [self.Q0], "action": [0], "state":[0]}
        self.R = {"Rval": [self.Rmax], "action": [0], "state":[0]}
        self.nSA = {"nval": [0], "action": [0], "state":[0]}
        self.nSAS = {"nvals": [0], "next_state":[0], "action": [0], "state":[0]}
    
        self.m = int(math.ceil(math.log(1 / (self.epsilon * (1-self.meta_gamma))) / (1-self.meta_gamma)))   #calculate m number
        
    def select_action(self, env, state):
        if np.random.random() > (1-self.epsilon):
            action = torch.randint(self.meta_size)
        else:
            #find possible indices of given state
            poss_indices = [i for i,x in enumerate(len(self.nSA.get("nval"))) if self.nSA.get("state") == self.find_meta_index(torch.flatten(state), self.radius, self.poss_combo)]
            
            #find action that corresponds to max. Q value
            action = torch.argmax([self.Q["Qval"][i] for i in poss_indices])     
        return action     #returns action index
    
    def find_meta_index(self, meta):
        index = int(0) #initialise index

        #for every digit in metastate/ metaaction:
        for i in range(list(meta.size())[0]):
            index += (meta[i]//self.radius) * (self.poss_combo ** i)
        return int(index)
    
    def find_pair_index(self, pair, dictionary):
        #returns the index of [state,action]/ [state,action,next_state] pair in the dictionary
        if len(pair)==2:      #if [state,action] pair
            state_arr = dictionary["state"]
            action_arr = dictionary["action"]
            state_locations = [i for i,x in enumerate(state_arr) if x==pair[0]]
            action_locatons = [i for i,x in enumerate(action_arr) if x==pair[1]]
            return list(set(state_locations) & set(action_locatons))
        
        else:                 #if [state,action,next_state] pair
            state_arr = dictionary["state"]
            action_arr = dictionary["action"]
            next_state_arr = dictionary["next_state"]
            state_locations = [i for i,x in enumerate(state_arr) if x==pair[0]]
            action_locatons = [i for i,x in enumerate(action_arr) if x==pair[1]]
            next_state_locations = [i for i,x in enumerate(state_arr) if x==pair[2]]
            return list(set(state_locations) & set(action_locatons) & set(next_state_locations))

    def index_to_table(self, env, index, agent_size) :
        #returns a table of size [agent, num_actions], given index
        #agent_size either 1/2, 1 for action table, 2 for state table
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
        
        return torch.reshape(reconstruct, (agent_size, env.num_actions)).to(device)

    def update(self, env, memory, state, action, next_state):
        state_mapped = self.find_meta_index( torch.flatten(state))
        action_mapped = self.find_meta_index( torch.flatten(action))
        next_state_mapped = self.find_meta_index( torch.flatten(next_state))
        
        pair_index = self.find_pair_index([state_mapped, action_mapped], self.nSA)
        pairs_index = self.find_pair_index([state_mapped, action_mapped, next_state_mapped], self.nSAS)
        
        #if s-a pair hasn't been visited before (thus not in dictionary)
        if len(pair_index) <= 1:
            #update nSA dictionary 
            self.nSA["state"].append(state_mapped)
            self.nSA["action"].append(action_mapped)
            self.nSA["nval"].append(1)
                
            #update R dictionary          
            self.R["state"].append(state_mapped)
            self.R["action"].append(action_mapped)
            self.R["Rval"].append(1)
            
            #update nSAS dictionary
            self.nSAS["state"].append(state_mapped)
            self.nSAS["action"].append(action_mapped)
            self.nSAS["next_state"].append(next_state_mapped)
            self.nSAS["nvals"].append(1)

            #update Q dicitonary
            self.Q["state"].append(state_mapped)
            self.Q["action"].append(action_mapped)
            self.Q["Qval"].append(self.Q0)
        
        #else if visitation frequency < m number 
        elif self.nSA["nval"][pair_index[-1]] < self.m:
            #update nSA dictionary 
            self.nSA["state"].append(state_mapped)
            self.nSA["action"].append(action_mapped)
            self.nSA["nval"].append(self.nSA["nval"][pair_index[-1]] + 1)
                
            #update R dictionary          
            self.R["state"].append(state_mapped)
            self.R["action"].append(action_mapped)
            self.R["Rval"].append(memory.rewards[-1])
            
            #update nSAS dictionary
            self.nSAS["state"].append(state_mapped)
            self.nSAS["action"].append(action_mapped)
            self.nSAS["next_state"].append(next_state_mapped)
            if len(pairs_index) == 0:     #if s-a-s pair hasn't been visited before
                self.nSAS["nvals"].append(1)
            else:                         #else +=1
                self.nSAS["nvals"].append(self.nSAS["nvals"][pairs_index[-1]] + 1)

            if self.nSA["nval"][pair_index[-1]] == self.m:
                print("hit m")

                for mval in range(self.m):

                    for s in range(self.meta_size * 2):

                        for a in range(self.meta_size):
                            #find indices that have been visited for at least m times
                            bigger_m_ind = [i for i,x in enumerate(len(self.nSA["nval"])) if x >= self.m]
                            
                            for ind in bigger_m_ind:   
                                q = (self.R["Rval"][ind] / self.nSA["nval"][ind])

                                for next_s in range(env.d * 2):
                                    transition = self.nSAS["nvals"][ind] / self.nSA["nval"][ind]
                                    #find state indices that corresponds to next_s
                                    stateind_arr = [i for i,x in enumerate(self.nSA["states"]) if x == next_s]
                                    
                                    #if there are Q values that are larger than R_max / (1 - meta_gamma)
                                    if np.max([self.Q["Qval"][states] for states in range(stateind_arr)]) > self.Q0:
                                        q += transition * np.max([self.Q["Qval"][states] for states in range(stateind_arr)])
                                    #else we use the explore Q value
                                    else:
                                        q += self.Q0
                                        
                                #update Q dicitonary
                                Q_pair_index = self.find_pair_index([s, a], self.nSA)  
                                self.Q["state"].append(s)
                                self.Q["action"].append(a)
                                self.Q["Qval"][Q_pair_index] = q 
                                

