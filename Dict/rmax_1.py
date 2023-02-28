import math
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.qvalues = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        
class RmaxAgent:
    def __init__(self, env, R_max, meta_gamma, inner_gamma, radius, epsilon, rmax_error):
        self.meta_gamma = meta_gamma
        self.inner_gamma = inner_gamma
        self.epsilon = epsilon        #epsilon for epsilon greedy
        self.rmax_error = rmax_error   #rmax_error for discretization error
        self.radius = radius               #added decimal place for Q & R matrix dimension
        
        self.m = int(math.ceil(math.log(1 / (self.rmax_error * (1-self.meta_gamma))) / (1-self.meta_gamma)))   #calculate m number
        self.Rmax = R_max * self.m        #max Rmax depends on number of visitation as well
       
        self.Q0 = torch.round(torch.Tensor([self.Rmax / (1 - self.meta_gamma)]), decimals=2).to(device)
        
        #no of possible combinations for an inner Q value
        self.poss_combo = math.ceil((1//(1-inner_gamma)) / radius) +1
        self.meta_size = self.poss_combo ** (env.d * env.num_actions)
        
        #Initialise dictionary with key names & null values
        self.Q = {"Qval": [self.Q0], "action": [0], "state":[0]}
        #self.R = {"Rval": [self.Rmax], "action": [0], "state":[0]},      try 0 instead
        self.R = {"Rval": [0], "action": [0], "state":[0]}
        self.nSA = {"nval": [0], "action": [0], "state":[0]}
        self.nSAS = {"nvals": [0], "next_state":[0], "action": [0], "state":[0]}
        
    def select_action(self, env, state):
        if np.random.random() < self.epsilon:
            action = torch.randint(self.meta_size)
        else:
            #find possible indices of given state
            #poss_indices = [i for i,x in enumerate(len(self.nSA.get("nval"))) if self.nSA.get("state") == self.find_meta_index(torch.flatten(state), self.radius, self.poss_combo)]
            
            #find action that corresponds to max. Q value
            #action = torch.argmax([self.Q["Qval"][i] for i in poss_indices])  
            #find possible indices of given state
            poss_val = [i for i in self.nSA.get("nval") if self.nSA.get("state") == self.find_meta_index(torch.flatten(state), self.radius, self.poss_combo)]
            
            rand_from_poss_max = random.choice(torch.argwhere(poss_val == torch.max(poss_val)).to(device)) 
            action = self.index_to_table(env, rand_from_poss_max.item(), 1) 
        return action     #returns action index
    
    def find_meta_index(self, meta):
        index = int(0) #initialise index

        #for every digit in meta-state/ meta-action:
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

    def index_to_table(self, env, index, agent_size):
        #returns a table of size [agent_size, num_states, num_actions], given index
        #agent_size either 1/2, 1 for action table, 2 for state table
        Q_size = agent_size * env.d * env.num_actions 
        reconstruct = torch.empty(Q_size).to(device)
        for i in reversed(range(Q_size)):
            if index >= self.poss_combo**i:
                q, mod = divmod(index, self.poss_combo**i)
                reconstruct[i] = q * self.radius     #recover original discretized value
                index = mod
            else:
                q = 0
                mod = 0
                reconstruct[i] = 0
        
        return torch.reshape(reconstruct, (agent_size, env.d, env.num_actions)).to(device)

    def update(self, env, memory, state, action, next_state):
        state_mapped = self.find_meta_index( torch.flatten(state))
        action_mapped = self.find_meta_index( torch.flatten(action))
        next_state_mapped = self.find_meta_index( torch.flatten(next_state))
        
        pair_index = self.find_pair_index([state_mapped, action_mapped], self.nSA)
        pairs_index = self.find_pair_index([state_mapped, action_mapped, next_state_mapped], self.nSAS)
        
        #if s-a pair hasn't been visited before (thus not in dictionary)
        if len(pair_index) == 0:
            #update nSA dictionary 
            self.nSA["state"].append(state_mapped)
            self.nSA["action"].append(action_mapped)
            self.nSA["nval"].append(1)
                
            #update R dictionary          
            self.R["state"].append(state_mapped)
            self.R["action"].append(action_mapped)
            self.R["Rval"].append(memory.rewards[-1])
            
            #update nSAS dictionary
            self.nSAS["state"].append(state_mapped)
            self.nSAS["action"].append(action_mapped)
            self.nSAS["next_state"].append(next_state_mapped)
            self.nSAS["nvals"].append(1)

            #update Q dicitonary
            self.Q["state"].append(state_mapped)
            self.Q["action"].append(action_mapped)
            self.Q["Qval"].append(self.Q0)
            
        #if s-a pair has been visited before

        #if visitation frequency < m (-1 since in this loop it reaches m)
        elif self.nSA["nval"][-1] < self.m:
            #update nSA dictionary 
            self.nSA["state"].append(state_mapped)
            self.nSA["action"].append(action_mapped)
            self.nSA["nval"].append(self.nSA["nval"][pair_index[-1]] + 1)
                
            #update R dictionary          
            self.R["state"].append(state_mapped)
            self.R["action"].append(action_mapped)
            self.R["Rval"].append(memory.rewards[-1] + self.meta_gamma * self.nSA["nval"][pair_index[-1]])
            
            #update nSAS dictionary
            self.nSAS["state"].append(state_mapped)
            self.nSAS["action"].append(action_mapped)
            self.nSAS["next_state"].append(next_state_mapped)
            if len(pairs_index) == 0:     #if s-a-s pair hasn't been visited before
                self.nSAS["nvals"].append(1)
            else:                         #else +=1
                self.nSAS["nvals"].append(self.nSAS["nvals"][pairs_index[-1]] + 1)
            
            #update Q dicitonary by optimistic value
            self.Q["state"].append(state_mapped)
            self.Q["action"].append(action_mapped)
            self.Q["Qval"].append(self.Q0)
            
        #else if visitation frequency reaches m 
        elif self.nSA["nval"][-1] >= self.m:
            #find s-a pair index
            m_pair_index = []
            for i,x in enumerate(self.nSA["nval"]):
                #if visited at least m times and nval is the max for entries that have the same state & action index
                if x >= self.m and i==max(self.find_pair_index([self.nSA["state"][i], self.nSA["action"][i]], self.nSA)):
                    #append the indexes
                    m_pair_index.append(i)
                    print(m_pair_index)

            for sas_ind in m_pair_index:
                #q = R/n + sum over next state(T * max_a(Q(s', a))                              
                q = (self.R["Rval"][sas_ind] / self.nSA["nval"][sas_ind])    #R/n first
                #calculate transition probability
                transition = self.nSAS["nvals"][sas_ind] / self.nSA["nval"][sas_ind]

                #find max Q value given next_state
                poss_list=[]
                #find list of indices that has next_s
                for j, y in enumerate(self.Q["Qval"]):
                    if self.Q["state"][j] == next_state_mapped:
                        poss_list.append(j)

                if len(poss_list) != 0:    #if next-state hasn't been visited before, we do nothing
                    #predict the transition of the next state

                    #find index & value of maxQ(next_s, a)
                    maxQ = max([z for k,z in enumerate(self.Q["Qval"]) if k in poss_list])
                    maxQ_index = [z for k,z in enumerate(self.Q["Qval"]) if k in poss_list].index(maxQ)

                    q += transition * maxQ.item()
                    #update Q dicitonary
                    Q_pair_index = self.find_pair_index([self.Q["state"][maxQ_index], self.Q["action"][maxQ_index]], self.nSA)  
                    self.Q["state"].append(self.Q["state"][maxQ_index])
                    self.Q["action"].append(self.Q["action"][maxQ_index])
                    self.Q["Qval"].append(q)  


