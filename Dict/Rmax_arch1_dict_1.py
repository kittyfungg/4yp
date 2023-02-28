#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import random
import pickle
from datetime import datetime
from collections import Counter

from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import gym
from gym.spaces import Discrete, Tuple

from env import MetaGames
from rmax_1 import RmaxAgent, Memory

def discretize(number, radius):
     #return (torch.round(torch.div(number, radius))) * radius
    return torch.round(torch.div(number, radius)) * radius


# In[2]:


inner_gamma = 0.5         #inner game discount factor
meta_gamma = 0.5          #meta game discount factor
meta_alpha = 0.4          #meta game learning rate
R_max = 1
inner_steps = 15
meta_epi = 1500
meta_steps = 500

rmax_error = 0.2
epsilon = 0.2
radius = 3               #radius for discretization, assuming radius>1


# In[3]:


# reward tensor for plotting purposes [episode, step, agents]
plot_rew = torch.zeros(meta_epi, meta_steps, 2).to(device)    

# creating environment
env = MetaGames("PD")

# creating rmax agent
memory = Memory()
rmax = RmaxAgent(env, R_max, meta_gamma, inner_gamma, radius, epsilon, rmax_error)


# In[8]:


for episode in tqdm(range(meta_epi)): #for each meta-episode
    #initialise meta-state and meta-action randomly
    meta_s = rmax.index_to_table(env, random.randint(0, rmax.meta_size**2), 2)
    memory.states.append(meta_s) 
    
    for step in range(meta_steps):    #for each meta time step
        #reset environment (i.e. inner_action & inner_r)
        env.reset()  
        #previous meta-state[0] set as our policy of the next game
        env.innerq[0,:,:] = meta_s[0].detach().clone()
        #--------------------------------------START OF INNER GAME--------------------------------------
        inner_s = env.init_state
        accum_reward = 0
        for t in range(inner_steps):
            #select oppo inner-action with epsilon greedy 
            oppo_action = env.select_action(env.state_mapping(inner_s, "oppo")).detach().clone()    
            #select our inner-action 
            #in case for 1 state, the q val of 2 actions are the same
            maxind = torch.where(env.innerq[0, env.state_mapping(inner_s, "our"),:] == torch.max(env.innerq[0, env.state_mapping(inner_s, "our"),:]))
            if maxind[0].size()[0] == 2:
                index = torch.randint(0, 2, (1,))
            else:
                index = torch.argmax(env.innerq[0, env.state_mapping(inner_s, "our"),:])
            our_action = env.action_unmapping(index)

            #run inner game according to best_action
            state, reward, info = env.step(our_action, oppo_action)  
            inner_s = [state[0].tolist(), state[1].tolist()]
            accum_reward += reward
            
            if t == 0:   #for init_state, update row 0
                #update inner r matrix [agent, state, action], state = 1/2/3/4
                env.innerr[0, 0, env.action_mapping(our_action)] = reward.detach().clone() 
                env.innerr[1, 0, env.action_mapping(oppo_action)] = info.detach().clone()
                #update opponent's inner q matrix
                env.innerq[1, 0, env.action_mapping(oppo_action)] = info.detach().clone() + inner_gamma * torch.max(env.innerq[1,env.state_mapping(inner_s, "oppo"),:])

            else:
                #update inner r matrix [agent, state, action], state = 1/2/3/4
                env.innerr[0, env.state_mapping(inner_s, "our"), env.action_mapping(our_action)] = reward.detach().clone() 
                env.innerr[1, env.state_mapping(inner_s, "oppo"), env.action_mapping(oppo_action)] = info.detach().clone()
                #update opponent's inner q matrix
                env.innerq[1, env.state_mapping(inner_s, "oppo"), env.action_mapping(oppo_action)] = info.detach().clone() + inner_gamma * torch.max(env.innerq[1,env.state_mapping(inner_s, "oppo"),:])
            
        #---------------------------------------END OF INNER GAME--------------------------------------
        #save reward, info for plotting              
        plot_rew[episode,step,0] = reward.detach().clone()
        plot_rew[episode,step,1] = info.detach().clone()

        #Update inner Q table, oppo agent: by Q learning
        oppo_innerq_learnt = (1-meta_alpha) * env.innerq[1, :, :] + meta_alpha * oppo_action * info.detach().clone()
        oppo_phi = discretize(oppo_innerq_learnt , radius)
        
        #find index inside dictionary that corresponds to meta_s entry
        #array of indices that corresponds to meta_s
        stateind_arr = [i for i in range(len(rmax.Q["state"])) if rmax.find_meta_index(torch.flatten(meta_s))==rmax.Q["state"][i]]
        if len(stateind_arr) == 0:        #if we haven't visited that meta_s before & no record, 
            #we put random meta_action as inner Q 
            our_phi = rmax.index_to_table(env, random.randint(0, rmax.meta_size), 1)
        
        else:                             #else if we visited that meta_s before
            #inner Q is the action that corresponds to max Q(meta_s) 
            maxQ_ind = random.choice([x for x in stateind_arr if rmax.Q["Qval"][x]==max(rmax.Q["Qval"])])
            our_phi = rmax.index_to_table(env, rmax.Q["action"][maxQ_ind], 1)
        
        #meta-action = action that corresponds to max Q(meta_s) = our inner Q
        meta_a = our_phi
        memory.actions.append(meta_a) 
        
        #meta-state = discretized inner game Q table of all agents,     our_phi[0] since index_to_table outputs [1,5,2]
        next_meta_s = discretize(torch.stack((our_phi[0], oppo_phi),0), radius)
        
        #if not last step of an episode, save to memory
        if t != inner_steps -1 :
            memory.states.append(next_meta_s)    
        
        #meta-reward = inner reward of our agent at the last inner timestep
        our_REW = accum_reward/inner_steps
        memory.rewards.append(our_REW)
        
        #rmax update step
        rmax.update(env, memory, meta_s, meta_a, next_meta_s)
    
        memory.qvalues.append(rmax.Q)
        
        #prepare meta_s for next step
        meta_s = next_meta_s.detach().clone()


# In[9]:


self=rmax
state = meta_s
action = meta_a
next_state= next_meta_s
state_mapped = self.find_meta_index( torch.flatten(state))
action_mapped = self.find_meta_index( torch.flatten(action))
next_state_mapped = self.find_meta_index( torch.flatten(next_state))

pair_index = self.find_pair_index([state_mapped, action_mapped], self.nSA)
pairs_index = self.find_pair_index([state_mapped, action_mapped, next_state_mapped], self.nSAS)
#find s-a pair index
m_pair_index = []
for i,x in enumerate(self.nSA["nval"]):
    #if visited at least m times and nval is the max for entries that have the same state & action index
    if x >= self.m and i==max(self.find_pair_index([self.nSA["state"][i], self.nSA["action"][i]], self.nSA)):
        #append the indexes
        m_pair_index.append(i)
        print(m_pair_index)


# In[ ]:


transition


# In[10]:


for sas_ind in m_pair_index:
    print(sas_ind)
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

        q += transition * maxQ
        #update Q dicitonary
        Q_pair_index = self.find_pair_index([self.Q["state"][maxQ_index], self.Q["action"][maxQ_index]], self.nSA)  
        self.Q["state"].append(self.Q["state"][maxQ_index])
        self.Q["action"].append(self.Q["action"][maxQ_index])
        self.Q["Qval"].append(q)  



# In[16]:


# Open a file and use dump()
with open('plot_rew' + str(datetime.now()) + '.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(plot_rew, file)
    
# Open a file and use dump()
with open('memory' + str(datetime.now()) + '.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(memory, file)

# Open a file and use dump()
with open('rmax' + str(datetime.now()) + '.pkl', 'wb') as file:  
    # A new file will be created
    pickle.dump(rmax, file)


# 
# # Plots

# In[9]:


poss_combo = math.ceil((1//(1-inner_gamma)) / radius) +1
meta_size = poss_combo ** (env.d * env.num_actions)
print(meta_size)


# In[7]:


#create visitation dictionary to plot histogram of visitation counts
visit_dict = {}
sa_array = np.stack((np.array(rmax.nSA['state']), np.array(rmax.nSA['action'])), axis=1)
for i in range(len(rmax.nSA['nval'])):
    
    visit_dict[tuple(sa_array[i])]= rmax.nSA['nval'][i]

histogram_dict = Counter(visit_dict.values())
plt.bar(histogram_dict.keys(), histogram_dict.values(), 0.5, color='g')
plt.text(0,-meta_epi*meta_steps*0.01 ,"visitation counts: "+str(histogram_dict), fontsize=12)
figure0 = plt.gcf()
figure0.set_size_inches(10, 8)

plt.savefig('histogram for' + str(datetime.now()) + '.png')


# In[12]:


plot_rew_mean= torch.mean(plot_rew[:,:,0],1)
fig_handle = plt.plot(plot_rew_mean.cpu().numpy())

plt.xlabel("episodes \n Average reward of our agent: " + str(round(torch.mean(plot_rew[:,:,0],(0,1)).detach().item(), 3)) + 
          "\n Average reward of another agent: " + str(round(torch.mean(plot_rew[:,:,1],(0,1)).detach().item(), 3)))

plt.ylabel("our agent's mean reward")

figure0 = plt.gcf() # get current figure
figure0.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(meta_epi) + '_' + str(meta_steps) + '_pd.png' , dpi = 100)


# In[13]:


plot_rew_epi_start = torch.mean(plot_rew[:int(meta_epi*0.1), :, 0], 0)
fig_handle = plt.plot(plot_rew_epi_start.cpu().numpy())

plt.xlabel("steps")

plt.ylabel("Average learning rate of first " + str(int(meta_epi*0.1)) + " episodes")

figure1 = plt.gcf() # get current figure
figure1.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(meta_epi) + '_' + str(meta_steps) + '_first_epi_mp1.png' , dpi = 100)


# In[14]:


plot_rew_epi_end = torch.mean(plot_rew[-int(meta_epi*0.1):, :, 0], 0)
fig_handle = plt.plot(plot_rew_epi_end.cpu().numpy())

plt.xlabel("steps")

plt.ylabel("Average learning rate of last " + str(int(meta_epi*0.1)) + " episodes")

figure2 = plt.gcf() # get current figure
figure2.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(meta_epi) + '_' + str(meta_steps)  + '_last_epi_mp1.png' , dpi = 100)


# In[2]:


import glob
path1 = "memory*.pkl"
path2 = "plot_rew*.pkl"
path3 = "rmax*.pkl"
for filename in glob.glob(path1):
    with open(filename, 'rb') as f:
        memory = pickle.load(f)
        
for filename in glob.glob(path2):    
    with open(filename, 'rb') as g:
        plot_rew = pickle.load(g)
        
for filename in glob.glob(path3):    
    with open(filename, 'rb') as g:
        rmax = pickle.load(g)        

