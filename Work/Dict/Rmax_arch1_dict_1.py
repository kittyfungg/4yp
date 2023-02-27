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


inner_gamma = 0.7         #inner game discount factor
meta_gamma = 0.7          #meta game discount factor
meta_alpha = 0.4          #meta game learning rate
R_max = 1
inner_steps = 20
meta_epi = 5000
meta_steps = 700

rmax_error = 0.6
epsilon = 0.3
radius = 3               #radius for discretization, assuming radius>1


# In[4]:


# reward tensor for plotting purposes [episode, step, agents]
plot_rew = torch.zeros(meta_epi, meta_steps, 2).to(device)    

# creating environment
env = MetaGames("PD")

# creating rmax agent
memory = Memory()
rmax = RmaxAgent(env, R_max, meta_gamma, inner_gamma, radius, epsilon, rmax_error)


# In[5]:


for episode in tqdm(range(meta_epi)): #for each meta-episode
    #initialise meta-state and meta-action randomly
    meta_s = rmax.index_to_table(env, random.randint(0, rmax.meta_size*2), 2)
    memory.states.append(meta_s) 
    
    for step in range(meta_steps):    #for each meta time step
        #reset environment (i.e. inner_action & inner_r)
        env.reset()  
        #previous meta-state[0] set as our policy of the next game
        env.innerq[0,:,:] = meta_s[0].detach().clone()
        #--------------------------------------START OF INNER GAME--------------------------------------
        inner_s = env.init_state
        reward_track = 0
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
            reward_track += reward
            
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
        
        #meta-action = inner game Q table of our agent
        meta_a = env.innerq[0, :, :].detach().clone()
        memory.actions.append(meta_a) 
        
        #Update inner Q table, oppo agent: by Q learning
        oppo_innerq_learnt = (1-meta_alpha) * env.innerq[1, :, :] + meta_alpha * oppo_action * info.detach().clone()
        oppo_phi = discretize(oppo_innerq_learnt , radius)
        
        #find index inside dictionary that corresponds to meta_s entry
        #array of indices that corresponds to meta_s
        stateind_arr = np.where(rmax.Q["state"] == rmax.find_meta_index(torch.flatten(meta_s)))[0]
        if len(stateind_arr) == 0:        #if we haven't visited that meta_s before & no record, 
            #we put random meta_action as inner Q 
            our_phi = rmax.index_to_table(env, random.randint(0, rmax.meta_size), 1)
        
        else:                             #else if we visited that meta_s before
            #inner Q is the action that corresponds to max Q(meta_s)
            maxQ_ind = np.argmax([x for i,x in enumerate(rmax.Q["Qval"]) if i in stateind_arr])
            our_phi = rmax.index_to_table(env, rmax.Q["action"][stateind_arr[maxQ_ind]], 1)
            
        #meta-state = discretized inner game Q table of all agents,     our_phi[0] since index_to_table outputs [1,5,2]
        next_meta_s = discretize(torch.stack((our_phi[0], oppo_phi),0), radius)
        
        #if not last step of an episode, save to memory
        if t != inner_steps -1 :
            memory.states.append(next_meta_s)    
        
        #meta-reward = inner reward of our agent at the last inner timestep
        our_REW = reward_track/inner_steps
        memory.rewards.append(our_REW)
        
        #rmax update step
        rmax.update(env, memory, meta_s, meta_a, next_meta_s)
    
        memory.qvalues.append(rmax.Q)
        
        #prepare meta_s for next step
        meta_s = next_meta_s.detach().clone()


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

# In[7]:


#create visitation dictionary to plot histogram of visitation counts
visit_dict = {}
sa_array = np.stack((np.array(rmax.nSA['state']), np.array(rmax.nSA['action'])), axis=1)
for i in range(len(rmax.nSA['nval'])):
    
    visit_dict[tuple(sa_array[i])]= rmax.nSA['nval'][i]

histogram_dict = Counter(visit_dict.values())
plt.bar(histogram_dict.keys(), histogram_dict.values(), 0.5, color='g')
plt.text(0,-meta_epi*meta_steps*0.1 ,"visitation counts: "+str(histogram_dict), fontsize=12)
figure0 = plt.gcf()
figure0.set_size_inches(10, 8)

#plt.savefig('histogram for' + str(datetime.now()) + '.png')


# In[9]:


plot_rew_mean= torch.mean(plot_rew[:,:,0],1)
fig_handle = plt.plot(plot_rew_mean.cpu().numpy())

plt.xlabel("episodes \n Average reward of our agent: " + str(round(torch.mean(plot_rew[:,:,0],(0,1)).detach().item(), 3)) + 
          "\n Average reward of another agent: " + str(round(torch.mean(plot_rew[:,:,1],(0,1)).detach().item(), 3)))

plt.ylabel("our agent's mean reward")

figure0 = plt.gcf() # get current figure
figure0.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(meta_epi) + '_' + str(meta_steps) + '_pd.png' , dpi = 100)


# In[10]:


plot_rew_epi_start = torch.mean(plot_rew[:int(meta_epi*0.1), :, 0], 0)
fig_handle = plt.plot(plot_rew_epi_start.cpu().numpy())

plt.xlabel("steps")

plt.ylabel("Average learning rate of first " + str(int(meta_epi*0.1)) + " episodes")

figure1 = plt.gcf() # get current figure
figure1.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(meta_epi) + '_' + str(meta_steps) + '_first_epi_mp1.png' , dpi = 100)


# In[11]:


plot_rew_epi_end = torch.mean(plot_rew[-int(meta_epi*0.1):, :, 0], 0)
fig_handle = plt.plot(plot_rew_epi_end.cpu().numpy())

plt.xlabel("steps")

plt.ylabel("Average learning rate of last " + str(int(meta_epi*0.1)) + " episodes")

figure2 = plt.gcf() # get current figure
figure2.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(meta_epi) + '_' + str(meta_steps)  + '_last_epi_mp1.png' , dpi = 100)


# # Interpreting results
# 

# In[28]:


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


# In[ ]:




