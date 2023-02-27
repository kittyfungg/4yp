#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import random
import pickle

from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import gym
from gym.spaces import Discrete, Tuple

from env_mp_1 import MetaGames
from rmax_1 import RmaxAgent, Memory

def discretize(number, radius):
    #return (torch.round(torch.div(number, radius))) * radius
    #change made: originally: [0,3,6,9], now: [0,1,2,3]
    return torch.round(torch.div(number, radius))


# In[3]:


inner_gamma = 0.7         #inner game discount factor
meta_gamma = 0.7          #meta game discount factor
meta_alpha = 0.4          #meta game learning rate
R_max = 1
max_meta_epi = 1500
max_meta_steps = 300

epsilon = 0.2
radius = 1                #radius for discretization, assuming radius>1


# In[2]:


#reward tensor for plotting purposes [episode, step, agents]
plot_rew = torch.zeros(max_meta_epi, max_meta_steps, 2).to(device)    

# creating environment
env = MetaGames("NL", "IPD")

# creating rmax agent
memory = Memory()
rmax = RmaxAgent(env, R_max, meta_gamma, inner_gamma, radius, epsilon)


# In[ ]:


#initialise meta-state and meta-action randomly
meta_s = rmax.index_to_table(env, random.randint(0, rmax.meta_size * env.num_agents), env.num_agents)
memory.states.append(meta_s)
for episode in tqdm(range(max_meta_epi)): #for each meta-episode
    #reset environment 
    env.reset() 
    for step in range(max_meta_steps):    #for each meta time step
        #previous meta-state set as the policy of the next game
        env.innerq[0,:] = meta_s[0].detach().clone() #
        #--------------------------------------START OF INNER GAME--------------------------------------  
        #select inner-action with epsilon greedy 
        oppo_action = env.select_action().detach().clone()      
        our_action = torch.argmax(env.innerq[0,:]).unsqueeze(0).to(device) 
        
        #run inner game according to actions
        reward, info = env.step(torch.cat((our_action, oppo_action))) 

        #update inner r matrix [agent, action]
        env.innerr[0, int(our_action)] = reward.detach().clone() 
        env.innerr[1, int(oppo_action)] = info.detach().clone()
        #---------------------------------------END OF INNER GAME--------------------------------------
        #save reward, info for plotting              
        plot_rew[episode,step,0] = reward.detach().clone()
        plot_rew[episode,step,1] = info.detach().clone()
        
        #meta-action = inner game Q table of our agent
        meta_a = env.innerq[0, :].detach().clone()
        memory.actions.append(meta_a) 
        
        #Compute new inner Q table, our agent: meta_a that corresponds to max Q(meta_s)/ random, oppo: by Q learning
        env.innerq[0, :] = rmax.select_action(env, meta_s[0])
        env.innerq[1, :] = (1-meta_alpha) * env.innerq[1, :] + meta_alpha * info
        
        #meta-state = discretized inner game Q table of all agents
        new_meta_s = discretize(env.innerq.detach().clone(), radius)
        memory.states.append(new_meta_s)    
        
        #meta-reward = sum of rewards of our agent in inner game of K episodes & T timesteps
        our_REW = reward.detach().clone()                
        memory.rewards.append(our_REW)
        
        #rmax update step
        rmax.update(env, memory, meta_s, meta_a, new_meta_s)
        
        #prepare meta_s for next step
        meta_s = new_meta_s.detach().clone()


# In[13]:


# Open a file and use dump()
with open('plot_rew.pkl', 'wb') as file:
      
    # A new file will be created
    pickle.dump(plot_rew, file)
    
# Open a file and use dump()
with open('memory.pkl', 'wb') as file:
      
    # A new file will be created
    pickle.dump(memory, file)


# # Plots

# In[2]:


from collections import Counter
from datetime import datetime

visit_dict = {}
array = [ int(x) for x in torch.flatten(rmax.nSA.cpu()).tolist()]
histo = Counter(array)

plt.bar(histo.keys(), histo.values(), 0.5, color='g')
plt.text(0, -200, "visitation counts: "+ str(histo), fontsize=12)
figure0 = plt.gcf()
figure0.set_size_inches(10, 8)

plt.savefig('histogram for' + str(datetime.now()) + '.png')


# In[17]:


plt.imshow(rmax.nSA.cpu().numpy(), cmap='hot', interpolation='nearest')
figure0 = plt.gcf()
figure0.set_size_inches(10, 8)
plt.savefig('heatmap for' + str(datetime.now()) + '.png')


# In[8]:


plot_rew_mean_diff = torch.mean(plot_rew[:,:,0],1)# - torch.mean(plot_rew[:,:,1],1)
fig_handle = plt.plot(plot_rew_mean_diff.cpu().numpy())

plt.xlabel("episodes \n Average reward of our agent: " + str(round(torch.mean(plot_rew[:,:,0],(0,1)).detach().item(), 3)) + 
          "\n Average reward of another agent: " + str(round(torch.mean(plot_rew[:,:,1],(0,1)).detach().item(), 3)))

plt.ylabel("difference in mean rewards")

figure = plt.gcf() # get current figure
figure.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(max_meta_epi) + '_' + str(max_meta_steps) + '_mp1.png' , dpi = 100)
plt.clf()


# In[24]:


plot_rew_epi_start = torch.mean(plot_rew[:int(max_meta_epi*0.1), :, 0], 0)
fig_handle = plt.plot(plot_rew_epi_start.cpu().numpy())

plt.xlabel("steps")

plt.ylabel("Average learning rate of first " + str(int(max_meta_epi*0.1)) + " episodes")

figure = plt.gcf() # get current figure
figure.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(max_meta_epi) + '_' + str(max_meta_steps) + 'first_epi_mp1.png' , dpi = 100)
plt.clf()


# In[25]:


plot_rew_epi_end = torch.mean(plot_rew[-int(max_meta_epi*0.1):, :, 0], 0)
fig_handle = plt.plot(plot_rew_epi_end.cpu().numpy())

plt.xlabel("steps")

plt.ylabel("Average learning rate of last " + str(int(max_meta_epi*0.1)) + " episodes")

figure = plt.gcf() # get current figure
figure.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(max_meta_epi) + '_' + str(max_meta_steps) + '_last_epi_mp1.png' , dpi = 100)
plt.clf()


# # Interpreting results 

# In[29]:


with open('memory.pkl', 'rb') as f:
    memory_loaded = pickle.load(f)
    
with open('plot_rew.pkl', 'rb') as g:
    plot_rew_loaded = pickle.load(g)


# In[32]:


plot_rew_epi_start = torch.mean(plot_rew_loaded[:int(max_meta_epi*0.1), :, 0], 0)
fig_handle = plt.plot(plot_rew_epi_start.cpu().numpy())

plt.xlabel("steps")

plt.ylabel("Average learning rate of first " + str(int(max_meta_epi*0.1)) + " episodes")

figure = plt.gcf() # get current figure
figure.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(max_meta_epi) + '_' + str(max_meta_steps) + 'first_epi_mp1.png' , dpi = 100)


# In[33]:


plot_rew_epi_end = torch.mean(plot_rew_loaded[-int(max_meta_epi*0.1):, :, 0], 0)
fig_handle = plt.plot(plot_rew_epi_end.cpu().numpy())

plt.xlabel("steps")

plt.ylabel("Average learning rate of last " + str(int(max_meta_epi*0.1)) + " episodes")

figure = plt.gcf() # get current figure
figure.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(max_meta_epi) + '_' + str(max_meta_steps) + '_last_epi_mp1.png' , dpi = 100)

