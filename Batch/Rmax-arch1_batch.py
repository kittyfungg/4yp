#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[6]:


import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from scipy.special import softmax
import random
import pickle
from datetime import datetime
from collections import Counter

from tqdm import tqdm

import gym
from gym.spaces import Discrete, Tuple

from env_mp_simple import MetaGamesSimplest
from rmax_1_batch import RmaxAgent, Memory

def discretize(number, radius):
    #[0,3,5,4,8] --> [0,3,6,3,9] for radius 3
    return np.round(np.divide(number, radius)) * radius

def Boltzmann(arr):
    #0.5 is just a temperature parameter, controls the spread of the softmax distribution
    action_value = np.zeros(arr.shape[0])
    prob = softmax(arr/0.4, 1)
    for b in range(arr.shape[0]):
        action_value[b] = np.random.choice(np.arange(arr.shape[1]), p=prob[b])
    return action_value


# In[7]:


bs = 1
inner_gamma = 0         #inner game discount factor, 0 since it's a one shot game
meta_gamma = 0.9         #meta game discount factor
meta_alpha = 0.4          #meta game learning rate
R_max = 1
rmax_error = 0.5
meta_epi = 50
meta_steps = 40

epsilon = 0
radius = 1              #radius for discretization, assuming radius>1


# In[8]:


#reward tensor for plotting purposes [bs, episode, step, agents]
plot_rew = np.zeros((bs, meta_epi, meta_steps, 2))

# creating environment
env = MetaGamesSimplest(bs)

# creating rmax agent
memory = Memory()
rmax = RmaxAgent(R_max, bs, meta_steps, meta_gamma, inner_gamma, radius, epsilon, rmax_error)


# In[4]:


#ourinnerq = np.zeros((bs, 2))
for episode in tqdm(range(meta_epi)): #for each meta-episode
    #reset environment 
    #initialise meta-state and meta-action randomly
    meta_s = env.reset()
    memory.states.append(meta_s)
   
    for step in range(meta_steps):    #for each meta time step
        #--------------------------------------START OF INNER GAME--------------------------------------  
        #select our inner-action with Boltzmann sampling, oppo inner-action with epsilon greedy 
        our_action = np.argmax(rmax.Q[np.arange(bs), rmax.find_meta_index(meta_s, "s").astype(int), :], axis=1)
        
        #print("inner actions: ", our_action, oppo_action)
        #run inner game according to actions
        obs, reward, done, _ = env.step(our_action) 

        #update inner r matrix [agent, action]
        our_innerr = np.transpose(np.stack([our_action, 1-our_action]) * reward)
        
        #---------------------------------------END OF INNER GAME--------------------------------------
        #save reward, info for plotting              
        plot_rew[:,episode,step,0] = reward
        plot_rew[:,episode,step,1] = 1-reward
        
        #Compute new inner Q table, our agent: meta_a that corresponds to max Q(meta_s)/ random, oppo: by Q learning
        #ourinnerq = rmax.select_action(meta_s, -1).squeeze()
        
        #print("inner-r: ", reward, "\n inner-q: ", env.innerq)

        #meta-action = action that corresponds to max Q(meta_s) = our inner Q
        meta_a = our_action
        memory.actions.append(meta_a) 

        #meta-state = discretized inner game Q table of all agents
        new_meta_s = obs
        memory.states.append(new_meta_s)    

        #meta-reward = sum of rewards of our agent in inner game of K episodes & T timesteps
        our_REW = reward               
        memory.rewards.append(our_REW)

        #rmax update step
        rmax.update(memory, meta_s, meta_a, new_meta_s)

        #print("updating s-a pair:", rmax.find_meta_index( torch.flatten(meta_s)), rmax.find_meta_index( torch.flatten(meta_a)),"\nrmax.R: ", rmax.R, "\nrmax.Q: ", rmax.Q, "\nrmax.nSA: ", rmax.nSA)
        #print(meta_s, meta_a, new_meta_s)

        #prepare meta_s for next step
        meta_s = new_meta_s


# In[12]:


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


# # Plots

# In[ ]:


idx02 = np.argwhere((self.nSA[np.arange(self.bs), state_mapped, action_mapped] < self.m) & (self.nSA[np.arange(self.bs), state_mapped, action_mapped] > 0))[0]


# In[9]:


self = rmax
action_mapped = self.find_meta_index(meta_a, "a").astype(int)
state_mapped = self.find_meta_index(meta_s, "s").astype(int)
next_state_mapped = self.find_meta_index(new_meta_s, "s").astype(int)
idx00 = np.argwhere(self.nSA[np.arange(self.bs), state_mapped, action_mapped] < self.m)[0]
idx01 = np.argwhere((self.nSA[np.arange(self.bs), state_mapped, action_mapped] < self.m) & (self.nSA[np.arange(self.bs), state_mapped, action_mapped] == 0))
#filter for 0<nSA<m
idx02 = np.argwhere((self.nSA[np.arange(self.bs), state_mapped, action_mapped] < self.m) & (self.nSA[np.arange(self.bs), state_mapped, action_mapped] > 0))


# In[10]:


idx00


# In[11]:


idx01


# In[12]:


idx02


# In[5]:


#generate histogram
visit_dict = {}
for i in range(len(rmax.nSA.flatten().tolist())):
    visit_dict[i]= rmax.nSA.flatten().tolist()[i]
    
histogram_dict = Counter(visit_dict.values())
plt.bar(histogram_dict.keys(), histogram_dict.values(), 0.5, color='g')
plt.xlabel("visitation counts: " + str(histogram_dict), fontsize=12)
figure0 = plt.gcf()
figure0.set_size_inches(10, 8)
plt.savefig('histogram at' + str(datetime.now()) + '.png')


# In[14]:


#generate reward mean
plot_rew_mean = torch.mean(plot_rew[:,:,0],1)
fig_handle = plt.plot(plot_rew_mean.cpu().numpy())

plt.xlabel("episodes \n Average reward of our agent: " + str(round(torch.mean(plot_rew[:,:,0],(0,1)).detach().item(), 3)) + 
          "\n Average reward of another agent: " + str(round(torch.mean(plot_rew[:,:,1],(0,1)).detach().item(), 3)))

plt.ylabel("Mean rewards")

figure2 = plt.gcf() # get current figure
figure2.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(meta_epi) + '_' + str(meta_steps) + '_mp1.png'  , dpi = 100)
plt.clf()


# In[15]:


#generate learning curve at start
plot_rew_epi_start = torch.mean(plot_rew[:int(meta_epi*0.1), :, 0], 0)
fig_handle = plt.plot(plot_rew_epi_start.cpu().numpy())

plt.xlabel("steps")

plt.ylabel("Average learning rate of first " + str(int(meta_epi*0.1)) + " episodes")

figure3 = plt.gcf() # get current figure
figure3.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(meta_epi) + '_' + str(meta_steps) + '_first_epi_mp1.png' , dpi = 100)
plt.clf()


# In[16]:


#generate learning curve at end
plot_rew_epi_end = torch.mean(plot_rew[-int(meta_epi*0.1):, :, 0], 0)
fig_handle = plt.plot(plot_rew_epi_end.cpu().numpy())

plt.xlabel("steps")

plt.ylabel("Average learning rate of last " + str(int(meta_epi*0.1)) + " episodes")

figure4 = plt.gcf() # get current figure
figure4.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(meta_epi) + '_' + str(meta_steps) + '_last_epi_mp1.png' , dpi = 100)
plt.clf()


# # Interpreting results 

# In[ ]:



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

