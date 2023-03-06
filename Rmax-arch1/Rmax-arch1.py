#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[25]:


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

from env_mp_1 import MetaGames
from rmax_1 import RmaxAgent, Memory

def discretize(number, radius):
    #[0,3,5,4,8] --> [0,3,6,3,9] for radius 3
    return torch.round(torch.div(number, radius)) * radius

def Boltzmann(tensor):
    #0.5 is just a temperature parameter, controls the spread of the softmax distribution
    prob = torch.softmax(env.innerq[0,:].cpu()/0.4, 0).numpy()
    action_value = np.random.choice(np.arange(tensor.size()[0]), p=prob)
    return torch.Tensor([action_value]).to(device)


# In[26]:


inner_gamma = 0         #inner game discount factor, 0 since it's a one shot game
meta_gamma = 0.5          #meta game discount factor
meta_alpha = 0.4          #meta game learning rate
R_max = 1
rmax_error = 0.2
meta_epi = 500
meta_steps = 500

epsilon = 0.2
radius = 0.5                #radius for discretization, assuming radius>1


# In[27]:


#reward tensor for plotting purposes [episode, step, agents]
plot_rew = torch.zeros(meta_epi, meta_steps, 2).to(device)    

# creating environment
env = MetaGames("NL", "IPD")

# creating rmax agent
memory = Memory()
rmax = RmaxAgent(env, R_max, meta_gamma, inner_gamma, radius, epsilon, rmax_error)


# In[ ]:


#initialise meta-state and meta-action randomly
meta_s = rmax.index_to_table(env, random.randint(0, rmax.meta_size ** env.num_agents), env.num_agents)
memory.states.append(meta_s)
for episode in tqdm(range(meta_epi)): #for each meta-episode
    #reset environment 
    env.reset() 
    print(rmax.nSA)
    for step in range(meta_steps):    #for each meta time step
        #previous meta-state set as the policy of the next game
        env.innerq[0,:] = meta_s[0].detach().clone() 
        #--------------------------------------START OF INNER GAME--------------------------------------  
        #select our inner-action with Boltzmann sampling, oppo inner-action with epsilon greedy 
        our_action = Boltzmann(env.innerq[0,:].detach().clone())
        oppo_action = env.select_action().detach().clone()      
        
        #print("inner actions: ", our_action, oppo_action)
        #run inner game according to actions
        reward, info = env.step(torch.cat((our_action, oppo_action))) 

        #update inner r matrix [agent, action]
        env.innerr[0, int(our_action)] = reward.detach().clone() 
        env.innerr[1, int(oppo_action)] = info.detach().clone()
        #---------------------------------------END OF INNER GAME--------------------------------------
        #save reward, info for plotting              
        plot_rew[episode,step,0] = reward.detach().clone()
        plot_rew[episode,step,1] = info.detach().clone()
        #Compute new inner Q table, our agent: meta_a that corresponds to max Q(meta_s)/ random, oppo: by Q learning
        env.innerq[0, :] = rmax.select_action(env, meta_s[0], -1)
        env.innerq[1, :] = (1-meta_alpha) * env.innerq[1, :] + torch.Tensor([meta_alpha * info * oppo_action, meta_alpha * info * (1-oppo_action)]).to(device)
        #print("inner-r: ", reward, "\n inner-q: ", env.innerq)

        #meta-action = action that corresponds to max Q(meta_s) = our inner Q
        meta_a = env.innerq[0, :]
        memory.actions.append(meta_a) 

        #meta-state = discretized inner game Q table of all agents
        new_meta_s = discretize(env.innerq.detach().clone(), radius)
        memory.states.append(new_meta_s)    

        #meta-reward = sum of rewards of our agent in inner game of K episodes & T timesteps
        our_REW = reward.detach().clone()                
        memory.rewards.append(our_REW)

        #rmax update step
        rmax.update(env, memory, meta_s, meta_a, new_meta_s)

        #print("updating s-a pair:", rmax.find_meta_index( torch.flatten(meta_s)), rmax.find_meta_index( torch.flatten(meta_a)),"\nrmax.R: ", rmax.R, "\nrmax.Q: ", rmax.Q, "\nrmax.nSA: ", rmax.nSA)
        #print(meta_s, meta_a, new_meta_s)

        #prepare meta_s for next step
        meta_s = new_meta_s.detach().clone()


# In[ ]:


#see which columns are empty
coll = []
for i in range(81):
    if all(rmax.nSA[i] == torch.zeros(9).to(device)):
        coll.append(i) 
print(coll)


# In[ ]:


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


# In[ ]:


#generate heatmap
plt.imshow(rmax.nSA.cpu().numpy(), cmap='hot', interpolation='nearest')
figure1 = plt.gcf()
figure1.set_size_inches(50, 40)
plt.savefig('heatmap for' + str(datetime.now()) + '.png')


# In[ ]:


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


# In[ ]:


#generate learning curve at start
plot_rew_epi_start = torch.mean(plot_rew[:int(meta_epi*0.1), :, 0], 0)
fig_handle = plt.plot(plot_rew_epi_start.cpu().numpy())

plt.xlabel("steps")

plt.ylabel("Average learning rate of first " + str(int(meta_epi*0.1)) + " episodes")

figure3 = plt.gcf() # get current figure
figure3.set_size_inches(10, 8)

plt.savefig('inner_gamma' + str(inner_gamma) + '_rad' + str(radius) + '_' + str(meta_epi) + '_' + str(meta_steps) + '_first_epi_mp1.png' , dpi = 100)
plt.clf()


# In[ ]:


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

# In[3]:


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
