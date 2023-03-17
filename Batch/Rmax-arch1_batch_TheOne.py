#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:
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

from env_mp_simple import MetaGamesTheOne
from rmax_1_batch_TheOne import RmaxAgent, Memory

def discretize(number, radius): 
    #[0,3,5,4,8] --> [0,3,6,3,9] for radius 3
    return np.round(np.divide(number, radius)) * radius

def Boltzmann(arr):
    #0.5 is just a temperature parameter, controls the spread of the softmax distribution
    action_value = np.zeros(arr.shape[0])
    prob = softmax(arr/0.4, 1)
    for b in range(arr.shape[0]):
        action_value[b] = np.random.choice(np.arange(arr.shape[1]), p=prob[b])
    return np.reshape(action_value, (bs))

# In[ ]:


bs = 1
inner_gamma = 0         #inner game discount factor, 0 since it's a one shot game
meta_gamma = 0.8          #meta game discount factor
meta_alpha = 0.4          #meta game learning rate
R_max = 1
rmax_error = 0.5
meta_epi = 40000

step=0.05              #distance between each interval for inner game
epsilon = 0

# creating environment
env = MetaGamesTheOne(bs, step)

# creating rmax agent
memory = Memory()
rmax = RmaxAgent(R_max, bs, meta_gamma, inner_gamma, step, epsilon, rmax_error)

#reward tensor for plotting purposes [bs, episode, step, agents]
plot_rew = [[[[0]*2 for i in range(2)] for j in range(meta_epi)]for k in range(bs)]


# In[ ]:
for episode in range(meta_epi): #for each meta-episode
    #reset environment 
    #initialise meta-state and meta-action randomly
    meta_s = env.reset()
    memory.states.append(meta_s)
    done = env.done
   
    while any(done)==0:     #while any batches still arent "done" with the game
        print(done)
        #--------------------------------------START OF INNER GAME--------------------------------------  
        #select our inner-action with Boltzmann sampling, oppo inner-action with epsilon greedy 
        our_innerq = rmax.Q[range(bs), rmax.find_meta_index(meta_s, "s").astype(int), :]
        
        #select our inner-action with from innerq = meta-a
        our_action = Boltzmann(our_innerq)
    
        #print("inner actions: ", our_action, oppo_action)
        #run inner game according to actions
        obs, reward, info, done = env.step(our_action)  
        #---------------------------------------END OF INNER GAME--------------------------------------
        #save reward, info for plotting       
        plot_rew[0][episode][0].append(reward[0])
        plot_rew[0][episode][1].append(info[0])

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

        #prepare meta_s for next step
        meta_s = new_meta_s

# # Plots

# In[5]:


#generate histogram
visit_dict = {}
for i in range(len(rmax.nSA[0].flatten().tolist())):
    visit_dict[i]= rmax.nSA[0].flatten().tolist()[i]
    
histogram_dict = Counter(visit_dict.values())
plt.bar(histogram_dict.keys(), histogram_dict.values(), 0.5, color='g')
plt.xlabel("visitation counts: " + str(histogram_dict), fontsize=12)
figure0 = plt.gcf()
figure0.set_size_inches(10, 8)
plt.savefig('histogram at' + str(datetime.now()) + '.png')


# In[ ]:

plt.clf()
#generate heatmap
plt.imshow(rmax.nSA[0], cmap='hot', interpolation='nearest')
figure1 = plt.gcf()
figure1.set_size_inches(50, 200)
plt.savefig('heatmap for' + str(datetime.now()) + '.png')


# In[ ]:

plt.clf()
#generate reward mean per step of batch 0
fig_handle = plt.plot(plot_rew[0][:][0])
#reward at batch 0 only
plt.xlabel("episodes \n Average reward of our agent: " + str(np.mean(plot_rew[0][:][0])) + 
          "\n Average reward of another agent: " + str(np.mean(plot_rew[0][:][1])))

plt.ylabel("Mean rewards")

figure2 = plt.gcf() # get current figure
figure2.set_size_inches(10, 8)
plt.savefig('reward for' + str(datetime.now()) + '.png')

