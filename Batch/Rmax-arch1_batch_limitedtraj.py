#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from scipy.special import softmax
import random
import pickle
from datetime import datetime
from collections import Counter


import gym
from gym.spaces import Discrete, Tuple

from env_mp_simple import MetaGamesLimitedtraj
from rmax_1_batch_limitedtraj import RmaxAgentTraj, Memory

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


bs = 1
inner_gamma = 0         #inner game discount factor, 0 since it's a one shot game
meta_gamma = 0         #meta game discount factor
meta_alpha = 0.4          #meta game learning rate
R_max = 1
rmax_error = 0.5
meta_epi = 30000
meta_steps = 30

game = "MP"
epsilon = 0.8
radius = 1              #radius for discretization, assuming radius>1
hist_step= 2

#reward tensor for plotting purposes [bs, episode, step, agents]
plot_rew = np.zeros((bs, meta_epi, meta_steps, 2))

# creating environment
env = MetaGamesLimitedtraj(bs, hist_step, meta_steps, game)

# creating rmax agent
memory = Memory()
rmax = RmaxAgentTraj(R_max, bs, meta_steps, meta_gamma, inner_gamma, radius, epsilon, rmax_error, hist_step)


for episode in range(meta_epi) : #for each meta-episode
    #reset environment 
    #initialise meta-state and meta-action randomly
    meta_s = env.reset()
   
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

        #meta-action = action that corresponds to max Q(meta_s) = our inner Q
        meta_a = our_action

        #meta-state = discretized inner game Q table of all agents
        new_meta_s = obs

        #meta-reward = sum of rewards of our agent in inner game of K episodes & T timesteps
        our_REW = reward               
        memory.rewards.append(our_REW)

        #rmax update step
        rmax.update(memory, meta_s, meta_a, new_meta_s)

        #prepare meta_s for next step
        meta_s = new_meta_s


# # Plots
plt.clf()
#generate histogram
visit_dict = {}
for i in range(len(rmax.nSA[0].flatten().tolist())):
    visit_dict[i]= rmax.nSA[0].flatten().tolist()[i]
    
histogram_dict = Counter(visit_dict.values())
plt.bar(histogram_dict.keys(), histogram_dict.values(), 0.5, color='g')
plt.xlabel("visitation counts: " + str(histogram_dict), fontsize=12)
figure0 = plt.gcf()
figure0.set_size_inches(10, 8)
figure0.savefig(game + 'histogram at' + str(datetime.now()) + '.png')

plt.clf()
#generate reward mean per step of all batches
plot_rew_mean = np.mean(plot_rew[0,:,:,0], axis=1)
fig_handle = plt.plot(plot_rew_mean)

plt.xlabel("episodes \n Average reward of our agent: " + str(np.mean(plot_rew[0,:,:,0])) + 
          "\n Average reward of another agent: " + str(np.mean(plot_rew[0,:,:,1])))
plt.ylabel("Mean rewards")

figure1 = plt.gcf() # get current figure
figure1.set_size_inches(10, 8)

figure1.savefig(game + 'hist' + str(hist_step) + '_epi' + str(meta_epi) + '_step' + str(meta_steps) + '_mp1.png'  , dpi = 100)

plt.clf()
#generate reward of first episode
plot_rew_epi_start = plot_rew[0, 1, :, 0]
fig_handle = plt.plot(plot_rew_epi_start)

plt.xlabel("steps")
plt.ylabel("Reward for first episode, all timesteps")

figure2 = plt.gcf() # get current figure
figure2.set_size_inches(10, 8)

figure2.savefig(game + 'hist' + str(hist_step)  + '_epi' + str(meta_epi) + '_step' + str(meta_steps) + '_first_epi.png' , dpi = 100)

plt.clf()
#generate reward of last episode
plot_rew_epi_start = plot_rew[0, -1, :, 0]
fig_handle = plt.plot(plot_rew_epi_start)

plt.xlabel("steps")
plt.ylabel("Reward for last episode, all timesteps")

figure3 = plt.gcf() # get current figure
figure3.set_size_inches(10, 8)

figure3.savefig(game + 'hist' + str(hist_step)  + '_epi' + str(meta_epi) + '_step' + str(meta_steps) + '_last_epi.png' , dpi = 100)

plt.clf()
#generate learning curve of first 10
plot_rew_epi_start = np.mean(plot_rew[0, :10, :, 0], axis=0)
fig_handle = plt.plot(plot_rew_epi_start)

plt.xlabel("steps" + "\n Average reward of first 10 episodes" + str(np.mean(plot_rew[0,:10,:,0])))

plt.ylabel("Average learning rate of first 10 episodes")

figure4 = plt.gcf() # get current figure
figure4.set_size_inches(10, 8)

figure4.savefig(game + 'hist' + str(hist_step) + '_epi' + str(meta_epi) + '_step' + str(meta_steps) + '_first10_epi_lr.png' , dpi = 100)


plt.clf()
#generate learning curve of last 10
plot_rew_epi_end = np.mean(plot_rew[0, -10:, :, 0], axis=0)
fig_handle = plt.plot(plot_rew_epi_end)

plt.xlabel("steps" + "\n Average reward of last 10 episodes" + str(np.mean(plot_rew[0,-10:,:,0])))
plt.ylabel("Average learning rate of last 10 episodes")

figure5 = plt.gcf() # get current figure
figure5.set_size_inches(10, 8)

figure5.savefig(game + 'hist' + str(hist_step)  + '_epi' + str(meta_epi) + '_step' + str(meta_steps) + '_last_epi10_lr.png' , dpi = 100)




