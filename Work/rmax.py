#Rmax code, cuda not yet incompatible
import math
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprob[:]
        del self.rewards[:]
        
class RmaxAgent:

    def __init__(self, env, R_max, gamma, max_episodes, max_steps, radius_dp, epsilon = 0.2):
        self.gamma = gamma
        self.epsilon = epsilon
        self.radius_dp = radius_dp               #added decimal place for Q & R matrix dimension
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.b = env.b
        #Q for meta-game, *2 for 2 player
        #self.Q = torch.ones(((10**(radius_dp)+1) * 2, env.num_actions)).mul(R_max / (1 - self.gamma)).to(device)   
        self.Q = torch.ones(self.b, ((10**(radius_dp)+1) * env.num_actions * 2), ((10**(radius_dp)+1) * env.num_actions)).mul(R_max / (1 - self.gamma)).to(device) 
        #self.R = torch.zeros(((10**(radius_dp)+1) * 2, env.num_actions)).to(device)  
        self.R = torch.zeros(self.b, ((10**(radius_dp)+1) * env.num_actions * 2), ((10**(radius_dp)+1) * env.num_actions)).to(device)
        
        self.nSA = torch.zeros(self.b, ((10**(radius_dp)+1) * env.num_actions * 2), ((10**(radius_dp)+1) * env.num_actions)).to(device)
        
        self.nSAS = torch.ones(self.b, ((10**(radius_dp)+1) * env.num_actions * 2), ((10**(radius_dp)+1) * env.num_actions), ((10**(radius_dp)+1) * env.num_actions * 2)).to(device)
        
        self.val1 = []
        self.val2 = []  #This is for keeping track of rewards over time and for plotting purposes  
        self.m = int(math.ceil(math.log(1 / (self.epsilon * (1-self.gamma))) / (1-self.gamma)))   #calculate m number
        
    def select_action(self, state):
        if np.random.random() > (1-self.epsilon):
            action = env.action_space.sample()
        else:
            action = torch.amax(self.Q[:,state,:], 2)
        return action     #returns action of length b
    
    def update(self, memory, best_action, state):
        
        for i in range(self.b):
            
            if self.nSA[i][state[i]][best_action] < self.m:
                self.nSA[i][state[i]][best_action] +=1
                self.R[i][state[i]][best_action] += memory.rewards[-1]
                self.nSAS[i][state[i]][best_action][new_obs] += 1

                if self.nSA[i][state[i]][best_action] == self.m:

                    for i in range(mnumber):

                        for s in range(env.d * 2):

                            for a in range(env.d):

                                if self.nSA[i][s][a] >= self.m:

                                    #In the cited paper it is given that reward[s,a]= summation of rewards / nSA[s,a]
                                    #We have already calculated the summation of rewards in line 28
                                    q = (self.R[i][s][a]/self.nSA[i][s][a])

                                    for next_state in range(env.d * 2):

                                        #In the cited paper it is given that transition[s,a] = nSAS'[s,a,s']/nSA[s,a]

                                        transition = self.nSAS[i][s][a][next_state]/self.nSA[i][s][a]
                                        q += (transition * torch.max(self.Q[i,next_state,:]))

                                    self.Q[i][s][a] = q 
                                    #print(q + self.gamma*(self.R[state][action]/self.nSA[state][action]))
                                    #In the cited paper it is given that reward[s,a]= summation of rewards / nSA[s,a]
                                    #We have already calculated the summation of rewards in line 28
            
            memory.state = new_state  

    def mean_rewards_per_500(self):
        
        total_reward = 0
        for episodes in range(500):
            state = env.reset()
            for _ in range(1000):

                action = self.choose_action(state)
                state, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    state = env.reset()
                    break
        return (total_reward/500) 
    
    

    