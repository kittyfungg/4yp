import numpy as np
import random
import gym
from gym.spaces import Discrete, Tuple

def mp_one_iteration(bs):
    dims = [1, 2]
    #/5 to cap reward to 1 for rmax
    payout_mat_1 = np.array([1, 0, 0, 1])   #changed -1 entries to 0 since cant negative-index
    payout_mat_2 = np.array([0, 1, 1, 0])
    rew1 = 0
    rew2 = 0
    
    def Reward(action):
        #x = np.stack([action[0], 1-action[0]], axis=1)    #[our agent, updated action, batch]
        #y = np.stack([action[1], 1-action[1]], axis=1)
        P = np.stack([action[0] * action[1], action[0] * (1 - action[1]), (1 - action[0]) * action[1], (1 - action[0]) * (1 - action[1])], axis=1)
        rew1= np.matmul(P, payout_mat_1)
        rew2= np.matmul(P, payout_mat_2)

        return [rew1, rew2]
    
    return dims, Reward

class MetaGames:
    def __init__(self, bs):
        
        d, self.game= mp_one_iteration(bs)
        self.epsilon = 0.8
        self.lr = 1
        self.bs = bs
        self.d = d[0]    
        self.num_actions = d[1]
        self.num_agents = 2
        self.action_space = Tuple([Discrete(self.num_actions), Discrete(self.num_actions)])
        
        self.rew1 = 0
        self.rew2 = 0
        
        #reward table with discretized dimensions, (actions, agents) (no states since num of state =1)
        self.innerr = np.zeros((self.bs, self.num_agents, self.num_actions))
        self.innerq = np.zeros((self.bs, self.num_agents, self.num_actions))   

    def reset(self, info=False):
        #random action of size [2 agents], action value either 0 (Coorperate) or 1 (Defect)
        self.init_action = np.random.randint(2, size = (self.bs, self.num_agents))
        #Initialise inner R table as 0
        self.innerr = np.zeros((self.bs, self.num_agents, self.num_actions))

    def select_action(self):
        action = np.zeros(self.bs)
        #select action for opponent only
        if np.random.random() < self.epsilon:
            action = np.random.randint(2, size = (self.bs))   
        else:
            #makes sure if indices have same Q value, randomise
            for b in range(self.bs):
                poss_max = np.argwhere(self.innerq[b,1] == np.amax(self.innerq[b,1,:]))
                action[b] = np.random.choice(poss_max.squeeze())   #find maximum from opponent's inner q table
        return action     #returns action for all agents
        
    def step(self, action):
        r1, r2 = self.game(action)
        self.rew1 = r1
        self.rew2 = r2
        return r1, r2
    
       
