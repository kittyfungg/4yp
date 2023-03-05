import torch
import numpy as np
import random
import gym
from gym.spaces import Discrete, Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mp_one_iteration():
    dims = [1, 2]
    #/5 to cap reward to 1 for rmax
    payout_mat_1 = torch.Tensor([[1, 0], [0, 1]]).to(device)    #changed -1 entries to 0 since cant negative-index
    payout_mat_2 = torch.Tensor([[0, 1], [1, 0]]).to(device)
    rew1 = 0
    rew2 = 0
    
    def Reward(action):
        x = torch.stack((action[0], 1-action[0]), dim=0)    #[our agent, updated action, batch]
        y = torch.stack((action[1], 1-action[1]), dim=0)
        rew1= torch.matmul(torch.matmul(x, payout_mat_1), y.unsqueeze(-1)).squeeze(-1).detach().clone()
        rew2= torch.matmul(torch.matmul(x, payout_mat_2), y.unsqueeze(-1)).squeeze(-1).detach().clone()

        return [rew1, rew2]
    
    return dims, Reward

class MetaGames:
    def __init__(self):
            
        d, self.game= mp_one_iteration()
        self.epsilon = 0.8
        self.lr = 1
        self.d = d[0]    
        self.num_actions = d[1]
        self.num_agents = 2
        self.action_space = Tuple([Discrete(self.num_actions), Discrete(self.num_actions)])
        
        self.rew1 = 0
        self.rew2 = 0
        
        #reward table with discretized dimensions, (actions, agents) (no states since num of state =1)
        self.innerr = torch.zeros(self.num_agents, self.num_actions).to(device) 
        self.innerq = torch.zeros(self.num_agents, self.num_actions).to(device)     

    def reset(self, info=False):
        #random action of size [2 agents], action value either 0 (Coorperate) or 1 (Defect)
        self.init_action = torch.randint(2, (self.num_agents, )).to(device)
        #Initialise inner R table as 0
        self.innerr = torch.zeros(self.num_agents, self.num_actions).to(device) 

    def select_action(self):
        #select action for opponent only
        if np.random.random() < self.epsilon:
            action = torch.randint(0,2, (1, )).to(device)   #convert tuple-->tensor
        else:
            #makes sure if indices have same Q value, randomise
            poss_max = torch.argwhere(self.innerq[1] == torch.max(self.innerq[1])).to(device) 
            action = random.choice(poss_max)   #find maximum from opponent's inner q table
        return action     #returns action for all agents
        
    def step(self, action):
        r1, r2 = self.game(action.float())
        self.rew1 = r1.detach().clone()
        self.rew2 = r2.detach().clone()
        return r1.detach(), r2.detach()
    
       
