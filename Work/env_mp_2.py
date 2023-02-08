import torch
import numpy as np
import os.path as osp
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
    def __init__(self, opponent="NL", game="IPD"):
        """
        Opponent can be:
        NL = Naive Learner (gradient updates through environment).
        LOLA = Gradient through NL.
        STATIC = Doesn't learn.
        COPYCAT = Copies what opponent played last step.
        """
        self.gamma_inner = 0.96
        
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
        if np.random.random() > (1-self.epsilon):
            action = torch.Tensor(list(self.action_space.sample())).to(device)   #convert tuple-->tensor
        else:
            action = torch.argmax(self.innerq, 1).to(device)    #find maximum from the second dimension(action dimension)
        return action     #returns action for all agents
        
    def step(self, action):
        r1, r2 = self.game(action.float())
        self.rew1 = r1.detach().clone()
        self.rew2 = r2.detach().clone()
        return r1.detach(), r2.detach()
    
    def choose_action(self):
    #chooses action that corresponds to the max Q value of the particular agent
        best_action = torch.empty((self.num_agents), dtype = torch.int64).to(device)
        
        best_action[0] = torch.argmax(self.innerq[0, :])
        best_action[1] = torch.argmax(self.innerq[1, :])
        return best_action.detach()
            
       
