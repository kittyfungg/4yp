import torch
import random
import numpy as np
import os.path as osp
import gym
from gym.spaces import Discrete, Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mp_one_iteration():
    dims = [1, 2]
    #changed -1 entries to 0 since cant negative-index
    payout_mat_1 = torch.Tensor([[1, 0], [0, 1]]).to(device)   
    payout_mat_2 = torch.Tensor([[0, 1], [1, 0]]).to(device)
    rew1 = 0
    rew2 = 0
    
    def Reward(action):
        x = torch.stack((action[0], 1-action[0]), dim=0)    #[our agent, updated action]
        y = torch.stack((action[1], 1-action[1]), dim=0)
        rew1= torch.matmul(torch.matmul(x, payout_mat_1), y.unsqueeze(-1)).squeeze(-1).detach().clone()
        rew2= torch.matmul(torch.matmul(x, payout_mat_2), y.unsqueeze(-1)).squeeze(-1).detach().clone()
        return [rew1, rew2]
    
    return dims, Reward

def pd_one_iteration():
    dims = [5, 2]
    #/5 to cap reward to 1 for rmax
    payout_mat_1 = torch.Tensor([[3/5, 0], [1, 1/5]]).to(device)
    payout_mat_2 = payout_mat_1.T
    rew1 = 0
    rew2 = 0
    
    def Reward(action):
        x = action[0].to(device)    #[our agent, updated action]
        y = action[1].to(device)
        rew1= torch.matmul(torch.matmul(x, payout_mat_1), y.unsqueeze(-1)).squeeze(-1).detach().clone()
        rew2= torch.matmul(torch.matmul(x, payout_mat_2), y.unsqueeze(-1)).squeeze(-1).detach().clone()
        return [rew1, rew2]

    def state_mapping(state, pov):
        #CC:1, CD:2, DC:3, DD:4
        if state[0] == [1,0] and state[1] == [1,0]:
            ind = 1
        elif state[0] == [1,0] and state[1] == [0,1]:
            if pov == "our":
                ind = 2
            else:
                ind = 3
        elif state[0] == [0,1] and state[1] == [1,0]:
            if pov == "our":
                ind = 3
            else: 
                ind = 2
        elif state[0] == [0,1] and state[1] == [0,1]:
            ind = 4
        return ind
    
    def action_mapping(action):
        #returns 1 value: 0 = C = [1,0] ,  1 = D = [0,1]
        return int(action[0]*0 + action[1]*1)
    
    def action_unmapping(action_val):
        #returns a 2-element array: 
        return torch.Tensor([1-action_val.item(), action_val.item()])
        
    return dims, Reward, state_mapping, action_mapping, action_unmapping
class MetaGames:
    def __init__(self, game):
        self.gamma_inner = 0.96
        if game == "PD":
            d, self.game, self.state_mapping, self.action_mapping, self.action_unmapping = pd_one_iteration()
        #elif game == "MP":
        #    d, self.game= mp_one_iteration()
            
        self.epsilon = 0.8
        self.lr = 1
        self.d = d[0]    
        self.num_actions = d[1]
        self.num_agents = 2
        self.action_space = Tuple([Discrete(self.num_actions), Discrete(self.num_actions)])
        
        self.rew1 = 0
        self.rew2 = 0
        
        #reward table with discretized dimensions, (agents, states, actions) 
        self.innerr = torch.zeros(self.num_agents, self.d, self.num_actions).to(device) 
        self.innerq = torch.zeros(self.num_agents, self.d, self.num_actions).to(device)     

    def reset(self, info=False):
        #random action of size [2 agents], action value either 0 (Coorperate) or 1 (Defect)
        seed = torch.randint(2, (self.num_agents,)).to(device)
        self.init_state = [[1-seed[0].item(), seed[0].item()], [1-seed[1].item(), seed[1].item()]]
        #Initialise inner R table as 0
        self.innerr = torch.zeros(self.num_agents, self.d, self.num_actions).to(device) 
        self.innerq = torch.zeros(self.num_agents, self.d, self.num_actions).to(device) 

    def select_action(self, state):
        #return inner action [C,D] according to oppo innerq
        if np.random.random() > (1-self.epsilon):
            #convert tuple-->tensor
            seed = random.randint(0,1)
            action = torch.Tensor([1-seed, seed]).to(device)
        else:
            #find maximum from action dimension
            action = self.action_unmapping(torch.argmax(self.innerq[1, state, :]).to(device))    
        return action.to(device)     #returns action for oppo
        
    def step(self, our_action, oppo_action): 
        state = [our_action, oppo_action]
        r1, r2 = self.game(state)
        self.rew1 = r1.detach().clone()
        self.rew2 = r2.detach().clone()
        return state, r1.detach(), r2.detach()
    
            
       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
