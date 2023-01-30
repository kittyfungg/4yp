import torch
import os.path as osp
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
       
def pd_one_iteration_batched(bs):
    dims = [1, 2]
    #/5 to cap reward to 1 for rmax, [0,0] CC, [1,1] DD
    payout_mat_1 = torch.Tensor([[3/5, 0], [5/5, 1/5]]).to(device)
    payout_mat_2 = payout_mat_1.T
    rew1 = torch.empty(bs).to(device)
    rew2 = torch.empty(bs).to(device)
    
    def Reward(action):
        for i in range(bs):
            x = torch.stack((action[0,i], 1-action[0,i]), dim=0)    #[our agent, updated action, batch]
            y = torch.stack((action[1,i], 1-action[1,i]), dim=0)
            rew1[i] = torch.matmul(torch.matmul(x, payout_mat_1), y.unsqueeze(-1)).squeeze(-1)
            rew2[i] = torch.matmul(torch.matmul(x, payout_mat_2), y.unsqueeze(-1)).squeeze(-1)

        return [rew1, rew2]
    
    return dims, Reward

def mp_one_iteration_batched(bs):
    dims = [1, 2]
    #/5 to cap reward to 1 for rmax
    payout_mat_1 = torch.Tensor([[-1, 1], [1, -1]]).to(device)
    payout_mat_2 = payout_mat_1.T
    rew1 = torch.empty(bs).to(device)
    rew2 = torch.empty(bs).to(device)
    
    def Reward(action):
        for i in range(bs):
            x = torch.stack((action[0,i], 1-action[0,i]), dim=0)    #[our agent, updated action, batch]
            y = torch.stack((action[1,i], 1-action[1,i]), dim=0)
            rew1[i] = torch.matmul(torch.matmul(x, payout_mat_1), y.unsqueeze(-1)).squeeze(-1)
            rew2[i] = torch.matmul(torch.matmul(x, payout_mat_2), y.unsqueeze(-1)).squeeze(-1)

        return [rew1, rew2]
    
    return dims, Reward

class MetaGames:
    def __init__(self, b, opponent="NL", game="IPD"):
        """
        Opponent can be:
        NL = Naive Learner (gradient updates through environment).
        LOLA = Gradient through NL.
        STATIC = Doesn't learn.
        COPYCAT = Copies what opponent played last step.
        """
        self.gamma_inner = 0.96
        self.b = b
        
        d, self.game_batched = pd_one_iteration_batched(b)
        self.std = 1
        self.lr = 1
        self.d = d[0]    
        self.num_actions = d[1]
        self.num_agents = 2
        
        self.rew1 = torch.empty(b).to(device)
        self.rew2 = torch.empty(b).to(device)
        
        #reward table with discretized dimensions, (batch_size, actions, agents)
        self.innerr = torch.zeros(self.b, self.num_actions, self.num_agents).to(device) 
        self.innerq = torch.zeros(self.b, self.num_actions, self.num_agents).to(device)     

    def reset(self, info=False):
        #random action of size [2 agents, size.b], action value either 0 (Coorperate) or 1 (Defect)
        self.init_action = torch.randint(2, (self.num_agents, self.b)).to(device)
        #state, _, _ = self.step(self.init_action)
        self.innerr = torch.zeros(self.b, self.num_actions, self.num_agents).to(device) 
        #return state

    def step(self, action):
        r1, r2 = self.game_batched(action.float())
        self.rew1 = r1
        self.rew2 = r2

        return r1.detach(), r2.detach()
    
    def choose_action(self):
    #chooses action that corresponds to the max Q value of the particular agent
        best_action = torch.empty((self.num_agents, self.b), dtype = torch.int64).to(device)
        
        for i in range(self.b):
            best_action[0,i] = torch.argmax(self.innerq[i, :, 0])
            best_action[1,i] = torch.argmax(self.innerq[i, :, 1])
        return best_action.detach()
            
       
