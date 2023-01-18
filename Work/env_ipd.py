import torch
import os.path as osp
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
       
def pd_one_iteration_batched(bs):
    dims = [2, 2]
    #/3 to cap reward to 1 for rmax
    payout_mat_1 = torch.Tensor([[1/3, 1], [0, 2/3]]).to(device)
    payout_mat_2 = payout_mat_1.T

    def Ls(action):
        loss1 = torch.empty(bs).to(device)
        loss2 = torch.empty(bs).to(device)
        
        for i in range(bs):
            x = torch.stack((action[0,i], (1-action[0,i])), dim=-1)
            y = torch.stack((action[1,i], (1-action[1,i])), dim=-1)
            L_1 = torch.matmul(torch.matmul(x, payout_mat_1), y)
            L_2 = torch.matmul(torch.matmul(x, payout_mat_2), y)
            loss1[i] = L_1
            loss2[i] = L_2
        return [loss1, loss2]
    return dims, Ls

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
        self.num_actions = 2

        d, self.game_batched = pd_one_iteration_batched(b)
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(self.num_actions), gym.spaces.Discrete(self.num_actions)])
        self.std = 1
        self.lr = 1
        self.d = d[0]    

        self.opponent = opponent
        #reward table with discretized dimensions, (batch_size, states, actions, agents)
        self.innerr = torch.zeros(self.b, self.num_actions**2 , self.num_actions, 2).to(device) 
        #states = 4 for IPD (CC,CD,DC,DD) and actions = 2 (C,D)
        self.innerq = torch.zeros(self.b, self.num_actions**2 , self.num_actions, 2).to(device)     
        #inner visitation freq
        #self.innernSA = torch.zeros(self.b, self.num_actions**2 , self.num_actions, 2).to(device) 
        #self.innernSAS = torch.zeros(self.b, self.num_actions**2 , self.num_actions, self.num_actions**2, 2).to(device) 
            
    def reset(self, info=False):
        #random action of size [size.d, size.b], action value either 0 (Coorperate) or 1(Defect)
        rand_action = torch.randint(2, (self.d, self.b)).to(device)        
        state, _, _ = self.step(rand_action)
        self.innerr = torch.zeros(self.b, self.num_actions**2 , self.num_actions, 2).to(device) 
        return state


    def step(self, action):
        l1, l2 = self.game_batched(action.float())
        state = torch.empty(self.b, dtype = torch.long).to(device)
        for i in range(self.b):
            if action[:,i].tolist() == [0,0]:     #CC
                state[i] = 0
            elif action[:,i].tolist() == [0,1]:   #CD
                state[i] = 1
            elif action[:,i].tolist() == [1,0]:   #DC
                state[i] = 2
            elif action[:,i].tolist() == [1,1]:   #DD
                state[i] = 3

        return state.detach(), l1.detach(), l2.detach()
    
    def choose_action(self, state):
    #chooses action that corresponds to the max Q value of the particular agent
        best_action = torch.empty((self.d, self.b), dtype = torch.int64).to(device)
        for i in range(self.b):
            best_action[:,i] = torch.argmax(self.innerq[i, state[i], :, :], dim=-2)
        return best_action.detach()
            
       
