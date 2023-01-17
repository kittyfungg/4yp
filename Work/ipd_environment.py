import torch
import os.path as osp
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
def get_gradient(function, param):
    grad = torch.autograd.grad(function, param, create_graph=True, allow_unused=True)[0]
    return grad

def compute_best_response(outer_th_ba):
    batch_size = 1
    std = 0
    num_steps = 1000
    lr = 1

    ipd_batched_env = ipd_batched(batch_size, gamma_inner=0.96)[1]
    inner_th_ba = torch.nn.init.normal_(torch.empty((batch_size, 5), requires_grad=True), std=std).cuda()
    for i in range(num_steps):
        th_ba = [inner_th_ba, outer_th_ba.detach()]
        l1, l2, M = ipd_batched_env(th_ba)
        grad = get_gradient(l1.sum(), inner_th_ba)
        with torch.no_grad():
            inner_th_ba -= grad * lr
    print(l1.mean() * (1 - 0.96))
    return inner_th_ba


def generate_mamaml(b, d, inner_env, game, inner_lr=1):
    """
    This is an improved version of the algorithm presented in this paper:
    https://arxiv.org/pdf/2011.00382.pdf
    Rather than calculating the loss using multiple policy gradients terms,
    this approach instead directly takes all of the gradients through because the environment is differentiable.
    """
    outer_lr = 0.01
    mamaml = torch.nn.init.normal_(torch.empty((1, d), requires_grad=True, device=device), std=1.0)
    alpha = torch.rand(1, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([mamaml, alpha], lr=outer_lr)

    for ep in range(1000):
        agent = mamaml.clone().repeat(b, 1)
        opp = torch.nn.init.normal_(torch.empty((b, d), requires_grad=True), std=1.0).cuda()
        total_agent_loss = 0
        total_opp_loss = 0
        for step in range(100):
            l1, l2, M = inner_env([opp, agent])
            total_agent_loss = total_agent_loss + l2.sum()
            total_opp_loss = total_opp_loss + l1.sum()

            opp_grad = get_gradient(l1.sum(), opp)
            agent_grad = get_gradient(l2.sum(), agent)
            opp = opp - opp_grad * inner_lr
            agent = agent - agent_grad * alpha

        optimizer.zero_grad()
        total_agent_loss.sum().backward()
        optimizer.step()
        print(total_agent_loss.sum().item())

    torch.save((mamaml, alpha), f"mamaml_{game}.th")
    
def ipd_one_iteration_batched(bs, gamma_inner=0.96):
    dims = [1, 1]
    payout_mat_1 = torch.Tensor([[-1, -3], [0, -2]]).to(device)
    payout_mat_2 = payout_mat_1.T
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)

    def Ls(th):
        p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
        x, y = torch.cat([p_1, 1-p_1], dim=-1), torch.cat([p_2, 1-p_2], dim=-1)
        L_1 = -torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_1), y.unsqueeze(-1))
        L_1 = torch.reshape(L_1, (bs,1,1)) 
        L_2 = -torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_2), y.unsqueeze(-1))
        L_2 = torch.reshape(L_2, (bs,1,1))
        return [L_1.squeeze(-1), L_2.squeeze(-1), None]
    return dims, Ls
    
    
class MetaGames:
    def __init__(self, b, opponent="NL", game="IPD", mmapg_id=0):
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

        d, self.game_batched = ipd_one_iteration_batched(b, gamma_inner=self.gamma_inner)
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(self.num_actions), gym.spaces.Discrete(self.num_actions)])
        self.std = 1
        self.lr = 1
        self.d = d[0]    

        self.opponent = opponent
        self.init_th_ba = None
            
    def reset(self, info=False):
        if self.init_th_ba is not None:
            self.inner_th_ba = self.init_th_ba.detach() * torch.ones((self.b, self.d), requires_grad=True).to(device)
        else:
            self.inner_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)
        outer_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)
        state, _, _, M = self.step(outer_th_ba)
        if info:
            return state, M
        else:
            return state

    def step(self, outer_th_ba):
        last_inner_th_ba = self.inner_th_ba.detach().clone()
        th_ba = [self.inner_th_ba, outer_th_ba.detach()]
        l1, l2, M = self.game_batched(th_ba)
        grad = get_gradient(l1.sum(), self.inner_th_ba)
        with torch.no_grad():
            self.inner_th_ba -= grad * self.lr

        return torch.sigmoid(torch.cat((outer_th_ba, last_inner_th_ba), dim=-1)).detach(), (-l2 * (1 - self.gamma_inner)).detach(), (-l1 * (1 - self.gamma_inner)).detach(), M
       
