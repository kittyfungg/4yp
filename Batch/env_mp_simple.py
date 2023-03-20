import numpy as np
import random
from collections import deque
import gym
from gym.spaces import Discrete, Tuple

def proper_argmax(lst):
    max_indices = np.zeros(lst.shape[0], dtype=int)
    
    for i in range(lst.shape[0]):
        max_vals = np.where(lst[i] == np.max(lst[i]))[0]
        max_indices[i] = random.choice(max_vals)
        
    return max_indices

def action_on_scale(q_s):
    if q_s >=1 :
        return np.ones((self.bs))
    else:
        return np.zeros((self.bs))
    
class MetaGames:
    def __init__(self, bs, game, radius, meta_steps):
        self.bs = bs
        self.epsilon = 0.8
        self.lr = 0.75
        #reward table with discretized dimensions, (actions, agents) (no states since num of state =1)
        self.innerq = np.zeros((self.bs, 2))
        self.radius = radius
        self.t = 0
        self.game = game
        self.meta_steps= meta_steps
        
    def reset(self):
        #Initialise inner Q table randomly
        self.innerq = np.ones(shape=(self.bs, 2))

        self.init_action = np.random.randint(0,2, size=(self.bs, 2))
        self.t = np.zeros((self.bs, 1))
        return np.concatenate([self.init_action, self.init_action, self.discretise_q(self.innerq), self.t], axis=1) # OBS: INIT_ACTION, ACTION, DISCRETIZED oppo_Q, t

    def discretise_q(self, qtable):
        return np.round(np.divide(qtable, self.radius)) * self.radius

    def select_action(self):
        #select action for opponent only
        if np.random.random() < 1- self.epsilon:
            action = np.random.randint(0,2, (1, )) #convert tuple-->tensor
        else:
            #makes sure if indices have same Q value, randomise
            action = np.argmax(self.innerq, axis=1)
        return action # returns action for all agents

    def step(self, our_action):
        opponent_action = self.select_action()
        action = np.stack([opponent_action, our_action], axis=1)
        r1 = np.zeros(self.bs)
        r2 = np.zeros(self.bs)
        
        if self.game == "MP":
        # PLAY MP GAME, GET REWARDS
            r1 = (opponent_action == our_action) * 1
            r2 = 1 - r1
            self.t += 1
        
        elif self.game == "PD":
        # PLAY PD GAME, GET REWARDS
            for i in range(self.bs):
                if our_action[i] == 0 and opponent_action[i] == 0:  #CC
                    action[i] = 0
                    r1[i] = 3/5
                    r2[i] = 3/5
                elif our_action[i] == 0 and opponent_action[i] == 1:  #CD
                    action[i] = 1
                    r1[i] = 0
                    r2[i] = 1
                elif our_action[i] == 1 and opponent_action[i] == 0:  #DC
                    action[i] = 2
                    r1[i] = 1
                    r2[i] = 0
                elif our_action[i] == 1 and opponent_action[i] == 1:  #DD
                    action[i] = 3
                    r1[i] = 1/5
                    r2[i] = 1/5
            self.t += 1
        
        # GENERATE OBSERVATION
        observation = np.concatenate([self.init_action, np.stack([opponent_action, our_action], axis=1), self.discretise_q(self.innerq), self.t], axis=1) 

        # UPDATE OPPONENT Q-TABLE
        self.innerq[:,opponent_action[0]] = self.lr * r2 + (1 - self.lr) * self.innerq[:,opponent_action[0]]

        # CHECK DONE
        done = self.t > 10

        return observation, r1, r2, {"r1": r1, "r2": r2}
    
class MetaGamesSimple:
    #meta-s = [discretized everyone's inner q, oppo init action, t]   meta-a = our inner q
    def __init__(self, bs, radius):
        self.epsilon = 0.8
        self.lr = 0.75
        self.t = 0
        self.bs = bs
        self.radius= radius
        self.innerq = np.zeros((self.bs,2))

    def reset(self):
        #Initialise inner Q table randomly
        self.innerq = np.random.randint(2, size=(self.bs, 2))
        self.init_action = np.random.randint(2, size=(self.bs, 1))
        self.t = np.zeros((self.bs, 1))
        return np.concatenate([self.discretise_q(self.innerq), self.init_action.T, self.t.T], axis=0).T # OBS: INNERQ, ACTION, TIMESTEP
    
    def discretise_q(self, qtable):
        return np.round(np.divide(qtable, self.radius)) * self.radius
        #return [int(i * self.radius) for i in qtable]

    def select_action(self):
        #select action for opponent only
        if np.random.random() < self.epsilon:
            action = np.random.randint(2, size=(self.bs, 1 )) #convert tuple-->tensor
        else:
            #makes sure if indices have same Q value, randomise
            action = np.reshape(np.argmax(self.innerq, axis=1), (self.bs, 1))
        return action # returns action for all agents

    def step(self, action):
        opponent_action = self.select_action()

        # PLAY GAME, GET REWARDS
        r1 = (opponent_action == action) * 1
        r2 = 1 - r1
        self.t += 1

        # GENERATE OBSERVATION
        observation = np.concatenate([self.discretise_q(self.innerq), self.init_action.T, self.t.T], axis=0).T

        # UPDATE OPPONENT Q-TABLE
        self.innerq[:, opponent_action] = self.lr * r2 + (1 - self.lr) * self.innerq[:, opponent_action]

        # CHECK DONE
        done = self.t > 10

        return observation, r1, done, {"r1": r1, "r2": r2}
    
class MetaGamesLimitedtraj:
    #meta-s = [everyone' action history up to timestep hist_step, t]   meta-a = our inner q
    def __init__(self, bs, hist_step, meta_steps, game):
        self.epsilon = 0.8
        self.lr = 0.75
        self.bs = bs
        self.hist_step = hist_step
        self.meta_steps = meta_steps    
        self.game = game
        
    def reset(self):
        #Initialise inner Q table randomly
        self.init_actions = np.random.randint(2, size=(self.bs, 2))    #the first action for both agents
        self.innerq = np.random.rand(self.bs,2)
        
        self.oppo_deque = [deque(np.random.randint(2, size=(self.hist_step)), maxlen=self.hist_step) for i in range(self.bs)]
        self.our_deque = [deque(np.random.randint(2, size=(self.hist_step)), maxlen=self.hist_step) for i in range(self.bs)]
        #self.oppo_deque = [deque(np.zeros((self.hist_step)), maxlen=self.hist_step) for i in range(self.bs)]
        #self.our_deque = [deque(np.zeros((self.hist_step)), maxlen=self.hist_step) for i in range(self.bs)]
        
        self.t = np.zeros((self.bs, 1))   #for zero-indexings

        #return np.concatenate([self.init_actions, self.oppo_deque, self.t], axis=1) # OBS: INNERQ, ACTION, TIMESTEP
        return np.concatenate([self.oppo_deque, self.our_deque, self.t], axis=1) # OBS: INNERQ, ACTION, TIMESTEP

    def select_action(self):
        #select action for opponent only
        if np.random.random() < 1-self.epsilon:
            action = np.random.randint(2, size=(self.bs )) #convert tuple-->tensor
        else:
            #makes sure if indices have same Q value, randomise
            action = proper_argmax(self.innerq)
        return action # returns action for all agents

    def step(self, action):
        opponent_action = self.select_action()
        r1 = np.zeros((self.bs))
        r2 = np.zeros((self.bs))
        
        if self.game == "MP":
        # PLAY MP GAME, GET REWARDS
            r1 = (opponent_action == action) * 1
            r2 = 1 - r1
            self.t += 1
        
        elif self.game == "PD":
        # PLAY PD GAME, GET REWARDS
            for i in range(self.bs):
                if action[i] == 0 and opponent_action[i] == 0:  #CC
                    r1[i] = 3/5
                    r2[i] = 3/5
                elif action[i] == 0 and opponent_action[i] == 1:  #CD
                    r1[i] = 0
                    r2[i] = 1
                elif action[i] == 1 and opponent_action[i] == 0:  #DC
                    r1[i] = 1
                    r2[i] = 0
                elif action[i] == 1 and opponent_action[i] == 1:  #DD
                    r1[i] = 1/5
                    r2[i] = 1/5
            self.t += 1
        
        # GENERATE OBSERVATION
        for i in range(self.bs):
            self.oppo_deque[i].append(opponent_action[i])
            self.our_deque[i].append(action[i])

#        observation = np.concatenate([self.init_actions, self.oppo_deque, self.t], axis=1)
        observation = np.concatenate([self.oppo_deque, self.our_deque, self.t], axis=1)
        # UPDATE OPPONENT Q-TABLE
        self.innerq[range(self.bs), opponent_action] = self.lr * r2 + (1 - self.lr) * self.innerq[range(self.bs), opponent_action]

        # CHECK DONE
        done = self.t >= self.meta_steps

        return observation, r1, r2, opponent_action
    
class MetaGamesSimplest:
    #meta-s = [oppo_act, our_act, t], meta-a = our_act
    def __init__(self, bs, meta_steps):
        self.epsilon = 0.8
        self.lr = 0.9
        self.bs = bs
        self.max_steps = meta_steps
        #reward table with discretized dimensions, (actions, agents) (no states since num of state =1)

    def reset(self):
        #Initialise inner policy randomly
        #temp_inner_policy = np.random.randint(2, size=(self.bs,))
        #self.init_action = np.random.randint(2, size=(self.bs,))
        temp_inner_policy = np.zeros((self.bs,))
        self.init_action = np.zeros((self.bs,))
        #self.inner_policy = 1 - self.init_action
        self.innerq = np.zeros((self.bs,2))
        self.t = np.ones(self.bs) * -1  #for zero-indexing
        
        return np.stack([temp_inner_policy, self.init_action, self.t], axis=1) # OBS: INNER_ACT, ACTION, TIMESTEP
        #return np.stack([temp_inner_policy, self.init_action], axis=1) # OBS: INNER_ACT, ACTION, TIMESTEP
    
    def step(self, action):
        if np.random.random() < 1-self.epsilon:
            opponent_action = np.random.randint(2, size=(self.bs,))
        else:
            #opponent_action = self.inner_policy
            opponent_action = np.reshape(np.argmax(self.innerq, axis=1), (self.bs, ))

        # PLAY GAME, GET REWARDS
        r1 = (opponent_action == action) * 1
        r2 = 1 - r1
        self.t += 1

        # GENERATE OBSERVATION
        observation = np.stack([opponent_action, action, self.t], axis=1) 
        #observation = np.stack([opponent_action, action], axis=1) 
        
        # UPDATE OPPONENT POLICY
        #self.inner_policy = 1 - action
        self.innerq[range(self.bs), opponent_action] = self.lr * r2 + (1 - self.lr) * self.innerq[range(self.bs), opponent_action]

        # CHECK DONE
        done = self.t >= self.max_steps

        return observation, r1, r2, done
    
    
class MetaGamesTheOne:
    #meta-s = [oppo_act, our_act, our_r, t], meta-a = our_act
    ##for removing t in meta-s
    ###for removing our_r in meta-s
    def __init__(self, bs, step, meta_steps):
        self.epsilon = 0.8
        self.lr = 0.7
        self.bs = bs
        self.oppo_r = np.zeros((self.bs))
        self.our_r = np.zeros((self.bs))
        self.radius = 1/3
        self.choice = np.tile(np.arange(0.1, 0.4, step), (self.bs,1))
        self.interval = int(1/(2*step) + 1)
        self.meta_steps = meta_steps
        self.done = np.zeros((self.bs))
        
    def reset(self):
        #Initialise action randomly   #0 = WAIT, 1 = JUMP
        self.innerq = np.zeros((self.bs, 2))
        self.outerq = np.zeros((self.bs, 2))
        self.done = np.zeros((self.bs))
        
        oppo_idx = np.random.randint(len(self.choice), size=self.bs)
        self.oppo_r = self.choice[np.arange(self.bs), oppo_idx]
        our_idx = np.random.randint(len(self.choice), size=self.bs)
        self.our_r = self.choice[np.arange(self.bs), our_idx]
        
        self.oppo_act = np.zeros((self.bs))
        self.our_act = np.zeros((self.bs))
        
        self.t = np.zeros((self.bs,1)) #for zero-indexing
        
        ###return np.stack([self.oppo_act, self.our_act, self.our_r], axis=1) # OBS: OPPO_ACT, OUR_ACT, OUR_R
        return np.concatenate([self.innerq, self.outerq, self.t], axis=1) # OBS: OPPO_ACT, OUR_ACT, T
    
    def discretise_q(self, qtable):
        return np.round(np.divide(qtable, self.radius)) * self.radius
    
    def step(self, action):
        #opponent choose jump/wait
        if np.random.random() < 1- self.epsilon:
            opponent_action = np.random.randint(0, 2, (1, )) #convert tuple-->tensor
        else:
            opponent_action = proper_argmax(self.innerq[:])

        oppo_idx = np.random.randint(self.choice[0].shape, size=self.bs)
        oppo_advance = self.choice[np.arange(self.bs), oppo_idx]
        our_idx = np.random.randint(self.choice[0].shape, size=self.bs)
        our_advance = self.choice[np.arange(self.bs), our_idx]
        
        # PLAY GAME, GET REWARDS
        for i in range(self.bs):
            if self.done[i] == 0:            #if opponent hasnt done
                #step for opponent
                if (opponent_action[i] == 0) and (self.oppo_r[i] + oppo_advance[i] > 1):    #if wait & crosses 1 before jump
                    self.oppo_r[i] = 0
                    self.done[i] = 1                                              #fking over
                    #print("opponent burst, FINAL REWARD= "+ str(self.oppo_r[i]))

                elif (opponent_action[i] == 0) and (self.oppo_r[i] + oppo_advance[i] <= 1):   #if wait & still within 1
                    self.oppo_r[i] += oppo_advance[i]
                    #print("opponent wait @ x=" + str(self.oppo_r[i]))

                elif opponent_action[i] == 1:                             #if jump and crosses
                    self.oppo_r[i] += oppo_advance[i]
                    self.done[i] = 1                                              #fking over
                    #if self.t[i] == (self.meta_steps-1):
                    #print("opponent jumps & ends @ x=" + str(self.oppo_r[i])+ "@time= " + str(self.t[i]))

                #step for us
                if (action[i] == 0) and (self.our_r[i] + our_advance[i] > 1):
                    self.our_r[i] = 0
                    self.done[i] = 1                                              #fking over
                    #print("we burst, FINAL REWARD= " + str(self.our_r[i]))

                elif (action[i] == 0) and (self.our_r[i] + our_advance[i] <= 1):
                    self.our_r[i] += our_advance[i]
                    #print("we wait @ x=" + str(self.our_r[i]))

                elif action[i] == 1:
                    self.our_r[i] += our_advance[i]
                    self.done[i] = 1                                             #fking over
                    #if self.t[i] == (self.meta_steps -1):
                    #print("we jump & end @ x=" + str(self.our_r[i]) + "@time= " + str(self.t[i]))

            #else for this batch it's done, it retains the previous rewards
            self.t[i] += 1
            #self.done[i] = np.where(self.t[i]>self.meta_steps, [1,1], self.done[i])
            self.done[i] = np.logical_or(self.t[i]>self.meta_steps, self.done[i])
                    
        # GENERATE OBSERVATION
        observation = np.concatenate([self.discretise_q(self.innerq), self.discretise_q(self.outerq), self.t], axis=1)

        # UPDATE OPPONENT POLICY
        self.innerq[:, opponent_action] = self.lr * self.oppo_r + (1 - self.lr) * self.innerq[:, opponent_action]
        self.outerq[:, int(action)] = self.lr * self.our_r + (1 - self.lr) * self.innerq[:, int(action)]

        return observation, self.our_r, self.oppo_r, self.done
   
    
    
    
    
