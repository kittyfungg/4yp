import numpy as np
import random
from collections import deque
import gym
from gym.spaces import Discrete, Tuple

class MetaGames:
    def __init__(self):
        self.epsilon = 0.8
        self.lr = 0.75
        #reward table with discretized dimensions, (actions, agents) (no states since num of state =1)
        self.innerq = [0, 0]
        self.t = 0

    def reset(self):
        #Initialise inner Q table randomly
        self.innerq = np.random.random(shape=(2,))
        self.outerq = np.random.random(shape=(2,))
        # self.init_action = np.random.randint(0,2)
        self.t = 0
        return np.concatenate([self.discretise_q(self.innerq), self.discretise_q(self.outerq), self.t], axis=0) # OBS: INNERQ, ACTION, TIMESTEP

    def discretise_q(self, qtable):
        return [int(i * 5) for i in qtable]

    def select_action(self):
        #select action for opponent only
        if np.random.random() < self.epsilon:
            action = np.random.randint(0,2,(1, )) #convert tuple-->tensor
        else:
            #makes sure if indices have same Q value, randomise
            action = np.argmax(self.innerq)
        return action # returns action for all agents

    def step(self, outer_qtable):
        action = np.argmax(outer_qtable) # PLAY WITH STOCHASTIC SAMPLING OF OUTER Q?
        opponent_action = self.select_action()

        # PLAY GAME, GET REWARDS
        r1 = int(opponent_action == action)
        r2 = 1 - r1
        self.t += 1

        # GENERATE OBSERVATION
        observation = np.concatenate([
            self.discretise_q(self.innerq),
            self.discretise_q(outer_qtable),
            self.t
        ], axis=0) # MAKE SURE SHAPES LINE UP

        # UPDATE OPPONENT Q-TABLE
        self.innerq[opponent_action] = self.lr * r2 + (1 - self.lr) * self.innerq[opponent_action]

        # CHECK DONE
        done = self.t > 10

        return observation, r1, done, {"r1": r1, "r2": r2}
    
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
        self.lr = 0.9
        self.t = 0
        self.bs = bs
        self.hist_step = hist_step
        self.meta_steps = meta_steps
        
        self.innerq = np.zeros((self.bs,2))
        self.oppo_deque = deque(np.zeros((self.hist_step, self.bs)), maxlen=self.hist_step)
        self.our_deque = deque(np.zeros((self.hist_step, self.bs)), maxlen=self.hist_step)
        
        self.game = game
    def reset(self):
        #Initialise inner Q table randomly
        self.innerq = np.zeros((self.bs,2))
        
        self.our_deque.append(np.zeros((self.bs, )))
        self.oppo_deque.append(np.zeros((self.bs, )))
        
        self.t = np.ones((self.bs))* -1  #for zero-indexings
        return np.stack([self.oppo_deque[0], self.our_deque[0], self.t], axis=1) # OBS: INNERQ, ACTION, TIMESTEP
    
    def select_action(self):
        #select action for opponent only
        if np.random.random() < 1-self.epsilon:
            action = np.random.randint(2, size=(self.bs )) #convert tuple-->tensor
        else:
            #makes sure if indices have same Q value, randomise
            action = np.reshape(np.argmax(self.innerq, axis=1), (self.bs))
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
                if action[i] == 0 & opponent_action[i] == 0:
                    r1[i] = 3/5
                    r2[i] = 3/5
                elif action[i] == 0 & opponent_action[i] == 1:
                    r1[i] = 0
                    r2[i] = 1
                elif action[i] == 1 & opponent_action[i] == 0:
                    r1[i] = 1
                    r2[i] = 0
                elif action[i] == 1 & opponent_action[i] == 1:
                    r1[i] = 1/5
                    r2[i] = 1/5
        
        # GENERATE OBSERVATION
        self.oppo_deque.append(opponent_action)
        self.our_deque.append(action)
        observation = np.stack([self.oppo_deque[0], self.our_deque[0], self.t], axis=1)

        # UPDATE OPPONENT Q-TABLE
        self.innerq[range(self.bs), opponent_action] = self.lr * r2 + (1 - self.lr) * self.innerq[range(self.bs), opponent_action]

        # CHECK DONE
        done = self.t >= self.meta_steps

        #return observation, r1, done, {"r1": r1, "r2": r2}
        return observation, r1, r2, done
    
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

        return observation, r1, done, {"r1": r1, "r2": r2}
    
    
class MetaGamesTheOne:
    #meta-s = [oppo_act, our_act, our_r, t], meta-a = our_act
    def __init__(self, bs, step):
        self.epsilon = 0.8
        self.lr = 0.9
        self.bs = bs
        self.oppo_r = np.zeros((self.bs))
        self.our_r = np.zeros((self.bs))

        self.final_oppo_r = 0
        self.final_our_r = 0
        
        self.choice = np.tile(np.arange(0, 1, step), (self.bs,1))
        self.interval = int(1//step + 1)
        self.done = np.zeros((self.bs))
        
    def reset(self):
        #Initialise action randomly
        self.done = np.zeros((self.bs))
        
        oppo_idx = np.random.randint(len(self.choice), size=self.bs)
        self.oppo_r = self.choice[np.arange(self.bs), oppo_idx]
        our_idx = np.random.randint(len(self.choice), size=self.bs)
        self.our_r = self.choice[np.arange(self.bs), our_idx]
        
        self.oppo_act = np.random.randint(2, size=(self.bs))    #0 = WAIT, 1 = JUMP
        self.our_act = np.random.randint(2, size=(self.bs))
        
        self.innerq = np.zeros((self.bs, (2*self.interval), 2))    #our inner q state = oppo_act * our_rew, action = 2
        self.t = np.ones(self.bs) * -1  #for zero-indexing
        
        return np.stack([self.oppo_act, self.our_act, self.our_r, self.t], axis=1) # OBS: OPPO_ACT, OUR_ACT, OUR_R, T

    
    def step(self, action):
        #opponent choose jump/wait
        if np.random.random() < 1-self.epsilon:
            opponent_action = np.random.randint(2, size=(self.bs,))
        else:
            max_idx = self.innerq.reshape(self.innerq.shape[0],-1).argmax(1)
            opponent_action = np.column_stack(np.unravel_index(max_idx, self.innerq[0,:,:].shape))[:,1]
        
        oppo_idx = np.random.randint(len(self.choice), size=self.bs)
        oppo_advance = self.choice[np.arange(self.bs), oppo_idx]
        our_idx = np.random.randint(len(self.choice), size=self.bs)
        our_advance = self.choice[np.arange(self.bs), our_idx]
        
        # PLAY GAME, GET REWARDS
        for i in range(self.bs):
            if self.done[i] == 0:            #if no agent shouted done
                #step for opponent
                if (opponent_action[i] == 0) and (self.oppo_r[i] + oppo_advance[i] > 1):    #if wait & crosses 1 before jump
                    self.oppo_r[i] = 0
                    self.done[i] = 1                                                         #fking over

                elif (opponent_action[i] == 0) and (self.oppo_r[i] + oppo_advance[i] < 1):   #if wait & still within 1
                    self.oppo_r[i] += oppo_advance[i]

                elif opponent_action[i] == 1:                                         #if jump and crosses
                    self.oppo_r[i] += oppo_advance[i]
                    self.done[i] = 1                                                         #fking over


                #step for us
                if (action[i] == 0) and (self.our_r[i] + our_advance[i] > 1):
                    self.our_r[i] = 0
                    self.done[i] = 1                                                         #fking over

                elif (action[i] == 0) and (self.our_r[i] + our_advance[i] < 1):
                    self.our_r[i] += our_advance[i]

                elif action[i] == 1:
                    self.our_r[i] += our_advance[i]
                    self.done[i] = 1                                                         #fking over
        
            #else for this batch it's done, it retains the previous rewards
        self.t += 1
        self.done = np.logical_or(self.t >= self.interval, self.done)
        
        # GENERATE OBSERVATION
        observation = np.stack([opponent_action, action, self.our_r, self.t], axis=1) 

        # UPDATE OPPONENT POLICY
        self.innerq[range(self.bs), opponent_action] = self.lr * self.oppo_r + (1 - self.lr) * self.innerq[range(self.bs), opponent_action]

        return observation, self.our_r, self.oppo_r, self.done
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
       
