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
    #meta-s = [everyone' aciton history up to timestep hist_step, t]   meta-a = our inner q
    def __init__(self, bs, hist_step, meta_steps):
        self.epsilon = 0.8
        self.lr = 0.9
        self.t = 0
        self.bs = bs
        self.hist_step = hist_step
        self.meta_steps = meta_steps
        
        self.innerq = np.zeros((self.bs,2))
        self.oppo_deque = deque(np.zeros((self.hist_step, self.bs)), maxlen=self.hist_step)
        self.our_deque = deque(np.zeros((self.hist_step, self.bs)), maxlen=self.hist_step)

    def reset(self):
        #Initialise inner Q table randomly
        self.innerq = np.zeros((self.bs,2))
        
#         self.our_deque.append(np.random.randint(2, size=(self.bs)))
#         self.oppo_deque.append(np.random.randint(2, size=(self.bs)))
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

        # PLAY GAME, GET REWARDS
        r1 = (opponent_action == action) * 1
        r2 = 1 - r1
        self.t += 1

        # GENERATE OBSERVATION
        self.oppo_deque.append(opponent_action)
        self.our_deque.append(action)
        observation = np.stack([self.oppo_deque[0], self.our_deque[0], self.t], axis=1)

        # UPDATE OPPONENT Q-TABLE
        self.innerq[range(self.bs), opponent_action] = self.lr * r2 + (1 - self.lr) * self.innerq[range(self.bs), opponent_action]

        # CHECK DONE
        done = self.t >= self.meta_steps

        return observation, r1, done, {"r1": r1, "r2": r2}
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
       
