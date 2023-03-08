import numpy as np
import random
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
    #meta-s = [discretized everyone's inner q, our action, t]   meta-a = our inner q
    def __init__(self, bs, radius):
        self.epsilon = 0.8
        self.lr = 0.75
        #reward table with discretized dimensions, (actions, agents) (no states since num of state =1)
        self.innerq = [0, 0]
        self.t = 0
        self.bs = bs
        self.radius= radius

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
    
class MetaGamesSimplest:
    #meta-s = [oppo_act, our_act, t], meta-a = our_act
    def __init__(self, bs):
        self.bs = bs
        self.epsilon = 0.8
        self.lr = 1.0
        self.max_steps = 5
        #reward table with discretized dimensions, (actions, agents) (no states since num of state =1)

    def reset(self):
        #Initialise inner policy randomly
        temp_inner_policy = np.random.randint(2, size=(self.bs,))
        self.init_action = np.random.randint(2, size=(self.bs,))
        self.inner_policy = 1 - self.init_action
        self.t = np.ones(self.bs) * -1  #for zero-indexing
        return np.stack([temp_inner_policy, self.init_action, self.t], axis=1) # OBS: INNER_ACT, ACTION, TIMESTEP

    def step(self, action):
        opponent_action = self.inner_policy

        # PLAY GAME, GET REWARDS
        r1 = (opponent_action == action)*1
        r2 = 1 - r1
        self.t += 1

        # GENERATE OBSERVATION
        observation = np.stack([self.inner_policy, action, self.t], axis=1) 

        # UPDATE OPPONENT POLICY
        self.inner_policy = 1 - action

        # CHECK DONE
        done = self.t >= self.max_steps

        return observation, r1, done, {"r1": r1, "r2": r2}
    
       
