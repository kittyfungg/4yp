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
    def __init__(self):
        self.epsilon = 0.8
        self.lr = 0.75
        #reward table with discretized dimensions, (actions, agents) (no states since num of state =1)
        self.innerq = [0, 0]
        self.t = 0

    def reset(self):
        #Initialise inner Q table randomly
        self.innerq = np.random.random(shape=(2,))
        self.init_action = np.random.randint(0,2)
        self.t = 0
        return np.concatenate([self.discretise_q(self.innerq), self.init_action, self.t], axis=0) # OBS: INNERQ, ACTION, TIMESTEP

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

    def step(self, action):

        opponent_action = self.select_action()

        # PLAY GAME, GET REWARDS
        r1 = int(opponent_action == action)
        r2 = 1 - r1
        self.t += 1

        # GENERATE OBSERVATION
        observation = np.concatenate([
            self.discretise_q(self.innerq),
            action,
            self.t
        ], axis=0) # MAKE SURE SHAPES LINE UP

        # UPDATE OPPONENT Q-TABLE
        self.innerq[opponent_action] = self.lr * r2 + (1 - self.lr) * self.innerq[opponent_action]

        # CHECK DONE
        done = self.t > 10

        return observation, r1, done, {"r1": r1, "r2": r2}
    
class MetaGamesSimplest:
    def __init__(self):
        self.epsilon = 0.8
        self.lr = 1.0
        #reward table with discretized dimensions, (actions, agents) (no states since num of state =1)

    def reset(self):
        #Initialise inner policy randomly
        self.inner_policy = np.random.randint(0, 2)
        self.init_action = np.random.randint(0,2)
        self.t = 0
        return np.concatenate([self.inner_policy, self.init_action, self.t], axis=0) # OBS: INNER_ACT, ACTION, TIMESTEP

    def step(self, action):
        opponent_action = self.inner_policy

        # PLAY GAME, GET REWARDS
        r1 = int(opponent_action == action)
        r2 = 1 - r1
        self.t += 1

        # GENERATE OBSERVATION
        observation = np.concatenate([
            self.inner_policy,
            action,
            self.t
        ], axis=0) # MAKE SURE SHAPES LINE UP

        # UPDATE OPPONENT POLICY
        self.inner_policy = 1 - action

        # CHECK DONE
        done = self.t > 10

        return observation, r1, done, {"r1": r1, "r2": r2}
    
       
