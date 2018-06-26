import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class ChainEnv(gym.Env):
    """
    Chain as http://papers.nips.cc/paper/6501-deep-exploration-via-bootstrapped-dqn.pdf

    """
    def __init__(self, n=10, small_reward = 0.001, big_reward = 1, init_state =1):
        self.init_state = init_state
        self.big_reward = big_reward
        self.small_reward = small_reward
        self.n = n
        self.state = init_state  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        low = np.zeros(n)
        high = np.ones(n)
        self.observation_space = spaces.Box(low, high)
        self.steps = 1


    def step(self, action):
        assert self.action_space.contains(action)
        if action:
            if self.state + 1 == self.n:
                reward = self.big_reward
            elif self.state + 2 == self.n:
                reward = self.big_reward
                self.state += 1
            else:
                self.state += 1
                reward = 0
        else:
            if self.state == 0:
                reward = self.small_reward
            elif self.state == 1:
                reward = self.small_reward
                self.state -= 1
            else:
                self.state -= 1
                reward = 0
        if self.steps == self.n + 9:
            done = True
        else:
            done = False
            self.steps += 1

        obs = self._get_state()
        return obs, reward, done, {}

    def reset(self):
        self.state = self.init_state
        self.steps = 1
        return self._get_state()

    def _get_state(self):
        """Get the observation."""
        obs = np.zeros(self.n)
        obs[0:self.state+1] = 1
        return obs
