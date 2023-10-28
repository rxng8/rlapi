import numpy as np


class RandomAgent:
  def __init__(self, act_space):
    self.act_space = act_space

  def policy(self, obs, state=None, mode='train'):
    act = {k: v.sample()[None, ...] for k, v in self.act_space.items() if k != 'reset'}
    return act, state
