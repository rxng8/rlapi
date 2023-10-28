import collections
import numpy as np
from .basics import convert

class Driver:
  _CONVERSION = {
    np.floating: np.float32,
    np.signedinteger: np.int32,
    np.uint8: np.uint8,
    bool: bool,
  }

  def __init__(self, env, **kwargs):
    self._env = env
    self._kwargs = kwargs
    self._on_steps = []
    self._on_episodes = []
    self.reset()

  def reset(self):
    self._acts = {
      k: convert(np.zeros(v.shape, v.dtype))
      for k, v in self._env.act_space.items()
    }
    self._acts['reset'] = np.ones((), bool)
    self._eps = collections.defaultdict(list)
    self._state = None

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    # preprocess previous action
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
    # step in the environment
    obs = self._env.step(acts)
    # preprocess obs: convert
    obs = {k: convert(v) for k, v in obs.items()}
    # execute policy for taking action to be step in the 
    # next mdp step, note that in here the agent will 
    # take in the batch dim expanded obs instead obs
    bacth_dim_obs = {k: v[None, ...] for k, v in obs.items()}
    acts, self._state = policy(bacth_dim_obs, self._state, **self._kwargs)
    # convert back and squeeze to the non dim to take action in the environment
    try:
      acts = {k: np.squeeze(convert(v), 0) for k, v in acts.items()}
    except:
      # TODO: Bug here: for the action space of 1, the policy of dreamerv3
      # autommatically vanish
      acts = {k: convert(v) for k, v in acts.items()}
      print(f"Abnormal behavior: {step}: action shape: {acts['action'].shape}")

    # mask action with zero and set the reset action to True when it is the last mdp step
    if obs['is_last']:
      acts = {k: np.zeros_like(v).astype(v.dtype) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()
    # Set the action for the next mdp step to be executed
    self._acts = acts
    # transition 
    trn = {**obs, **acts}
    # clear tmp episode when the environment is reseted
    if obs["is_first"]:
      self._eps.clear()
    # append each key value of the transition to the episode 
    [self._eps[k].append(v) for k, v in trn.items()]
    # callback functions every steps
    [fn(trn, **self._kwargs) for fn in self._on_steps]
    # increase step
    step += 1
    # if this is the alst step, then callback on the episode end
    if obs['is_last']:
      # convert list of np array to a whole numpy array using (i.e., stacking) and then convert dtype
      ep = {k: convert(v) for k, v in self._eps.items()}
      [fn(ep.copy(), **self._kwargs) for fn in self._on_episodes]
      # increase episode
      episode += 1
    return step, episode

  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value
