# %%

import importlib
import pathlib
import sys
import warnings
import functools
from functools import partial as bind
import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# expert learning
import gymnasium as gym

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory))
sys.path.append(str(directory.parent))

import embodied
from embodied import wrappers
import ruamel.yaml as yaml
from embodied.core import Driver


############################# TRAIN ###############################

def make_env(config, **overrides):
  # You can add custom environments by creating and returning the environment
  # instance here. Environments with different interfaces can be converted
  # using `embodied.envs.from_gym.FromGym` and `embodied.envs.from_dm.FromDM`.
  suite, task = config.task.split('_', 1)
  ctor = {
    'dummy': 'embodied.envs.dummy:Dummy',
    'gym': 'embodied.envs.gym:GymEnv',
    'dm': 'embodied.envs.dm:DMEnv',
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  env = ctor(task, **kwargs)
  return wrap_env(env, config)

def wrap_env(env, config):
  args = config.wrapper
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    elif space.discrete:
      env = wrappers.OneHotAction(env, name)
    elif args.discretize:
      env = wrappers.DiscretizeAction(env, name, args.discretize)
    else:
      env = wrappers.NormalizeAction(env, name)
  if args.repeat > 1:
    env = wrappers.ActionRepeat(env, args.repeat)
  env = wrappers.ExpandScalars(env)
  # env = wrappers.ExpandBatchDimSoft(env)
  if args.delta:
    env = wrappers.TransitionDelta(env, args.delta)
  if args.length:
    env = wrappers.TimeLimit(env, args.length, args.reset)
  if args.checks:
    env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  return env

configs = yaml.YAML(typ='safe').load((embodied.Path(__file__).parent / 'configs.yaml').read())
# Whether we are doing interactive or not
if embodied.check_vscode_interactive():
  _args = [
    "--expname=test5",
    "--configs=gym_mtc"
  ]
  parsed, other = embodied.Flags(configs=['defaults']).parse_known(_args)
else:
  parsed, other = embodied.Flags(configs=['defaults']).parse_known(sys.argv[1:])

# Preping and parsing all configs and overrides
config = embodied.Config(configs['defaults'])
for name in parsed.configs:
  config = config.update(configs[name])
config = embodied.Flags(config).parse(other)
# Create path and necessary folders, Setup more path for the config
# logdir initialization
logdir = embodied.Path(config.logroot) / config.expname
logdir.mkdirs()
config = config.update({"logdir": str(logdir)})
# datadir initialization
datadir = logdir / "data"
datadir.mkdirs()
config = config.update({"datadir": str(datadir)})
# tboarddir initialization
tboarddir = embodied.Path(config.logroot) / "tboard" / config.expname
tboarddir.mkdirs()
config = config.update({"tboarddir": str(tboarddir)})
# DONE preparing config. Save config
config.save(logdir / 'config.yaml')
print(config, '\n')
print('Logdir', logdir)

# %%

env = make_env(config, reward_shaping=True, freeze_when_done=False)

# Ì you want it to be gym like, wrap the gym as final layer. Otherwise, comment the following line
env = embodied.wrappers.GymWrapperFinalLayer(env, obs_key="state")

# test gym env
obs, info = env.reset()
for i in range(100):
  next_obs, reward, done, truncated, info = env.step(env.action_space.sample())
  # print
  print(f"Step {i}, next_obs: {next_obs}")
  # preping for next step
  obs = next_obs


