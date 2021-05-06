  
import gym
from rlpyt.envs.gym import EnvInfoWrapper, info_to_nt
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.envs.base import EnvSpaces, EnvStep
import procgen
import cv2, os
import numpy as np
from typing import Any, Dict, List, Sequence, Tuple
from gym.spaces import Box, Dict, Discrete as DiscreteG

from rlpyt.samplers.collections import TrajInfo

from procgen import ProcgenGym3Env
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecFrameStack,
    VecEnvWrapper
)

from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit

from collections import namedtuple
from rlpyt.utils.collections import is_namedtuple_class

class GymEnvWrapper(Wrapper):
    """Gym-style wrapper for converting the Openai Gym interface to the
    rlpyt interface.  Action and observation spaces are wrapped by rlpyt's
    ``GymSpaceWrapper``.
    Output `env_info` is automatically converted from a dictionary to a
    corresponding namedtuple, which the rlpyt sampler expects.  For this to
    work, every key that might appear in the gym environments `env_info` at
    any step must appear at the first step after a reset, as the `env_info`
    entries will have sampler memory pre-allocated for them (so they also
    cannot change dtype or shape).  (see `EnvInfoWrapper`, `build_info_tuples`,
    and `info_to_nt` in file or more help/details)
    Warning:
        Unrecognized keys in `env_info` appearing later during use will be
        silently ignored.
    This wrapper looks for gym's ``TimeLimit`` env wrapper to
    see whether to add the field ``timeout`` to env info.   
    """

    def __init__(self, env,
            act_null_value=0, obs_null_value=0, force_float32=True):
        super().__init__(env)
        o = self.env.reset()
        o, r, d, info = self.env.step(self.env.action_space.sample())
        env_ = self.env
        time_limit = isinstance(self.env, TimeLimit)
        while not time_limit and hasattr(env_, "env"):
            env_ = env_.env
            time_limit = isinstance(env_, TimeLimit)
        if time_limit:
            info["timeout"] = False  # gym's TimeLimit.truncated invalid name.
        self._time_limit = time_limit
        self.action_space = GymSpaceWrapper(
            space=self.env.action_space,
            name="act",
            null_value=act_null_value,
            force_float32=force_float32,
        )
        self.observation_space = GymSpaceWrapper(
            space=self.env.observation_space,
            name="obs",
            null_value=obs_null_value,
            force_float32=force_float32,
        )
        build_info_tuples(info)

    def step(self, action):
        """Reverts the action from rlpyt format to gym format (i.e. if composite-to-
        dictionary spaces), steps the gym environment, converts the observation
        from gym to rlpyt format (i.e. if dict-to-composite), and converts the
        env_info from dictionary into namedtuple."""
        a = self.action_space.revert(action)
        o, r, d, info = self.env.step(a)
        obs = self.observation_space.convert(o)
        if self._time_limit:
            if "TimeLimit.truncated" in info:
                info["timeout"] = info.pop("TimeLimit.truncated")
            else:
                info["timeout"] = False
        info = info_to_nt(info)
        
        if isinstance(r, float):
            r = np.dtype("float32").type(r)  # Scalar float32.
        return EnvStep(obs, r, d, info)

    def reset(self):
        """Returns converted observation from gym env reset."""
        return self.observation_space.convert(self.env.reset())

    @property
    def spaces(self):
        """Returns the rlpyt spaces for the wrapped env."""
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )

def build_info_tuples(info, name="info"):
    # Define namedtuples at module level for pickle.
    # Only place rlpyt uses pickle is in the sampler, when getting the
    # first examples, to avoid MKL threading issues...can probably turn
    # that off, (look for subprocess=True --> False), and then might
    # be able to define these directly within the class.
    ntc = globals().get(name)  # Define at module level for pickle.
    info_keys = [str(k).replace(".", "_") for k in info.keys()]
    info_keys.append('rew')
    if ntc is None:
        globals()[name] = namedtuple(name, info_keys)
    elif not (is_namedtuple_class(ntc) and
            sorted(ntc._fields) == sorted(info_keys)):
        raise ValueError(f"Name clash in globals: {name}.")
    for k, v in info.items():
        if isinstance(v, dict):
            build_info_tuples(v, "_".join([name, k]))

def info_to_nt(value, name="info"):
    if not isinstance(value, dict):
        return value
    ntc = globals()[name]
    # Disregard unrecognized keys:
    values = {k: info_to_nt(v, "_".join([name, k]))
        for k, v in value.items() if k in ntc._fields}
    # Can catch some missing values (doesn't nest):
    values.update({k: 0 for k in ntc._fields if k not in values})
    return ntc(**values)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(num_colors, self._height, self._width),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        obs = obs.astype(np.uint8)
        if len(obs.shape) == 4:
            obs = obs[0]
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        obs = obs.transpose(2,0,1)
        return obs

class GymEnvWrapperFixed(GymEnvWrapper):
    def step(self, a):
        a = int(a.item())
        o, r, d, info = self.env.step(a)
        obs = self.observation_space.convert(o)
        if self._time_limit:
            if "TimeLimit.truncated" in info:
                info["timeout"] = info.pop("TimeLimit.truncated")
            else:
                info["timeout"] = False
                
        info = info_to_nt(info)
        try:
            info = info.coins
        except:
            pass
        return EnvStep(obs, r, d, info)

    def seed(self, seed):
        pass

class EpisodeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        env.metadata = {'render.modes': []}
        env.reward_range = (-float('inf'), float('inf'))
        nenvs = env.num_envs
        self.num_envs = nenvs
        super(EpisodeRewardWrapper, self).__init__(env)

        self.aux_rewards = None
        self.num_aux_rews = None

        def reset(**kwargs):
            self.rewards = np.zeros(nenvs)
            self.lengths = np.zeros(nenvs)
            self.aux_rewards = None
            self.long_aux_rewards = None

            return self.env.reset(**kwargs)

        def step(action):
            obs, rew, done, infos = self.env.step(action)

            if self.aux_rewards is None:
                # info = infos[0]
                info = infos
                if 'aux_rew' in info:
                    self.num_aux_rews = len(info['aux_rew'])
                else:
                    self.num_aux_rews = 0

                self.aux_rewards = np.zeros((nenvs, self.num_aux_rews), dtype=np.float32)
                self.long_aux_rewards = np.zeros((nenvs, self.num_aux_rews), dtype=np.float32)

            self.rewards += rew
            self.lengths += 1

            use_aux = self.num_aux_rews > 0

            if use_aux:
                for i, info in enumerate(infos):
                    self.aux_rewards[i,:] += info['aux_rew']
                    self.long_aux_rewards[i,:] += info['aux_rew']

            for i, d in enumerate(done):
                if d:
                    infos['r'] =  round(self.rewards[i], 6)

                    self.rewards[i] = 0
                    self.lengths[i] = 0
                    self.aux_rewards[i,:] = 0
                    
            return obs, rew, done, infos

        self.reset = reset
        self.step = step

class FakeEnv:
    """
    Create a baselines VecEnv environment from a gym3 environment.
    :param env: gym3 environment to adapt
    """

    def __init__(self, env, obs_space, act_space, num_envs):
        self.env = env
        # for some reason these two are returned as none
        # so I passed in a baselinevec object and copied
        # over it's values
        # self.baseline_vec = baselinevec
        self.observation_space = obs_space
        self.action_space = act_space
        self.rewards = np.zeros(num_envs)
        self.lengths = np.zeros(num_envs)
        self.aux_rewards = None
        self.long_aux_rewards = None

    def reset(self):
        _rew, ob, first = self.env.observe()
        if not first.all():
            print("Warning: manual reset ignored")
        return ob

    def step_async(self, ac):
        # if type(ac) == np.ndarray:
        #     ac = np.array([ac])
        # if type(ac) == int:
        ac = np.array([ac])
        # print(ac,type(ac))
        self.env.act(ac)

    def step_wait(self):
        rew, ob, first = self.env.observe()
        return ob, rew, first, self.env.get_info()[0]

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()

    @property
    def num_envs(self):
        return self.env.num

    def render(self, mode="human"):
        # gym3 does not have a generic render method but the convention
        # is for the info dict to contain an "rgb" entry which could contain
        # human or agent observations
        info = self.env.get_info()[0]
        if mode == "rgb_array" and "rgb" in info:
            return info["rgb"]

    def close(self):
        pass
    
    # added this in to see if it'll properly call the method for the gym3 object
    def callmethod(
        self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]
    ) -> List[Any]:
        return self.env.callmethod(method, *args, **kwargs)

def make_procgen_env(*args, info_example=None, **kwargs):
    import re
    num_levels = 500
    env_id = kwargs['game']

    env_maker  = lambda args: WarpFrame(
        gym.make(id="procgen:procgen-%s-v0"%(env_id),
                start_level=0,
                num_levels=int(num_levels),
                paint_vel_info=True,
                distribution_mode='easy',
                use_sequential_levels=False),
        width=64, height=64,grayscale=False)
    env = env_maker(None)
    return GymEnvWrapperFixed(env)

class ProcgenVecEnvCustom():
    def __init__(self,env_name,num_levels,mode,paint_vel_info=True,num_envs=32):
        self.observation_space_vec = Dict(rgb=Box(shape=(3,64,64),low=0,high=255))
        self.action_space_vec = DiscreteG(15)

        self.observation_space = Box(shape=(3,64,64),low=0,high=255)
        self.action_space = DiscreteG(15)

        self.action_space = GymSpaceWrapper(
            space=self.action_space,
            name="act",
            null_value=0,
            force_float32=True,
        )
        self.observation_space = GymSpaceWrapper(
            space=self.observation_space,
            name="obs",
            null_value=0,
            force_float32=True,
        )
        
        self.gym3_env_train = ProcgenGym3Env(num_levels=num_levels, num=num_envs, env_name=env_name, paint_vel_info=paint_vel_info, distribution_mode=mode)
        self.venv_train = FakeEnv(self.gym3_env_train, self.observation_space_vec, self.action_space_vec, num_envs)   
        self.venv_train = VecExtractDictObs(self.venv_train, "rgb")
        self.venv_train = VecNormalize(venv=self.venv_train, ob=False)
        self.env = GymEnvWrapper(EpisodeRewardWrapper(self.venv_train))
        self.spaces = EnvSpaces(observation=self.observation_space,action=self.action_space)

    def reset(self):
        return self.env.reset().transpose(0,3,1,2)
    
    def step(self,action):
        o, r, x, info, = self.env.step(action)
        return o.transpose(0,3,1,2), r[0], x[0], info

class ProcgenTrajInfo(TrajInfo):
    """TrajInfo class for use with Atari Env, to store raw game score separate
    from clipped reward signal."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.GameScore = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        rew = getattr(env_info, "r", 0)
        self.GameScore = rew