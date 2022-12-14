from typing import Dict, Optional, Callable, Any
from collections import OrderedDict
from tqdm import tqdm

import numpy as np

import jax
import jax.numpy as jnp
from jax import random

import flax
import optax
from flax import linen as nn
from flax import struct
from flax.core.frozen_dict import FrozenDict

from rlrecs.envs.dataset import DataLoader


class BaseAgent:

    def __init__(
        self, 
    ):
        pass
    
    def init_params(self, batch_size, key):
        raise NotImplementedError()
    
    def train_step(self, data):
        raise NotImplementedError()
    
    def recommend(self, data, **kwargs):
        raise NotImplementedError()

    def save(self, save_path:str):
        self.model.save(save_path)
    
    def load(self, load_path:str):
        self.model.load(load_path)
    