from flax import linen as nn
import flax
import optax
from flax.training import train_state, common_utils

import jax
from jax import numpy as jnp

class GRUState(nn.Module):
    @nn.compact
    def __call__(
        self, 
        input_seqs, # (batch_size, seq_len)
    ):
        x = nn.GRUCell()
        

