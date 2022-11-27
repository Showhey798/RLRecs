from typing import Optional
from flax import linen as nn
from jax import numpy as jnp
import jax

class GRUState(nn.Module):
    num_items: int
    num_reward_type: Optional[int]=2
    embed_dim: Optional[int]=100
    hidden_dim : Optional[int]=200
    @nn.compact
    def __call__(
        self, 
        q, # (batch_size, seq_len), アイテム
        f, # (batch_size, seq_len) フィードバック
    ):
        q = nn.Embed(self.num_items, self.embed_dim)(q)
        f = nn.Embed(self.num_reward_type, self.embed_dim)(f)
        
        carry = self.param("init_carry", lambda rng, shape: jnp.zeros(shape), self.hidden_dim)
        
        state = q * f # (batch_size, seq_len, embed_dim)
        
        for i in range(q.shape[1]):
            carry, _ = nn.GRUCell()(carry, state[:, i, :])
        
        output = nn.Dense(self.num_items)(carry)
        return output