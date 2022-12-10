from typing import Optional
from flax import linen as nn
from jax import numpy as jnp
import jax

class GRUState(nn.Module):
    num_items: int
    num_reward_type: Optional[int]=2
    embed_dim: Optional[int]=100
    hidden_dim : Optional[int]=200
    dropout_rate: Optional[float]=0.1
    @nn.compact
    def __call__(
        self, 
        q:jnp.ndarray, # (batch_size, seq_len), アイテム
        f:jnp.ndarray, # (batch_size, seq_len) フィードバック
        deterministic:Optional[bool]=False,
    ):
        q = nn.Embed(self.num_items, self.embed_dim)(q)
        f = nn.Embed(self.num_reward_type, self.embed_dim)(f)
        
        carry = self.param("init_carry", lambda rng, shape: jnp.zeros(shape), self.hidden_dim)
        
        state = q * f # (batch_size, seq_len, embed_dim)
        
        for i in range(q.shape[1]):
            carry, _ = nn.GRUCell()(carry, state[:, i, :])
        
        output = nn.Dense(self.num_items)(carry)
        return output
    
class NormalGRU(nn.Module):
    num_items: int
    embed_dim: Optional[int]=100
    hidden_dim : Optional[int]=200
    @nn.compact
    def __call__(
        self, 
        q:jnp.ndarray, # (batch_size, seq_len), アイテム
    ):
        not_zero = jnp.asarray((q != 0), dtype=jnp.float32) # (batch_size, seq_len)
        not_zero = jnp.expand_dims(not_zero, axis=-1)
        q = nn.Embed(self.num_items, self.embed_dim)(q)
        q *= not_zero
        
        carry = jnp.zeros((q.shape[0], self.hidden_dim))
        
        for i in range(q.shape[1]):
            carry, out = nn.GRUCell()(carry, q[:, i, :])
        
        output = nn.Dense(self.num_items)(out)
        return output