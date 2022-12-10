from typing import Dict, Optional, Any, Tuple
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import optax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict

from rlrecs.agents.models.state_encoder import gru
from rlrecs.agents.models.base_agent import BaseAgent
from rlrecs.agents.models.common import Model

def sparse_softmax_cross_entoropy_with_logits(target_y, pred_y):
    @jax.vmap
    def log_loss(y, probs):
        y = jax.nn.one_hot(y, pred_y.shape[1])
        return -jnp.sum(y*nn.log_softmax(probs))
    return jnp.mean(log_loss(target_y, pred_y))

@partial(jax.jit, static_argnums=(10,))
def update(
    model:Model,
    #---データセット---#
    state: jnp.ndarray,
    action: jnp.ndarray
):
    def loss_fn(params):
        out = model.apply_fn({"params": params}, state)
        loss = sparse_softmax_cross_entoropy_with_logits(action, out)
        return loss, loss
    model, info = model.apply_gradient(loss_fn)
    return model, info

@jax.jit
def greedy_action(
    model : Model, 
    state:jnp.ndarray, 
    click_masks:jnp.ndarray, # (num_users, num_items)
):
    qvalue = model(state) # (batch_size, num_items)
    qvalue += (-1e10*click_masks)
    recommend_items = jnp.argsort(qvalue, axis=1)
    return recommend_items

class GRU4Rec(BaseAgent):
    def __init__(
        self, 
        num_items:int,
        hidden_dim:int,
        seq_len: int,
        embed_dim: int,
        learning_rate:float
    ):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.learning_rate = learning_rate        
    
    def init_params(        
        self, 
        batch_size: int,
        key
    ):
        module = gru.NormalGRU(self.num_items, embed_dim=self.embed_dim, hidden_dim=self.hidden_dim)
        
        tx = optax.adam(learning_rate=self.learning_rate)
        
        rng, key1 = jax.random.split(key, 2)
        self.model = Model.create(
            module, 
            inputs=[
                key1,
                jnp.ones((batch_size, self.seq_len), dtype=jnp.int32)
            ],
            tx=tx
        )
        
    
    def train_step(self, data):
        state, _, action, _, _, _, _ = data
        action -=  1
        self.model, loss = update(
            self.model,
            state,
            action
        )
        return loss
    
    def recommend(self, inputs:jnp.ndarray, click_masks:Optional[np.ndarray]=None, is_greedy:Optional[bool]=True):
        state = inputs
        if click_masks is None:
            click_masks = np.identity(self.num_items)[state].sum(axis=1)
        actions = greedy_action(self.model, state, click_masks)
        return actions
    