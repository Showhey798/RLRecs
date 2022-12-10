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

def td_loss(target_q, q, k):
    @jax.vmap
    def huber_loss(y, y_hat):
        loss = jnp.abs(y - y_hat)

        return jnp.where(
            loss < k, 
            0.5*jnp.square(loss),
            k * (loss - 0.5*k))
    return jnp.mean(huber_loss(target_q, q))

@partial(jax.jit, static_argnums=(10,))
def update(
    model:Model,
    target_model:Model,
    #---データセット---#
    state: jnp.ndarray,
    feedback:jnp.ndarray,
    action: jnp.ndarray,
    n_state: jnp.ndarray,
    n_feedback:jnp.ndarray,
    reward: jnp.ndarray,
    done:jnp.ndarray,
    gamma:float,
    num_items:int
):
    action -= 1
    def loss_fn(params):
        tar_qvalue = target_model(n_state, n_feedback)
        qvalue = model.apply_fn({"params": params}, state, feedback)
        n_action_qvalue = jnp.max(tar_qvalue, axis=1)
        tar_qvalue = reward + (1 - done) * gamma*n_action_qvalue 
        action_onehot = jax.nn.one_hot(action, num_items)
        qvalue = jnp.sum(qvalue * action_onehot, axis=1)
        loss = td_loss(tar_qvalue, qvalue, 1.0)
        return loss, loss
    
    model, info = model.apply_gradient(loss_fn)

    return model, info

@jax.jit
def greedy_action(
    model : Model, 
    state:jnp.ndarray, 
    feedback:jnp.ndarray,
    click_masks:jnp.ndarray, # (num_users, num_items)
):
    qvalue = model(state, feedback) # (batch_size, num_items)
    qvalue += (-1e10*click_masks) 
    recommend_items = jnp.argsort(qvalue, axis=1)
    return recommend_items

class DQN(BaseAgent):
    """
    DQNのエージェント
    Args:
        num_items (int): 全アイテム数
        hidden_dim (int): 隠れ層の次元数
        seq_len (int): 1状態が持つアイテム履歴の長さ
        embed_dim (int): 各アイテムの埋め込み次元数
        learning_rate (float): 学習率
        gamma (float): 割引率
    """
    def __init__(
        self, 
        num_items:int,
        hidden_dim:int,
        seq_len: int,
        embed_dim: int,
        learning_rate:float,
        gamma:float,
    ):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        self.interaction_count = 0
        self.epsilon = 0.8
    
    def init_epsilon(self):
        self.epsilon = 0.8
        
    
    def init_params(        
        self, 
        batch_size: int,
        key
    ):
        self.update_count = 0
        module = gru.GRUState(self.num_items, embed_dim=self.embed_dim, hidden_dim=self.hidden_dim)
        
        tx = optax.adam(learning_rate=self.learning_rate)
        
        rng, key1, key2 = jax.random.split(key, 3)
        self.model = Model.create(
            module, 
            inputs=[
                key1,
                jnp.ones((batch_size, self.seq_len), dtype=jnp.int32), 
                jnp.ones((batch_size, self.seq_len), dtype=jnp.int32)
            ],
            tx=tx
        )
        self.target_model = Model.create(
            module, 
            inputs=[
                key2,
                jnp.ones((batch_size, self.seq_len), dtype=jnp.int32), 
                jnp.ones((batch_size, self.seq_len), dtype=jnp.int32)
            ])
        
        self.target_model.replace(params=self.model.params)
        
    
    def train_step(self, data):
        state, feedback, action, n_state, n_feedback, reward, done = data
        self.model, loss = update(
            self.model,
            self.target_model,
            state,
            feedback,
            action,
            n_state,
            n_feedback,
            reward,
            done,
            self.gamma,
            self.num_items
        )
        return loss
    
    def recommend(self, inputs:Tuple[np.ndarray], click_masks:Optional[np.ndarray]=None, is_greedy:Optional[bool]=True, k:Optional[int]=1):
        state, feedback = inputs
        epsilon = 0 if is_greedy else self.epsilon
        if click_masks is None:
            click_masks = np.identity(self.num_items)[state].sum(axis=1)
        if np.random.uniform() < epsilon:
            actions = np.random.randint(self.num_items, size=state.shape[0])
        else:
            actions = greedy_action(self.model, state, feedback, click_masks)
            actions = jax.device_get(actions)
            actions = actions[:, :k]
        
        self.interaction_count += 1
        self.epsilon = np.max([self.epsilon - 0.1, 0.1]) if self.interaction_count%200000==0 else self.epsilon
        return actions
    