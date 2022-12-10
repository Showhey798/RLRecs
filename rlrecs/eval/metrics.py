import jax
from jax import numpy as jnp
from functools import partial

partial(jax.jit, static_argnums=(2,))
def recall_at_k(
        y_trues: jnp.ndarray, # (batch_size, )
        y_preds: jnp.ndarray, # (batch_size, k)
        k : int=10
):
    y_trues = jnp.expand_dims(y_trues, axis=-1)
    recall = (y_preds[:, :k] == y_trues)
    recall = jnp.sum(recall, axis=-1)
    return recall

partial(jax.jit, static_argnums=(2,))
def ndcg_at_k(
    y_trues: jnp.ndarray, 
    y_preds: jnp.ndarray,
    k: int=10
):

    dcg_score = jnp.where(y_trues == y_preds[:k])[0]    
    score = jnp.where(
        dcg_score.shape[0] == 0, 
        0., 
        1. / jnp.log((dcg_score + 2)))
    return jnp.sum(score, axis=-1)