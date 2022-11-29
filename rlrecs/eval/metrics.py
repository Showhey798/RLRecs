import jax
from jax import numpy as jnp


def recall_at_k(
        y_trues: jnp.ndarray, # (batch_size, )
        y_preds: jnp.ndarray, # (batch_size, k)
        k : int=10
):
    @jax.vmap
    def recall(
        y_trues: jnp.ndarray, # (1, )
        y_preds: jnp.ndarray, # (k, )
    ):
        y_preds = y_preds[:k]
        return jnp.sum(y_preds == y_trues)
    return jax.device_get(jnp.mean(recall(y_trues, y_preds)))

def ndcg_at_k(
    y_trues: jnp.ndarray, 
    y_preds: jnp.ndarray,
    k: int=10
):
    @jax.vmap
    def ndcg(
        y_trues: jnp.ndarray,
        y_preds: jnp.ndarray
    ):
        dcg_score = jnp.where(y_trues == y_preds[:k])[0]
        
        return jnp.where(
            dcg_score.shape[0] == 0, 
            0., 
            1. / jnp.log((dcg_score + 2))
        )

    return jnp.mean(ndcg(y_trues, y_preds))