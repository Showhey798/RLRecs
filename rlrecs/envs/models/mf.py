from typing import Dict, Optional
from collections import OrderedDict
from functools import partial
import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random, jit


def init_params( 
    num_users:int, 
    num_items:int, 
    embed_dim:int,
    key:random.PRNGKey
)->Dict[str, jnp.ndarray]:
    
    key, rng1, rng2, rng3, rng4 = random.split(key, 5)
    
    params = {
        "user_embedding": random.truncated_normal(rng1, lower=0., upper=3., shape=(num_users, embed_dim)),
        "item_embedding": random.truncated_normal(rng2, lower=0., upper=3., shape=(num_items, embed_dim)),
        "user_bias" : random.truncated_normal(rng3, lower=0., upper=3., shape=(num_users, )),
        "item_bias" : random.truncated_normal(rng3, lower=0., upper=3., shape=(num_items, ))
    }
    
    return params

@jax.jit
def update(
    params:Dict[str, jnp.ndarray],
    data:np.ndarray,
    alpha:float,
    lam:float
)->Dict[str, jnp.ndarray]:
    #data = common_utils.shard(data)
    user, item, rating = data[:, 0], data[:, 1], data[:, 2]

    
    u_emb = params.get("user_embedding")[user] # (batch_size, embed_dim)
    i_emb = params.get("item_embedding")[item] # (batch_size, embed_dim)
    
    u_bias = params.get("user_bias")[user] # (batch_size, )
    i_bias = params.get("item_bias")[item] #(batch_size, )

    preds = (u_emb * i_emb).sum(axis=1) + u_bias + i_bias
    loss = jnp.square(rating - preds)
    error = (rating - preds).reshape(len(u_emb), 1)

    params["user_embedding"] = params["user_embedding"].at[user].set(u_emb + alpha * (2 * error * i_emb - lam * u_emb))
    params["item_embedding"] = params["item_embedding"].at[item].set(i_emb + alpha * (2 * error * u_emb - lam * i_emb))
    
    params["user_bias"] = params["user_bias"].at[user].set(
        u_bias + alpha * 2 * (jnp.squeeze(error) - lam * u_bias))
    params["item_bias"] = params["item_bias"].at[item].set(
        i_bias + alpha * 2 * (jnp.squeeze(error) - lam * i_bias))
    
    return params, jnp.mean(loss)

@partial(jax.jit, static_argnums=(2,))
def recommend(
    params:Dict[str, jnp.ndarray],
    u:int,
    k:Optional[int]=10
):
    u_emb = params.get("user_embedding")[u] #(embed_dim,)
    i_emb = params.get("item_embedding")    # (num_items, emebd_dim)
    u_bias = params.get("user_bias")[u] # (batch_size, )
    i_bias = params.get("item_bias")[i] #(batch_size, )

    pred = (u_emb * i_emb).sum(axis=1) + u_bias + i_bias
    
    pred_ids = jnp.argsort(pred)[::-1][:k]
    
    return pred_ids

@jax.jit
def predict(
    params : Dict[str, jnp.ndarray],
    u:jnp.ndarray,
    i:jnp.ndarray,
    r:jnp.ndarray
):
    u_emb = params.get("user_embedding")[u]
    i_emb = params.get("item_embedding")[i]
    
    u_bias = params.get("user_bias")[u] # (batch_size, )
    i_bias = params.get("item_bias")[i] #(batch_size, )

    preds = (u_emb * i_emb).sum(axis=1) + u_bias + i_bias
    
    loss = jnp.square(r - preds)
    return jnp.mean(loss)