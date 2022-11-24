from typing import Dict, Optional, Tuple
from collections import OrderedDict
from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
from jax import random, jit


def init_params( 
    num_users:int, 
    num_items:int, 
    key:random.PRNGKey
)->Dict[str, jnp.ndarray]:
    
    key, rng1, rng2 = random.split(key, 3)
    
    params = {
        "w": random.truncated_normal(rng1, lower=0., upper=1., shape=(num_users+num_items, )),
        "b" : random.truncated_normal(rng1, lower=0., upper=1., shape=(1, )),
        }
    
    return params


def batch_log_loss(w, b, x, y, lam):
    
    @jax.vmap
    def log_loss(x, y):
        preds = jax.nn.sigmoid(x@w + b)
        
        loss = -y*jnp.log(preds) - (1-y)*jnp.log(1 - preds)  + lam * (jnp.sum(w**2) + b**2)
        return loss

    return jnp.mean(log_loss(x, y))


@jax.jit
def update(
    params:Dict[str, jnp.ndarray],
    data:Tuple[jnp.ndarray],
    alpha:float,
    lam:float
):
    x, y = data[0], data[1]
    loss = batch_log_loss(params["w"], params["b"], x, y, lam)
    w_grad = jax.grad(lambda w: batch_log_loss(w, params["b"], x, y, lam))(params["w"])
    b_grad = jax.grad(lambda b: batch_log_loss(params["w"], b, x, y, lam))(params["b"])
    params["w"] -= alpha * w_grad
    params["b"] -= alpha * b_grad
    preds = predict(params, x)
    return params, loss, preds

@jax.jit
def predict(
    params:Dict[str, jnp.ndarray],
    data: jnp.ndarray
):
    w = params["w"]
    b = params["b"]
    @jax.vmap
    def probs(x):
        return jax.nn.sigmoid(x@w + b)
    
    prob = probs(data)
    return prob