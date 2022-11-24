from typing import Dict, Optional
from collections import OrderedDict
from functools import partial
import numpy as np

np.random.seed(0)
import jax
import jax.numpy as jnp
from jax import random, jit

from rlrecs.envs.models import logistics
from tqdm import tqdm

class Prospensity():
    def __init__(self, num_items, num_users, key):
        self.num_items = num_items
        self.num_users = num_users
        self.params = logistics.init_params(num_users, num_items, key)
    def fit(
        self,
        x, #user, item pair
        y, # ratings
        batch_size=256,
        max_itr=10,
        alpha=0.01,
        lam=0.001
    ):
        with tqdm(range(max_itr), desc="training prospensity") as ts:
            for itr in ts:
                index = np.arange(len(x))
                np.random.shuffle(index)
                num_batches = int(index.shape[0] // batch_size)
                loss_hist = []
                for batch in range(num_batches):
                    batch_idx = index[batch*batch_size: (batch+1)* batch_size]
                    user = jax.nn.one_hot(x[batch_idx, 0], num_classes=self.num_users)
                    item = jax.nn.one_hot(x[batch_idx, 1], num_classes=self.num_items)
                    batch_x = jnp.concatenate([user, item], axis=1)
                    self.params, loss, p = logistics.update(self.params, (batch_x ,y[batch_idx]), alpha, lam)
                    print(jax.device_get(p))
                    loss = jax.device_get(loss)
                    loss_hist += [loss]
                ts.set_postfix(OrderedDict(loss=np.mean(loss_hist)))
    
    def get_prospensity(self, u, i):
        """

        Args:
            u (jnp.ndarray): (batch_size, )
            i (jnp.ndarray): (batch_size, )
        """
        probs = jnp.array([0])
        index = np.arange(len(x))
        for batch in range(num_batches):
            batch_idx = index[batch*batch_size: (batch+1)* batch_size]
            user = np.identity(self.num_users)[u[batch_idx]]
            item = np.identity(self.num_items)[i[batch_idx]]
            
            batch_x = np.concatenate([user, item], axis=1)
            prob = logistics.predict(self.params, batch_x)
            probs = jnp.concatenate([probs, prob])
        probs = jax.device_get(probs)[1:]
        return probs


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



def ips_mse(u_emb, i_emb, u_bias, i_bias, rating, p, lam):
    @jax.vmap
    def one_ips_mse(u_emb, i_emb, u_bias, i_bias, rating, p):
        preds = u_emb@i_emb + u_bias + i_bias
        loss = jnp.square(rating - preds) + lam * (jnp.sum(u_emb**2) + jnp.sum(i_emb**2) + u_bias**2 + i_bias**2)
        loss /= p
        return loss
    return jnp.mean(one_ips_mse(u_emb, i_emb, u_bias, i_bias, rating, p))


@jax.jit
def update(
    params:Dict[str, jnp.ndarray],
    data:np.ndarray,
    alpha:float,
    lam:float
)->Dict[str, jnp.ndarray]:
    #data = common_utils.shard(data)
    user, item, rating, p = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    user, item = user.astype(jnp.int32), item.astype(jnp.int32)
    
    u_emb = params.get("user_embedding")[user] # (batch_size, embed_dim)
    i_emb = params.get("item_embedding")[item] # (batch_size, embed_dim)
    
    u_bias = params.get("user_bias")[user] # (batch_size, )
    i_bias = params.get("item_bias")[item] #(batch_size, )
    
    loss = ips_mse(u_emb, i_emb, u_bias, i_bias, rating, p, lam)

    u_emb_grad = jax.grad(lambda u:ips_mse(u, i_emb, u_bias, i_bias, rating, p, lam))(u_emb)
    i_emb_grad = jax.grad(lambda i:ips_mse(u_emb, i, u_bias, i_bias, rating, p, lam))(i_emb)
    u_bias_grad = jax.grad(lambda ub:ips_mse(u_emb, i_emb, ub, i_bias, rating, p, lam))(u_bias)
    i_bias_grad = jax.grad(lambda ib:ips_mse(u_emb, i_emb, u_bias, ib, rating, p, lam))(i_bias)

    params["user_embedding"] = params["user_embedding"].at[user].set(u_emb - alpha * u_emb_grad)
    params["item_embedding"] = params["item_embedding"].at[item].set(i_emb - alpha * i_emb_grad)
    
    params["user_bias"] = params["user_bias"].at[user].set(u_bias - alpha * u_bias_grad)
    params["item_bias"] = params["item_bias"].at[item].set(i_bias - alpha * i_bias_grad)
    
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
    u, i = u.astype(jnp.int32), i.astype(jnp.int32)
    u_emb = params.get("user_embedding")[u]
    i_emb = params.get("item_embedding")[i]
    
    u_bias = params.get("user_bias")[u] # (batch_size, )
    i_bias = params.get("item_bias")[i] #(batch_size, )

    preds = (u_emb * i_emb).sum(axis=1) + u_bias + i_bias
    
    loss = jnp.square(r - preds)
    return jnp.mean(loss)

if __name__ == "__main__":
    import pandas as pd
    import os
    from sklearn.model_selection import train_test_split
    import scipy.sparse as sp
    
    HOME = os.environ["HOME"]
    data_path = "%s/work/dataset/YahooR3/rating.csv"%(HOME)
    df = pd.read_csv(data_path)
    dtrain, dtest = train_test_split(df, test_size=0.1, random_state=0)
    
    train = sp.coo_matrix((np.ones(len(dtrain)), (dtrain["userId"].values, dtrain["itemId"].values))).toarray()
    
    dtrain = []
    for u in range(train.shape[0]):
        for i in range(train.shape[1]):
            dtrain += [[u, i, train[u, i]]]
    
    dtrain = np.asarray(dtrain)
    
    num_users, num_items = df["userId"].unique().shape[0], df["itemId"].unique().shape[0]
    
    pmodel = Prospensity(num_items, num_users)
    

    pmodel.fit(dtrain[:, :2], dtrain[:, 2])
    
    print("Predict Observation Probability : ", pmodel.get_prospensity(dtest["userId"].values, dtest["itemId"].values)[:, 1])