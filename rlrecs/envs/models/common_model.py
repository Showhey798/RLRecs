from typing import Dict, Optional
from collections import OrderedDict
from tqdm import tqdm

import numpy as np

import jax
import jax.numpy as jnp
from jax import random


class Model(object):
    
    def __init__(
        self, 
        module,
        logger
    ):
        self.module = module
        self.logger = logger
        
    def init_params(self, **kwargs):
        self.params = self.module.init_params(**kwargs)
        
    def train(
        self,
        rng:random.PRNGKey,
        data:np.ndarray, 
        batch_size:Optional[int]=64,
        epochs:Optional[int]=10, 
        **kwargs
    ):
        rng, key = random.split(rng)
        index = jnp.arange(data.shape[0])
        random.permutation(key, index, independent=True)
        num_batches = int(data.shape[0] // batch_size)

        def train_one_epoch(**kwargs):
            
            batch_loss = []
        
            for batch in range(num_batches):
                batch_idx = index[batch * batch_size: (batch + 1) * batch_size]
                self.params, loss = self.module.update(self.params, data[batch_idx], **kwargs)
                batch_loss += [jax.device_get(loss)]
            
            return batch_loss
        

        loss_epochs = []
        with tqdm(range(epochs), desc="training", postfix="loss=") as ts:
            for epoch in ts:
                losshist = train_one_epoch(**kwargs)
                ts.set_postfix(OrderedDict(loss=np.mean(losshist)))
                self.logger.write_loss({"train_loss":np.mean(losshist)}, epoch)
                loss_epochs += [np.mean(losshist)]
    
        return loss_epochs
    
    def evaluate(
        self,
        data: jnp.ndarray,
        batch_size: Optional[int]=64
    ):
        index = jnp.arange(data.shape[0])
        num_batches = int(data.shape[0] // batch_size) + 1
        batch_loss = []
        with tqdm(range(num_batches), desc="evaluating") as ts:
            for batch in ts:
                batch_idx = index[batch * batch_size: (batch + 1) * batch_size]
                user, item, rating = data[batch_idx, 0], data[batch_idx, 1], data[batch_idx, 2]
                loss = self.module.predict(self.params, user, item, rating)
                batch_loss += [loss]
                ts.set_postfix(OrderedDict(loss=np.mean(batch_loss)))
        return np.mean(batch_loss)
    

    
if __name__ == "__main__":
    import configparser
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import pickle
    
    import sys
    sys.path.append("/home/inoue/work/RLRecs/")
    from rlrecs.envs.models import mf
    from rlrecs.logger import Logger
    
    config_path = "/home/inoue/work/RLRecs/config/mf.conf"
    config = configparser.ConfigParser()
    config.read(config_path)
    
    logger = Logger(
        logdir="/home/inoue/work/RLRecs/logs/test",
        datasetname="ml-100k",
        modelname="mf"
    )
    
    dataname = config["ENV"]["DATASETNAME"]
    
    data_path = "~/work/dataset/%s/rating.csv"%dataname
    df = pd.read_csv(data_path)

    dtrain, dtest = train_test_split(df, test_size=float(config["ENV"]["TESTSIZE"]), random_state=0)

    num_users, num_items = df["userId"].unique().shape[0], df["itemId"].unique().shape[0]
    
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    
    model = Model(mf, logger)

    model.init_params(num_users=num_users, num_items=num_items, embed_dim=int(config["ENV"]["EMBED_DIM"]), key=key)
    
    rng, key = jax.random.split(rng)
    
    losshist = model.train(
        rng, 
        dtrain[["userId", "itemId", "rating"]].values, 
        batch_size=int(config["ENV"]["BATCH_SIZE"]), 
        epochs=int(config["ENV"]["EPOCHS"]),
        alpha=float(config["ENV"]["ALPHA"]), 
        lam=float(config["ENV"]["LAM"])
    )
    
    model.evaluate(
        dtest[["userId", "itemId", "rating"]].values, 
        batch_size=int(config["ENV"]["BATCH_SIZE"])
    )
    