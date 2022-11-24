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
        logger=None
    ):
        self.module = module
        self.logger = logger
        
    def init_params(self, **kwargs):
        self.params = self.module.init_params(**kwargs)
    
    def train_one_epoch(self, rng, data, batch_size, **kwargs):
        rng, key = random.split(rng)
        index = jnp.arange(data.shape[0])
        random.permutation(key, index, independent=True)
        num_batches = int(data.shape[0] // batch_size)
        
        batch_loss = []
    
        for batch in range(num_batches):
            batch_idx = index[batch * batch_size: (batch + 1) * batch_size]
            self.params, loss = self.module.update(self.params, data[batch_idx], **kwargs)
            batch_loss += [jax.device_get(loss)]
        
        return np.mean(batch_loss)
        
    def train(
        self,
        rng:random.PRNGKey,
        data:np.ndarray, 
        batch_size:Optional[int]=64,
        epochs:Optional[int]=10, 
        **kwargs
    ):

        loss_epochs = []
        with tqdm(range(epochs), desc="training", postfix="loss=") as ts:
            for epoch in ts:
                batch_loss = self.train_one_epoch(rng, data, batch_size, **kwargs)
                ts.set_postfix(OrderedDict(loss=batch_loss))
                if self.logger:
                    self.logger.write_loss({"train_loss":batch_loss}, epoch)
                loss_epochs += [batch_loss]
    
        return loss_epochs
    
    def evaluate(
        self,
        data: jnp.ndarray,
        batch_size: Optional[int]=64
    ):
        index = jnp.arange(data.shape[0])
        num_batches = int(data.shape[0] // batch_size)
        batch_loss = []

        for batch in range(num_batches):
            batch_idx = index[batch * batch_size: (batch + 1) * batch_size]
            user, item, rating = data[batch_idx, 0], data[batch_idx, 1], data[batch_idx, 2]
            loss = self.module.predict(self.params, user, item, rating)
            batch_loss += [loss]
        return np.mean(batch_loss)

    def fit(
        self,
        rng:random.PRNGKey,
        train_data:jnp.ndarray,
        valid_data:jnp.ndarray,
        batch_size:Optional[int]=64,
        epochs:Optional[int]=64,
        **kwargs
    ):

        loss_epochs = {"train_loss":[], "valid_loss":[]}
        with tqdm(range(epochs), desc="training") as ts:
            for epoch in ts:
                train_loss = self.train_one_epoch(rng, train_data, batch_size, **kwargs)
                valid_loss = self.evaluate(valid_data, batch_size)
                ts.set_postfix(OrderedDict(train_loss=train_loss, valid_loss=valid_loss))
                if self.logger:
                    self.logger.write_loss({"train_loss":train_loss, "valid_loss":valid_loss}, episode=epoch)
                loss_epochs["train_loss"] += [train_loss]
                loss_epochs["valid_loss"] += [valid_loss]
        return loss_epochs

    
if __name__ == "__main__":
    import configparser
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    import os
    HOME = os.environ['HOME']
    import sys
    sys.path.append("%s/work/RLRecs/"%HOME)
    from rlrecs.envs.models import mf
    from rlrecs.logger import Logger
    
    config_path = "%s/work/RLRecs/config/mf.conf"%HOME
    config = configparser.ConfigParser()
    config.read(config_path)
    
    dataname = config["ENV"]["DATASETNAME"]
    if not int(config["ENV"]["TUNING"]):
        logger = Logger(
            logdir="%s/work/RLRecs/logs/test/"%(HOME),
            datasetname=dataname,
            modelname="mf"
        )
    else:
        logger = None
    
    #data_path = "%s/work/dataset/%s/rating.csv"%(HOME, dataname)
    #df = pd.read_csv(data_path)
    dtrain = pd.read_csv("/home/inoue/work/dataset/YahooR3/ydata-ymusic-rating-study-v1_0-train.txt", sep="\t", header=None)
    dtest = pd.read_csv("/home/inoue/work/dataset/YahooR3/ydata-ymusic-rating-study-v1_0-test.txt", sep="\t", header=None)
    dtrain.columns = ["userId", "itemId", "rating"]
    dtest.columns = ["userId", "itemId", "rating"]
    
    
    dvalid, dtest = train_test_split(dtest, test_size=float(config["ENV"]["VALIDSIZE"]), random_state=0)
    
    #dtrain, dtest = train_test_split(df, test_size=float(config["ENV"]["TESTSIZE"]), random_state=0)
    #dtrain, dvalid = train_test_split(dtrain, test_size=float(config["ENV"]["VALIDSIZE"]), random_state=0)

    num_users, num_items = dtrain["userId"].unique().shape[0]+1, dtrain["itemId"].unique().shape[0] + 1
    
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    
    model = Model(mf, logger)

    if(not int(config["ENV"]["TUNING"])):
        model.init_params(num_users=num_users, num_items=num_items, embed_dim=int(config["ENV"]["EMBED_DIM"]), key=key)
        
        rng, key = jax.random.split(rng)
        
        loss_hist = model.fit(
            rng,
            dtrain[["userId", "itemId", "rating"]].values,
            dvalid[["userId", "itemId", "rating"]].values,
            batch_size=int(config["ENV"]["BATCH_SIZE"]), 
            epochs=int(config["ENV"]["EPOCHS"]),
            alpha=float(config["ENV"]["ALPHA"]), 
            lam=float(config["ENV"]["LAM"])
        )
        
        loss = model.evaluate(
            dtest[["userId", "itemId", "rating"]].values, 
            batch_size=int(config["ENV"]["BATCH_SIZE"])
        )
        
        print("Test Loss : ", loss)
    else:
        print("------------------Parameter Tuning------------------")
        param_results = pd.DataFrame()
        i = 0
        for embed_dim in [5, 10, 20, 40]:
            for lam in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]:
                i += 1
                print("Trial %d/28 ----"%i)
                model.init_params(num_users=num_users, num_items=num_items, embed_dim=embed_dim, key=key)
                
                rng, key = jax.random.split(rng)
                
                loss_hist = model.fit(
                    rng,
                    dtrain[["userId", "itemId", "rating"]].values,
                    dvalid[["userId", "itemId", "rating"]].values,
                    batch_size=int(config["ENV"]["BATCH_SIZE"]), 
                    epochs=int(config["ENV"]["EPOCHS"]),
                    alpha=float(config["ENV"]["ALPHA"]), 
                    lam=lam
                )
    
                param_result = pd.DataFrame(
                    [embed_dim, lam, loss_hist["valid_loss"][-1]],
                    index=["EMBED_DIM", "LAM", "valid_loss"]
                ).T
                param_results = pd.concat([param_results, param_results], axis=0)
                
        param_results.to_csv("%s/work/RLRecs/results/envs/mf_params.csv"%HOME, index=False)