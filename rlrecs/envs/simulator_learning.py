import configparser
from sklearn.model_selection import train_test_split
import pandas as pd

import os
HOME = os.environ['HOME']
import sys
sys.path.append("%s/work/RLRecs/"%HOME)

import jax
from jax import numpy as jnp
import numpy as np

from rlrecs.logger import LossLogger, create_logger
from rlrecs.envs.common_model import Model
from rlrecs.envs.models import mfips, mf
from rlrecs.envs.dataset import preprocess_data

def main():
    config_path = "%s/work/RLRecs/config/mfips.conf"%HOME
    config = configparser.ConfigParser()
    config.read(config_path)
    
    dataname = config["ENV"]["DATASETNAME"]

    logger = create_logger("mf-ips")
    
    logger.info("Preprocessing Data")
    dtrain, dvalid, dtest, num_users, num_items = preprocess_data("YahooR3")
    logger.info("Preprocessed Data")
    
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    
    losslogger = LossLogger(
        logdir="%s/work/RLRecs/logs/test/"%(HOME),
        datasetname=dataname,
        modelname="mf-ips")

    logger.info("Setup logger")

    model = Model(mfips, losslogger)
    logger.info("Setup model")
    
    model.init_params(num_users=num_users, num_items=num_items, embed_dim=int(config["ENV"]["EMBED_DIM"]), key=key)

    rng, key = jax.random.split(rng)
    logger.info("Training...")
    loss_hist = model.fit(
        key,
        dtrain[["userId", "itemId", "rating", "pscores"]].values,
        dvalid[["userId", "itemId", "rating"]].values,
        batch_size=int(config["ENV"]["BATCH_SIZE"]), 
        epochs=int(config["ENV"]["EPOCHS"]),
        alpha=float(config["ENV"]["ALPHA"]), 
        lam=float(config["ENV"]["LAM"])
    )
    logger.info("Evaluating...")
    
    loss = model.evaluate(
        dtest[["userId", "itemId", "rating"]].values, 
        batch_size=int(config["ENV"]["BATCH_SIZE"]))
    print("Test Loss : ", loss)
    
    u_emb = jax.device_get(model.params["user_embedding"])
    i_emb = jax.device_get(model.params["item_embedding"])
    u_bias = jax.device_get(model.params["user_bias"])
    i_bias = jax.device_get(model.params["item_bias"])
    
    ratings = u_emb @i_emb.T + np.expand_dims(u_bias, axis=1) + np.expand_dims(i_bias, axis=0)
    
    def sigmoid(x):
        return 1./(np.exp(-x) + 1)
    
    ratings = np.clip(np.round(sigmoid(ratings)*4)+1, 1, 5)
    np.save("%s/work/RLRecs/results/envs/%s/rating.npy"%(HOME, dataname), ratings)
    logger.info("Rating File saved.")
    

if __name__ == "__main__":
    main()