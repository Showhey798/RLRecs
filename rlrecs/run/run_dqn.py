import configparser
from sklearn.model_selection import train_test_split
import pandas as pd

import os
HOME = os.environ['HOME']
import sys
sys.path.append("%s/work/RLRecs/"%HOME)

import jax
from cpprb import ReplayBuffer
import numpy as np

from rlrecs.agents.trainer import DQNTrainer
from rlrecs.agents.models import DQN
from rlrecs.logger import LossLogger, create_logger
from rlrecs.envs import Env

def main():
    config = configparser.ConfigParser()
    file_path = "/home/inoue/work/RLRecs/config/dqn.conf"
    config.read(file_path)
    
    logger = create_logger("DQN")
    env = Env(config, logger)
    
    hidden_dim = int(config["AGENT"]["HIDDEN_DIM"])
    seq_len=int(config["ENV"]["SEQ_LEN"])
    embed_dim=int(config["AGENT"]["EMBED_DIM"])
    learning_rate=float(config["AGENT"]["LEARNING_RATE"])
    gamma=float(config["AGENT"]["GAMMA"])
    max_iteration = int(config["AGENT"]["MAX_ITERATION"])
    batch_size = int(config["AGENT"]["BATCH_SIZE"])
    
    agent = DQN(
        env.num_items,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        embed_dim=embed_dim,
        learning_rate=learning_rate,
        gamma=gamma
    )
    
    losslogger = LossLogger(
        "/home/inoue/work/RLRecs/logs",
        "YahooR3",
        "DQN")
    
    logger.info("Agent Constructed.")
    
    trainer = DQNTrainer(
        agent,
        logger,
        losslogger=losslogger,
        update_count=int(config["AGENT"]["UPDATE_COUNT"])
    )
    
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    
    memory = ReplayBuffer(
        int(config["AGENT"]["BUFFER_SIZE"]),
        env_dict={
            "state": {"shape": (seq_len,), "dtype":np.int32},
            "feedback": {"shape" : (seq_len, ), "dtype":np.int32},
            "action": {"dtype":np.int32},
            "n_state" : {"shape" : (seq_len,), "dtype":np.int32},
            "n_feedback": {"shape":(seq_len, ), "dtype":np.int32},
            "reward": {"dtype":np.float32},
            "done": {"dtype":np.float32}
        }
    )
    trainer.online_train(
        key,
        env,
        memory,
        max_iteration=max_iteration       ,
        batch_size=batch_size 
    )
    
if __name__ == "__main__":
    main()
    