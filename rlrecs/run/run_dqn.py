import configparser
from typing import Optional
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
from rlrecs.envs.dataset import session_preprocess_data, DataLoader
from rlrecs.eval.agent_evaluator import evaluate

def run_online():
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
        "/home/inoue/work/RLRecs/logs/online",
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
    
def run_offline(eval:Optional[bool]=False):
    config = configparser.ConfigParser()
    file_path = "/home/inoue/work/RLRecs/config/dqn.conf"
    config.read(file_path)
    hidden_dim = int(config["AGENT"]["HIDDEN_DIM"])
    seq_len=int(config["ENV"]["SEQ_LEN"])
    embed_dim=int(config["AGENT"]["EMBED_DIM"])
    learning_rate=float(config["AGENT"]["LEARNING_RATE"])
    gamma=float(config["AGENT"]["GAMMA"])
    max_iteration = int(config["AGENT"]["MAX_ITERATION"])
    batch_size = int(config["AGENT"]["BATCH_SIZE"])
    
    logger = create_logger("DQN")
    
    
    train, num_items = session_preprocess_data(seq_len=seq_len, logger=logger)
    
    train_loader = DataLoader(train, train_rate=0.8)
    
    agent = DQN(
        num_items,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        embed_dim=embed_dim,
        learning_rate=learning_rate,
        gamma=gamma
    )
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    
    if eval:
        logger.info("Evaluating...")
        agent.init_params(batch_size, key)
        agent.load("/home/inoue/work/RLRecs/models/RC15/dqn")
        logger.info("Loaded Agent")
        train_loader.valid()
        results = evaluate(train_loader,agent)
        print("Result : ", results)
    
    else:
        losslogger = LossLogger(
            "/home/inoue/work/RLRecs/logs/offline",
            "YahooR3",
            "DQN")
        
        trainer = DQNTrainer(
            agent,
            logger,
            losslogger=losslogger,
            update_count=int(config["AGENT"]["UPDATE_COUNT"])
        )
        
        train_loader.train()
        trainer.fit(
            key,
            train_loader,
            max_iteration,
            batch_size
        )
        
        trainer.agent.save("%s/work/RLRecs/models/RC15/dqn"%(HOME))
    
    
    
        
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--offline", action="store_true")
    
    args = parser.parse_args()
    
    if args.offline:
        run_offline(args.eval)
    else:
        run_online()
    