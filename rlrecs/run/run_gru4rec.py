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

from rlrecs.agents.trainer import AgentTrainer
from rlrecs.agents.models import GRU4Rec
from rlrecs.logger import LossLogger, create_logger
from rlrecs.envs import Env
from rlrecs.envs.dataset import session_preprocess_data, DataLoader
from rlrecs.eval.agent_evaluator import evaluate
    
def run_offline(eval:Optional[bool]=False):
    config = configparser.ConfigParser()
    file_path = "/home/inoue/work/RLRecs/config/gru4rec.conf"
    config.read(file_path)
    hidden_dim = int(config["AGENT"]["HIDDEN_DIM"])
    seq_len=int(config["ENV"]["SEQ_LEN"])
    embed_dim=int(config["AGENT"]["EMBED_DIM"])
    learning_rate=float(config["AGENT"]["LEARNING_RATE"])
    max_iteration = int(config["AGENT"]["MAX_ITERATION"])
    batch_size = int(config["AGENT"]["BATCH_SIZE"])
    
    logger = create_logger("GRU4Rec")
    
    
    train, num_items = session_preprocess_data(seq_len=seq_len, logger=logger)
    
    train_loader = DataLoader(train, train_rate=0.8)
    
    agent = GRU4Rec(
        num_items,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        embed_dim=embed_dim,
        learning_rate=learning_rate,
    )
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    
    if eval:
        logger.info("Evaluating...")
        agent.init_params(batch_size, key)
        agent.load("/home/inoue/work/RLRecs/models/RC15/gru4rec")
        logger.info("Loaded Agent")
        train_loader.valid()
        results = evaluate(train_loader,agent)
        print("Result : ", results)
    
    else:
        losslogger = LossLogger(
            "/home/inoue/work/RLRecs/logs/offline",
            "RC15",
            "GRU4Rec")
        
        trainer = AgentTrainer(
            agent,
            logger,
            losslogger=losslogger
        )
        
        train_loader.train()
        trainer.fit(
            key,
            train_loader,
            max_iteration,
            batch_size
        )
        
        trainer.agent.save("%s/work/RLRecs/models/RC15/gru4rec"%(HOME))
    
        train_loader.valid()
        results = evaluate(train_loader,trainer.agent)
        print("Result : ", results)
    
        
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    
    args = parser.parse_args()
    run_offline(args.eval)
    