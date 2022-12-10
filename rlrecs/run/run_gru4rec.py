import configparser
from typing import Optional
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
from rlrecs.envs.dataset import session_preprocess_data, DataLoader, split_data
from rlrecs.eval.agent_evaluator import evaluate
    
def run_offline(train_rate:Optional[float]=0.8, eval:Optional[bool]=False):
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
    train, test = split_data(train, train_rate)
    train, valid = split_data(train, 0.9)
    
    train_loader = DataLoader(train)
    valid_loader = DataLoader(valid)
    test_loader = DataLoader(test)
    
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
        results = evaluate(test_loader,agent)
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
        
        trainer.fit(
            key,
            train_data=train_loader,
            test_data=valid_loader,
            epochs=max_iteration,
            batch_size=batch_size
        )
        
        trainer.agent.save("%s/work/RLRecs/models/RC15/gru4rec"%(HOME))
        
        logger.info("Agent Saved")
    
        results = evaluate(test_loader,trainer.agent)
        print("Result : ", results)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train_rate", type=float, default=0.8)
    
    args = parser.parse_args()
    run_offline(args.train_rate, args.eval)
    