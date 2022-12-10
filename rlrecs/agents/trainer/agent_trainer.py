from typing import Dict, Optional
from logging import Logger
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
np.random.seed(0)

from rlrecs.envs.dataset import DataLoader
from rlrecs.agents.models import BaseAgent
from rlrecs.logger import LossLogger
from rlrecs.eval.agent_evaluator import evaluate


class AgentTrainer(object):
    
    def __init__(
        self, 
        agent: BaseAgent,
        logger: Logger,
        losslogger:Optional[LossLogger]=None
    ):
        self.agent = agent
        self.losslogger = losslogger
        self.logger = logger
        
        self.logger.info("Agent Trainer Constructed.")
        
    def begin_epochs(self):
        pass
    
    def end_epochs(self):
        pass
    
    def fit(
        self, 
        key,
        train_data:DataLoader,
        test_data:Optional[DataLoader]=None,
        epochs:Optional[int]=100,
        batch_size:Optional[int]=256,
    ):
        self.agent.init_params(batch_size, key)
        loss_hist = []
        with tqdm(range(epochs), desc="Training Agent") as ts:
            for epoch in ts:
                self.begin_epochs()
                losses = []        
                for batch in train_data.shuffle().batch(batch_size):
                    loss = self.agent.train_step(batch)
                    losses += [loss]
                
                if test_data:
                    result = evaluate(
                        train_data,
                        self.agent,
                        batch_size=batch_size,
                        verbose=False
                        )
                else:
                    result = {}
                
                result["train_loss"] = np.mean(losses)
                
                if self.losslogger is not None:
                    self.losslogger.write_loss(result, epoch)
                    
                loss_hist += [np.mean(losses)]
                ts.set_postfix(result)
                self.end_epochs()
        return loss_hist
    