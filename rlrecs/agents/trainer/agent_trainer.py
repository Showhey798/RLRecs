from typing import Dict, Optional
from logging import Logger
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
np.random.seed(0)

from rlrecs.envs.dataset import DataLoader
from rlrecs.agents.models import BaseAgent
from rlrecs.logger import LossLogger


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
        raise NotImplementedError()
    
    def end_epochs(self):
        raise NotImplementedError()
    
    def fit(
        self, 
        key,
        data:DataLoader,
        epochs:Optional[int]=100,
        batch_size:Optional[int]=256,
    ):
        self.agent.init_params(batch_size, key)
        loss_hist = []
        with tqdm(range(epochs), desc="Training Agent") as ts:
            for epoch in ts:
                #self.begin_epochs()
                losses = []        
                for batch in data.shuffle().batch(batch_size):
                    loss = self.agent.train_step(batch)
                    losses += [loss]
                
                if self.losslogger is not None:
                    self.losslogger.write_loss(
                        {"train_loss": np.mean(losses)},
                        epoch)
                loss_hist += [np.mean(losses)]
                ts.set_postfix(OrderedDict(loss=np.mean(losses)))
                self.end_epochs()
        return loss_hist
    