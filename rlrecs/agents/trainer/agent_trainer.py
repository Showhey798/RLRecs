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
        for epoch in range(epochs):
            self.begin_epochs()
            losses = []
            with tqdm(data.shuffle().batch(batch_size), desc="[Epoch %d]"%epoch, postfix="loss=") as ts:
                for batch in ts:
                    loss = self.agent.train_step(batch)
                    losses += [loss]
                ts.set_postfix(OrderedDict(loss=np.mean(loss)))
            loss_hist += [np.mean(losses)]
            self.end_epochs()
            
        return loss_hist
    