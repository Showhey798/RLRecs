from typing import Optional
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from logging import Logger
from collections import OrderedDict
import numpy as np
from cpprb import ReplayBuffer

from rlrecs.agents.trainer import AgentTrainer
from rlrecs.agents.models import BCQ
from rlrecs.logger import LossLogger
from rlrecs.envs import Env


class BCQTrainer(AgentTrainer):
    
    def __init__(
        self, 
        agent: BCQ, 
        logger: Logger,
        losslogger: Optional[LossLogger]=None, 
        update_count:Optional[int]=200
    ):
        
        self.epoch_count = 0
        self.update_count = update_count
        
        super().__init__(agent, logger, losslogger)

    def end_epochs(self):
        self.epoch_count += 1
        if self.epoch_count % self.update_count == 0:
            self.agent.target_model.replace(params=self.agent.model.params)