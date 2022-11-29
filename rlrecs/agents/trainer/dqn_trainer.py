from typing import Optional
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from logging import Logger
from collections import OrderedDict
import numpy as np
from cpprb import ReplayBuffer

from rlrecs.agents.trainer import AgentTrainer
from rlrecs.agents.models import DQN
from rlrecs.logger import LossLogger
from rlrecs.envs import Env


class DQNTrainer(AgentTrainer):
    
    def __init__(
        self, 
        agent: DQN, 
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
        
        
            
    def online_train(
        self,
        key,
        env:Env,
        memory:Optional[ReplayBuffer]=None,
        max_iteration:Optional[int]=10000,
        batch_size:Optional[int]=64,
        update_count_by_each_iter:Optional[int]=10
    ):
        self.logger.info("Start Training Agent")
        
        self.agent.init_params(batch_size, key)
        reward_hist = []
        
        num_users = env.num_users
        user_inds = np.arange(num_users)
        np.random.shuffle(user_inds)
        
        num_batches = int(num_users // batch_size) + 1
        with logging_redirect_tqdm():
            with tqdm(range(max_iteration), desc="Training Agent") as ts:
                for itr in ts:
                    #self.logger.info("iteration %d Start."%(itr+1))
                    cum_rewards = np.zeros((num_users, ), dtype=np.float32)
                    losses = []
                    for batch in range(num_batches):
                        users = user_inds[batch*batch_size:(batch+1)*batch_size]
                        state, feedback = env.reset(users)
                        
                        while True:
                            click_masks = env.click_mask
                            actions = self.agent.recommend((state, feedback), click_masks=click_masks)
                            n_state, n_feedback, rewards, done = env.step(actions)
                            
                            for i in range(state.shape[0]):
                                memory.add(
                                    state=state[i, :],
                                    feedback=feedback[i, :],
                                    action=actions[i],
                                    n_state=n_state[i, :],
                                    n_feedback=n_feedback[i, :],
                                    reward=rewards[i],
                                    done=done)
                                cum_rewards[users[i]] += rewards[i]
                            state = n_state
                            feedback=n_feedback
                            
                            if done:
                                break
                        
                        
                        for i in range(update_count_by_each_iter):
                            samples = memory.sample(batch_size)
                            samples = [np.squeeze(samples[key]) for key in samples.keys()]
                            loss = self.agent.train_step(samples)
                            losses += [loss]
                        
                        self.end_epochs()
                        
                        
                    mean_total_reward = np.mean(cum_rewards)
                    reward_hist += [mean_total_reward]
                    ts.set_postfix(OrderedDict(CumReward=mean_total_reward, loss=np.mean(losses)))
                    
                    if self.losslogger is not None:
                        self.losslogger.write_loss({"CumReward":mean_total_reward, "TD Error": np.mean(losses)}, itr)
                    #self.logger.info("Iteration %d End."%(itr+1))

        self.logger.info("End Agent Training")