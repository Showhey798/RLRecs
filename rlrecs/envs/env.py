import numpy as np
from logging import Logger
from configparser import ConfigParser

class Env(object):

    def __init__(self, config:ConfigParser, logger:Logger):
        super(Env, self).__init__()
        
        self.seq_len = int(config["ENV"]["SEQ_LEN"])
        self.config = config
        self.logger = logger
        # config setting
        self.episode_length = int(self.config["ENV"]['EPISODE_LENGTH'])

        # load ratings file
        self.ratings = np.load(self.config["ENV"]['RATING_FILE']) 
        
        self.num_users = self.ratings.shape[0]
        self.num_items = self.ratings.shape[1]    
    
        # train and test split (to-do-list)

        self.action_space = np.arange(self.num_items)
        
        self.batch_size = int(self.config["AGENT"]["BATCH_SIZE"])

        # initialize the env
        self.click_probability()
        self.state = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        self.feedback = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        
        self.logger.info("Env is constructed.")

    def reset(self, user_ids):
        self.user_ids = user_ids
        self.batch_size = user_ids.shape[0]
        self.history_items = np.zeros((self.num_users, self.num_items), dtype=np.float32)
        self.step_count = 0
        self.state = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        self.feedback = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        return self.state, self.feedback

    
    def step(self, actions):
        self.state = np.roll(self.state, shift=-1, axis=1)
        self.feedback = np.roll(self.feedback, shift=-1, axis=1)
        rewards = np.zeros((self.batch_size, ), dtype=np.float32)
        for u, (i, action) in zip(self.user_ids, enumerate(actions)):
            click_flag = self.get_respond(u, action)
            self.history_items[u, action] = 1
            self.state[i, -1] = action
            self.feedback[i, -1] = click_flag
            rewards[i] = 1 if click_flag else -2
            
            self.step_count += 1
            done = False
            if self.step_count >= self.episode_length:
                done = True
            
        return (self.state, self.feedback, rewards, done)

    def click_probability(self):
        # self.click_p = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
        self.click_p = [0, 0, 0, 0, 1., 1.]
    def get_rating(self, user_id, item_id):
        return self.ratings[user_id, item_id]
    
    def get_respond(self, user_id, item_id):
        r = self.get_rating(user_id, item_id)
        if np.random.rand() <= self.click_p[int(r)]:
            return 1
        else:
            return 0

    @property    
    def click_mask(self):
        return self.history_items[self.user_ids, :]
    