import numpy as np
import scipy.sparse as sp

class SOFAEnv(object):

    def __init__(self, config):
        super(SOFAEnv, self).__init__()
        self.config = config
        # config setting
        self.episode_length = int(self.config['EPISODE_LENGTH'])

        # load ratings file
        self.ratings = np.loadtxt(fname=self.config['RATING_FILE']) 
        self.ratings = self.config['RATINGS']

        
        # train and test split (to-do-list)

        self.action_space = np.arange(self.num_items)
        # if one item is not avalible for a user, just mask this slot, self.user_action_mask[u, j] = True
        self.user_action_mask = sp.dok_matrix((self.num_users, self.num_items), dtype=bool)

        # initialize the env
        self.click_probability()
        self.state = [[], []] # [items, feedbacks]
        # np.random.seed(2020)

    def reset(self, user_id):
        self.user_id = user_id # to-do-list: multiple users
        self.history_items = set()
        self.step_count = 0
        self.state = [[], []]

    
    def step(self, action):
        '''
        Input: Action (item_id or item_ids)
        Return: state, reward, done, {some info}
        '''
        if action in self.state[0]: # to-do-list: consider to change it into unclick
            print("recommend repeated item")
            exit(0)
        
        click_flag = self.get_respond(action)
        self.history_items.add(action) # now state is history_items, to-do-list: user_id/info
        # if click_flag == 1:
        #     self.state.append(action)
        self.state[0].append(action)
        self.state[1].append(click_flag)
        self.step_count += 1
        done = False
        if self.step_count >= self.episode_length:
            done = True
        return (self.state, click_flag, done)
            

    def click_probability(self):
        # self.click_p = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
        self.click_p = [0, 0, 0, 0, 1., 1.]
    def get_rating(self, item_id):
        return self.ratings[self.user_id, item_id]
    def get_respond(self, item_id):
        r = self.get_rating(item_id)
        if np.random.rand() <= self.click_p[r]:
            return 1
        else:
            return 0
    