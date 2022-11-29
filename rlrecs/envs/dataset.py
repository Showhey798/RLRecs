from typing import Dict, Optional
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import json
np.random.seed(0)

import gc

from tqdm import tqdm

tqdm.pandas()

HOME = os.environ["HOME"]

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

class DataLoader():
    
    def __init__(
        self, 
        data:Dict[str, np.ndarray],
        train_rate:Optional[float]=None
    ):
        self._key = list(data.keys())[0]
        self.data = data
        self.index = np.arange(len(data[self._key]))
            
        self.num_batches = len(data[self._key])
        self.batch_size = 1
        
        if train_rate:
            self.train_ind = self.index[:int(len(self.index)*train_rate)]
            self.valid_ind = self.index[int(len(self.index)*train_rate):]
        else:
            self.train_ind = None
            self.valid_ind = None
        
        self.istrain = False
        self.isvalid = False
    
    def batch(self, batch_size):
        self.batch_size = batch_size
        
        if self.istrain:
            self.num_batches = int(len(self.train_ind) // batch_size)
        elif self.isvalid:
            self.num_batches = int(len(self.valid_ind) // batch_size)
        else:
            self.num_batches = int(len(self.index) // batch_size)
        
        return self
    
    def shuffle(self):
        if self.istrain:
            np.random.shuffle(self.train_ind)
        elif self.isvalid:
            np.random.shuffle(self.valid_ind)
        else:
            np.random.shuffle(self.index)
            
        return self
    
    def train(self):
        if self.train_ind is not None:
            self.istrain = True
            self.isvalid = False

    def valid(self):
        if self.valid_ind is not None:
            self.isvalid = True
            self.istrain = False    
    
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return next(self)
        
    def __next__(self):
        for batch in range(self.num_batches):
            if self.istrain:
                batch_idx = self.train_ind[batch * self.batch_size: (batch + 1) * self.batch_size]
            elif self.isvalid:
                batch_idx = self.valid_ind[batch * self.batch_size: (batch + 1) * self.batch_size]
            else:
                batch_idx = self.index[batch * self.batch_size: (batch + 1) * self.batch_size]
            yield (self.data[key][batch_idx] for key in self.data.keys())

def preprocess_data(datasetname="YahooR3", test_size=0.1, valid_size=0.1):
    file_path = Path("%s/work/dataset/%s/"%(HOME, datasetname))
    
    dtrain = pd.read_csv(file_path/"ydata-ymusic-rating-study-v1_0-train.txt", sep="\t", header=None)
    dtest = pd.read_csv(file_path/"ydata-ymusic-rating-study-v1_0-test.txt", sep="\t", header=None)
    dtrain.columns = ["userId", "itemId", "rating"]
    dtest.columns = ["userId", "itemId", "rating"]
    dtrain["userId"] -= 1
    dtrain["itemId"] -= 1
    dtest["userId"] -= 1
    dtest["itemId"] -= 1
    
    dvalid, dtest = train_test_split(dtest, test_size=valid_size)
    
    num_users, num_items = dtrain["userId"].unique().shape[0], dtrain["itemId"].unique().shape[0]
    
    user_item_count = dtrain.groupby("userId")["itemId"].count() # ユーザーごとに観測したアイテム数
    user_item_count = user_item_count.reset_index()
    max_user = user_item_count["itemId"].max()
    user_item_count["pscores"] = user_item_count["itemId"] / max_user
    user_item_count.drop(["itemId"], axis=1, inplace=True)
    dtrain = pd.merge(dtrain, user_item_count, on="userId", how="left")
    

    return dtrain, dvalid, dtest, num_users, num_items

def session_preprocess_data(
    datasetname="RC15", 
    test_size=0.1, 
    valid_size=0.1, 
    seq_len=10,
    logger=None
):
    file_path = Path("%s/work/dataset/%s/"%(HOME, datasetname))
    train_path = file_path / "yoochoose-clicks.dat"
    test_file = file_path / "yoochoose-test.dat"
    df = pd.read_csv(train_path, sep=",", header=None)
    
    
    if logger is not None:
        logger.info("Loaded Data")
    
    item_encoder = LabelEncoder()
    session_encoder = LabelEncoder()
    
    df.columns = ["sessionId", "timestamp", "itemId", "category"]
    itemIds = df["itemId"].unique()
    
    sessions = df.groupby("sessionId")["itemId"].nunique()
    sessions = sessions[sessions > seq_len].index # セッション内の長さがseq_len件以上のセッションのみを取り出す
    random_sessions = np.random.choice(sessions, size=2000000) # 200k件のセッションをサンプリング
    df = df[df["sessionId"].isin(random_sessions)]
    
    df["sessionId"] = session_encoder.fit_transform(df["sessionId"])
    df["itemId"] = item_encoder.fit_transform(df["itemId"])
    df["itemId"] += 1
    
    num_items = df["itemId"].max()+1
    
    if os.path.exists("/home/inoue/work/dataset/RC15/derived/train_valid.df"):
        logger.info("Loading Train Data")
        del df
        gc.collect()
        with open("/home/inoue/work/dataset/RC15/derived/train_valid.df", "rb") as f:
            train_data =  pickle.load(f)
        return train_data, num_items
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%X.%fZ")
    df.sort_values("timestamp", inplace=True)
    
    if logger is not None:
        logger.info("Preprocessed Data")
    
    
    def get_mdpdata(data):
        states = data[:-1]
        feedbacks = np.ones(states.shape)
        actions = data[1:, -1].reshape(-1, 1)
        n_states = data[1:]
        rewards = np.ones(len(actions)).reshape(-1, 1)
        dones = np.zeros(len(actions))
        dones[-1] = 1.
        dones = dones.reshape(-1, 1)
        return (states, feedbacks, actions, n_states, feedbacks, rewards, dones)
    
    def rolling(x):
        x = rolling_window(np.asarray(x), seq_len)
        return get_mdpdata(x)
        
    df = df.groupby("sessionId")["itemId"].progress_apply(rolling)
    
    train_data = {
        "state":np.vstack(df.apply(lambda x: x[0]).values).astype(np.int32),
        "feedabck":np.vstack(df.apply(lambda x: x[1]).values).astype(np.int32),
        "action": np.squeeze(np.vstack(df.apply(lambda x: x[2]).values)).astype(np.int32),
        "n_state": np.vstack(df.apply(lambda x: x[3]).values).astype(np.int32),
        "n_feedback": np.vstack(df.apply(lambda x: x[4]).values).astype(np.int32),
        "reward" : np.squeeze(np.vstack(df.apply(lambda x: x[2]).values)) .astype(np.float32),
        "done" : np.squeeze(np.vstack(df.apply(lambda x: x[2]).values)).astype(np.float32)
    }
    
    with open("/home/inoue/work/dataset/RC15/derived/train_valid.df", "wb") as files:
        pickle.dump(
            train_data,
            files)
    
    
    if logger is not None:
        logger.info("train data preprocessd.")
    
    return train_data, num_items
        
    # del df, data
    # gc.collect()

    # if logger is not None:
    #     logger.info("Test data constructing...")
    # test = pd.read_csv(test_file, sep=",", header=None)
    # test.columns = ["sessionId", "timestamp", "itemId", "category"]
    # if logger is not None:
    #     logger.info("Test data loaded")
    
    # # トレインと同じアイテムが存在しているデータを抽出
    # test = test[test["itemId"].isin(itemIds)]
    # test["sessionId"] = session_encoder.transform(test["sessionId"])
    # test["itemId"] = item_encoder.transform(test["itemId"])
    # test["itemId"] += 1
    # test["timestamp"] = pd.to_datetime(test["timestamp"], format="%Y-%m-%dT%X.%fZ")
    # test.sort_values("timestamp", inplace=True)
    
    # if logger is not None:
    #     logger.info("Test data preprocessed")
    
    # test = test.groupby("sessionId")["itemId"].progress_apply(rolling).reset_index()
    # data = np.vstack(test["itemId"].values)
    
    # test_data = {
    #     "state": data[:, :seq_len].astype(np.int32),
    #     "action":data[:, seq_len].astype(np.int32),
    #     "n_state":data[:, (seq_len+1):(2*seq_len+1)].astype(np.int32),
    #     "reward":data[:, 2*seq_len+1].astype(np.float32),
    #     "done": data[:, -1].astype(np.float32)
    # }
    
    # pickle.dump(test_data,open("/home/inoue/work/dataset/RC15/derived/test.df", "wb"))
    

if __name__ == "__main__":
    import sys
    sys.path.append("/home/inoue/work/RLRecs")
    from rlrecs.logger import create_logger
    logger = create_logger("session_data_preprocess")
    train, num_items = session_preprocess_data(seq_len=10, logger=logger)
    print(num_items)
    print("Preprocessed Data")