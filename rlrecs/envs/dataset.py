from typing import Dict
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
np.random.seed(0)

HOME = os.environ["HOME"]


class DataLoader():
    
    def __init__(
        self, 
        data:Dict[str, np.ndarray]
    ):
        self.data = data
        self.index = np.arange(len(data[data.keys()[0]]))
        self.num_batches = len(data[data.keys()[0]])
        self.batch_size = 1
    
    def batch(self, batch_size):
        self.batch_size = batch_size
        self.num_batches = int(self.data[self.data.keys()[0]].shape[0] // batch_size)
        return self
    
    def shuffle(self):
        np.random.shuffle(self.index)
        return self
    
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return next(self)
        
    def __next__(self):
        for batch in range(self.num_batches):
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

def session_preprocess_data(datasetname="RC15", test_size=0.1, valid_size=0.1, seq_len=50):
    file_path = Path("%s/work/dataset/%s/"%(HOME, datasetname))
    file_path = file_path / "yoochoose-clicks.dat"
    test_file = file_path / "yoochoose-test.dat"
    df = pd.read_csv(file_path, sep=",", header=None)
    
    item_encoder = LabelEncoder()
    session_encoder = LabelEncoder()
    
    
    df.columns = ["sessionId", "timestamp", "itemId", "category"]
    
    df["sessionId"] = session_encoder.fit_transform(df["sessionId"])
    df["itemId"] = item_encoder.fit_transform(df["itemId"])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%X.%fZ")
    df.sort_values("timestamp", inplace=True)
    
    df = df.groupby("sessionId")["itemId"].unique().reset_index()
    for s in df["sessionId"]:
        df[df["sessionId"] == s]
    
    
    
    
    

if __name__ == "__main__":
    dtrain, _, _, _, _ = preprocess_data()
    print(dtrain).head(20)