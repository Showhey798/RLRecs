import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.sparse as sp

import sys
sys.path.append("../..")
from rlrecs.envs.models.mfips import Prospensity
import jax

def preprocess_data(datasetname="YahooR3", test_size=0.1, valid_size=0.1, prospensity=True):
    HOME = os.environ["HOME"]
    file_path = Path("%s/work/dataset/%s/"%(HOME, datasetname))
    
    dtrain = pd.read_csv(file_path/"ydata-ymusic-rating-study-v1_0-train.txt", sep="\t", header=None)
    dtest = pd.read_csv(file_path/"ydata-ymusic-rating-study-v1_0-test.txt", sep="\t", header=None)
    
    dtrain.columns = ["userId", "itemId", "rating"]
    dtest.columns = ["userId", "itemId", "rating"]
    
    dvalid, dtest = train_test_split(dtest, test_size=valid_size)
    
    num_users, num_items = dtrain["userId"].unique().shape[0]+ 1, dtrain["itemId"].unique().shape[0] + 1
    
    user_item_count = dtrain.groupby("userId")["itemId"].count() # ユーザーごとに観測したアイテム数
    user_item_count = user_item_count.reset_index()
    max_user = user_item_count["itemId"].max()
    user_item_count["pscores"] = user_item_count["itemId"] / max_user
    user_item_count.drop(["itemId"], axis=1, inplace=True)
    dtrain = pd.merge(dtrain, user_item_count, on="userId", how="left")
    #pmodel = Prospensity(num_items, num_users, key=jax.random.PRNGKey(0))

    #pmodel.fit(dtrain.iloc[:, :2].values, dtrain.iloc[:, 2].values)
    
    #pscores = pmodel.get_prospensity(dtrain["userId"].values, drain["itemId"].values)
    #dtrain["pscores"] = pscores[:, 1]

    return dtrain, dvalid, dtest, num_users, num_items

if __name__ == "__main__":
    dtrain, _, _, _, _ = preprocess_data()
    print(dtrain).head(20)