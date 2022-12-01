from typing import Optional
import numpy as np
from tqdm import tqdm

from rlrecs.agents.models import BaseAgent
from rlrecs.envs.dataset import DataLoader
from .metrics import recall_at_k


def evaluate(
    dataloader: DataLoader,
    agent:BaseAgent,
    metric_funcs={"recall":recall_at_k},
    batch_size:int=256,
    k:int=10
):
    metric_scores = {key:[] for key in metric_funcs.keys()}
    for batch in tqdm(dataloader.batch(batch_size)):
        state, feedback, trueIds, _, _, _, _ = batch
        inputs = (state, feedback)
        recommends = agent.recommend(inputs, is_greedy=True, k=k)

        for metric in metric_scores.keys():
            metric_scores[metric] += [metric_funcs[metric](trueIds, recommends, k)]
    
    results = {metric+"@%d"%k : np.mean(metric_scores[metric]) for metric in metric_scores.keys()}
    return results
        
    
    
        
        
    
