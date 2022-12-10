from typing import Optional
import numpy as np
from tqdm import tqdm

import jax
from jax import numpy as jnp

from rlrecs.agents.models import BaseAgent
from rlrecs.envs.dataset import DataLoader
from .metrics import recall_at_k


def evaluate(
    dataloader: DataLoader,
    agent:BaseAgent,
    metric_funcs={"recall":recall_at_k},
    batch_size:int=256,
    k:int=10,
    verbose:Optional[bool]=True
):
    metric_scores = {key:jnp.array([]) for key in metric_funcs.keys()}
    if verbose:
        ts = tqdm(dataloader.batch(batch_size))
    else:
        ts = dataloader.batch(batch_size)

    for batch in ts:
        state, _, trueIds, _, _, _, _ = batch
        recommends = agent.recommend((state), is_greedy=True)
        for metric in metric_scores.keys():
            metric_score = metric_funcs[metric](trueIds-1, recommends, k)
            metric_scores[metric] = jnp.hstack([metric_scores[metric], metric_score])
    results = {metric+"@%d"%k : np.mean(metric_scores[metric])for metric in metric_scores.keys()}
    return results
    
    
        
        
    
