from typing import Optional, Dict
import tensorflow as tf
import os

from logging import Logger, getLogger, StreamHandler, DEBUG, Formatter

def create_logger(
    name:Optional[str]=None
):
    logger = getLogger(name)
    logger.setLevel(DEBUG)
    logger.propagate = False
    
    ch = StreamHandler()
    ch.setLevel(DEBUG)
    ch.setFormatter(Formatter("%(asctime)s - %(filename)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)
    return logger

class LossLogger(object):

    def __init__(
        self, 
        logdir:str,
        datasetname: str,
        modelname : str
    ):
        self.logdir = logdir

        self.summary_writer = tf.summary.create_file_writer(os.path.join(logdir, datasetname, modelname))
    
    def write_loss(
        self,
        info:Dict[str, float],
        episode:int
    ):
        with self.summary_writer.as_default():
            for key in info.keys():
                tf.summary.scalar(key, info[key], step=episode)