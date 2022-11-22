from typing import Optional, Dict
import tensorflow as tf
import os


class Logger(object):

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
        