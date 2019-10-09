import datetime
import logging
import time
import numpy as np 
import os 
class Logger(object):
    """ pass """
    def __init__(self, task_name):
        self.filename = f'{task_name}_{time.strftime("%Y-%m-%d-%H%M%S")}'
    
    def if_exist(self):
        try:
            os.remove(filename)
        except OSError:
            pass
    
    def set_log(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        file_handler = logging.FileHandler(self.filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.WARNING)
        logger.addHandler(stream_handler)
        return logger
    
def nan_equal(a,b):
        """pass"""
        try:
            np.testing.assert_equal(a,b)
        except AssertionError:
            return False
        return True