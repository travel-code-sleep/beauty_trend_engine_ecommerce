import datetime
import logging
import time
import numpy as np 
import os 
import missingno as msno
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class Browser(object):
    """ pass """
    def __init__(self, driver_path):
        """ pass """
        self.driver_path = driver_path

    def open_browser(self, show=False):
        if show:
            return webdriver.Chrome(executable_path=self.driver_path)
        else:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            return webdriver.Chrome(executable_path=self.driver_path, options=chrome_options)

    def create_driver(self, url):
        drv = self.open_browser(True)
        drv.get(url)
        return drv 

    @staticmethod
    def _scroll_down_page(driver, speed=8, h1=0, h2=1):
        current_scroll_position, new_height= h1, h2
        while current_scroll_position <= new_height:
            current_scroll_position += speed
            driver.execute_script("window.scrollTo(0, {});".format(current_scroll_position))
            new_height = driver.execute_script("return document.body.scrollHeight")
            
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

def show_missing_value(dataframe, viz_type=None):
        """pass"""
        if viz_type=='matrix':
            return msno.matrix(dataframe, figsize=(12,4))
        elif viz_type=='percentage':
            return dataframe.isna().mean() * 100
        elif viz_type=='dendrogram':
            return msno.dendrogram(dataframe, figsize=(12,8))
        else:
            return dataframe.isna().sum()
