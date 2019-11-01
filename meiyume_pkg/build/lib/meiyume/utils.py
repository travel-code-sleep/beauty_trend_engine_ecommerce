import datetime
import logging
import time
import numpy as np 
import os 
import missingno as msno
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import gc

class MeiyumeException(Exception):
    """class to define custom exceptions in runtime
    
    Arguments:
        Exception {[type]} -- [description]
    """
    pass
class Browser(object):
    """[summary]
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self, driver_path, show):
        """ pass """
        self.show = show
        self.driver_path = driver_path

    def open_headless(self, show=False):
        self.show = show
    
    def open_browser(self):
        """[summary]
        """
        if self.show:
            return webdriver.Chrome(executable_path=self.driver_path)
        else:
            chrome_options = Options()
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--headless')
            return webdriver.Chrome(executable_path=self.driver_path, options=chrome_options)

    # def create_driver(self, url):
    #     """[summary]
        
    #     Arguments:
    #         url {[type]} -- [description]
    #     """
    #     drv = self.open_browser()
    #     drv.get(url)
    #     return drv 

    @staticmethod
    def scroll_down_page(driver, speed=8, h1=0, h2=1):
        """[summary]
        
        Arguments:
            driver {[type]} -- [description]
        
        Keyword Arguments:
            speed {int} -- [description] (default: {8})
            h1 {int} -- [description] (default: {0})
            h2 {int} -- [description] (default: {1})
        """
        current_scroll_position, new_height= h1, h2
        while current_scroll_position <= new_height:
            current_scroll_position += speed
            driver.execute_script("window.scrollTo(0, {});".format(current_scroll_position))
            new_height = driver.execute_script("return document.body.scrollHeight")

class Sephora(Browser):
    """[summary]
    
    Arguments:
        Browser {[type]} -- [description]
    """
    def __init__(self, driver_path=None, path=Path.cwd(), show=True):
        super().__init__(driver_path=driver_path, show=show)
        self.path = Path(path)
        self.metadata_path = self.path/'sephora/metadata'
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self.detail_path = self.path/'sephora/detail'
        self.detail_path.mkdir(parents=True, exist_ok=True)
        self.review_path = self.path/'sephora/review'
        self.review_path.mkdir(parents=True, exist_ok=True)
        self.crawl_logs_path = self.path/'sephora/crawler_logs'
        self.crawl_logs_path.mkdir(parents=True, exist_ok=True)
        
class Logger(object):
    """[summary]
    
    Arguments:
        object {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """""" pass """
    def __init__(self, task_name, path):
        self.filename = path/f'{task_name}_{time.strftime("%Y-%m-%d-%H%M%S")}'
    
    def if_exist(self):
        try:
            os.remove(filename)
        except OSError:
            pass
    
    def start_log(self):
        """[summary]
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        self.file_handler = logging.FileHandler(self.filename)
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.WARNING)
        self.logger.addHandler(stream_handler)
        return self.logger, self.filename

    def stop_log(self):
        """[summary]
        """
        #self.logger.removeHandler(self.file_handler)
        del self.logger, self.file_handler
        gc.collect()

def nan_equal(a,b):
    """[summary]
    
    Arguments:
        a {[type]} -- [description]
        b {[type]} -- [description]
    """
    try:
        np.testing.assert_equal(a,b)
    except AssertionError:
        return False
    return True

def show_missing_value(dataframe, viz_type=None):
    """[summary]
    
    Arguments:
        dataframe {[type]} -- [description]
    
    Keyword Arguments:
        viz_type {[type]} -- [description] (default: {None})
    """
    if viz_type=='matrix':
        return msno.matrix(dataframe, figsize=(12,4))
    elif viz_type=='percentage':
        return dataframe.isna().mean() * 100
    elif viz_type=='dendrogram':
        return msno.dendrogram(dataframe, figsize=(12,8))
    else:
        return dataframe.isna().sum()
