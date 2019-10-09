from pathlib import Path
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

class Browser():
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

class Metadata(Browser):
    """ pass """
    def __init__(self, driver_path, base_url='https://www.sephora.com'):
        super(Metadata, self).__init__(driver_path)
        self.url = base_url
    
    def begin_extraction(self):
        drv = self.open_browser(True)
        drv.get(self.url)