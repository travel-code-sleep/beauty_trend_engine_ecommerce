from pathlib import Path
import tldextract
import pandas as pd
import time
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from utils import Logger, Browser


class Metadata(Browser):
    """ pass """
    base_url="https://www.sephora.com"
    info = tldextract.extract(base_url)
    source = info.registered_domain

    @classmethod
    def update_base_url(cls, url):
        cls.base_url = url
        cls.info = tldextract.extract(cls.base_url)
        cls.source = cls.info.registered_domain

    def __init__(self, driver_path):
        super().__init__(driver_path)
        #self.url = base_url
        self.logger = Logger("sph_prod_metadata_extraction").set_log()
    
    def get_categories(self):
        """ pass """
        drv  = self.begin_extraction()
        cats = drv.find_elements_by_class_name("css-1t5gbpr")
        cat_urls = []
        for c in cats:
            cat_urls.append((c.get_attribute("href").split("/")[-1], c.get_attribute("href")))
            self.logger.info(str.encode(f'Category:- name:{c.get_attribute("href").split("/")[-1]} , url:{c.get_attribute("href")}', "utf-8", "ignore"))
        sub_cat_urls = []
        for cu in cat_urls: 
            cat_name = cu[0]
            cat_url = cu[1]
            drv.get(cat_url)
            time.sleep(5)
            sub_cats = drv.find_elements_by_class_name("css-16yq0cc")
            sub_cats.extend(drv.find_elements_by_class_name("css-or7ouu"))
            if len(sub_cats)>0:
                for s in sub_cats: 
                    sub_cat_urls.append((cat_name, s.get_attribute("href").split("/")[-1], s.get_attribute("href")))
                    self.logger.info(str.encode(f'Subcategory:- name:{s.get_attribute("href").split("/")[-1]} , url:{s.get_attribute("href")}', "utf-8", "ignore"))
            else:
                sub_cat_urls.append((cat_name, cat_url.split('/')[-1], cat_url))
        item_urls = []
        for su in sub_cat_urls:
            cat_name = su[0]
            sub_cat_name = su[1]
            sub_cat_url = su[2]
            drv.get(sub_cat_url)
            time.sleep(3)
            item_types = drv.find_elements_by_class_name('css-h6ss0r')
            if len(item_types)>0:
                for item in item_types:
                    item_urls.append((cat_name, sub_cat_name, item.get_attribute("href").split("/")[-1], item.get_attribute("href")))
                    self.logger.info(str.encode(f'ItemType:- name:{item.get_attribute("href").split("/")[-1]} , url:{item.get_attribute("href")}', "utf-8", "ignore"))
            else:
                item_urls.append((cat_name, sub_cat_name, sub_cat_url.split('/')[-1], sub_cat_url))
        df = pd.DataFrame(item_urls, columns = ['category_raw', 'sub_category_raw', 'item_type', 'item_url'])
        df.to_csv('sph_item_type_urls_to_extract.csv', index=None)        
        return df
    
    def begin_extraction(self, logs=True):
        drv = self.open_browser(True)
        drv.get(self.base_url)
        return drv 

