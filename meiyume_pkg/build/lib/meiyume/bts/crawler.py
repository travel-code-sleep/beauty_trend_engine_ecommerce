
"""[summary]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import concurrent.futures
import os
import shutil
import time
from datetime import datetime, timedelta
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract
from pyarrow.lib import ArrowIOError
from selenium import webdriver
from selenium.common.exceptions import (ElementClickInterceptedException,
                                        NoSuchElementException,
                                        StaleElementReferenceException)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from meiyume.sph.cleaner import Cleaner
from meiyume.utils import Browser, Logger, MeiyumeException, Boots, chunks, convert_ago_to_date

warnings.simplefilter(action='ignore', category=FutureWarning)


class Metadata(Boots):
    """[summary]

    Arguments:
        Sephora {[type]} -- [description]
    """
    base_url = "https://www.boots.com"
    info = tldextract.extract(base_url)
    source = info.registered_domain

    @classmethod
    def update_base_url(cls, url):
        """[summary]

        Arguments:
            url {[type]} -- [description]
        """
        cls.base_url = url
        cls.info = tldextract.extract(cls.base_url)
        cls.source = cls.info.registered_domain

    def __init__(self, driver_path, log=True, path=Path.cwd(),
                 show=True):
        """[summary]

        Arguments:
            driver_path {[type]} -- [description]

        Keyword Arguments:
            log {bool} -- [description] (default: {True})
            path {[type]} -- [description] (default: {Path.cwd()})
            show {bool} -- [description] (default: {True})
        """
        super().__init__(driver_path=driver_path, show=show, path=path, data_def='meta')
        self.current_progress_path = self.metadata_path/'current_progress'
        self.current_progress_path.mkdir(parents=True, exist_ok=True)
        if log:
            self.prod_meta_log = Logger(
                "bts_prod_metadata_extraction", path=self.crawl_log_path)
            self.logger, _ = self.prod_meta_log.start_log()

    def get_product_type_urls(self):
        """[summary]
        """
        driver = self.open_browser()
        driver.get(self.base_url)
        id1 = driver.find_elements_by_css_selector('a[id*="subcategoryLink"]')

        # Extracting the category-sub category structure
        sub_cat = []
        for i in id1:
            sub_cat.append({'tab': i.get_attribute('name').split(':')[1],
                            'department': i.get_attribute('name').split(':')[2],
                            'category_raw': i.get_attribute('name').split(':')[3],
                            'sub_category_raw': i.get_attribute('name').split(':')[4],
                            'link': i.get_attribute('href')})
        df = pd.DataFrame(sub_cat)
        df2 = df[df['tab'] == 'shop by department']

        product_type = []
        for subcat_link in df2['link']:
            driver.get(subcat_link)
            try:
                if (len(driver.find_elements_by_css_selector('div.category-link>a')) > 0):
                    id2 = driver.find_elements_by_css_selector(
                        'div.category-link>a')
                    temp = []
                    for i in id2:
                        temp.append({'subcategory_link': subcat_link,
                                     'product_type': i.get_attribute('href').split('/')[-1],
                                     'url': i.get_attribute('href'),
                                     'type_available': 'Y',
                                     'tab': df2.loc[df2['link'] == subcat_link, 'tab'].values[0],
                                     'department': df2.loc[df2['link'] == subcat_link, 'department'].values[0],
                                     'category_raw': df2.loc[df2['link'] == subcat_link, 'category_raw'].values[0]
                                     })

                    product_type.extend(temp)
                    temp = [{'subcategory_link': subcat_link,
                             'product_type': df2.loc[df2['link'] == subcat_link, 'sub_category_raw'].values[0],
                             'url':subcat_link,
                             'type_available':'N',
                             'tab':df2.loc[df2['link'] == subcat_link, 'tab'].values[0],
                             'department':df2.loc[df2['link'] == subcat_link, 'department'].values[0],
                             'category_raw':df2.loc[df2['link'] == subcat_link, 'category_raw'].values[0]
                             }]
                    product_type.extend(temp)
                    print(
                        "Extraction done by extracting within page categories as product types: ", subcat_link)

                else:
                    temp = [{'subcategory_link': subcat_link,
                             'product_type': df2.loc[df2['link'] == subcat_link, 'sub_category_raw'].values[0],
                             'url':subcat_link,
                             'type_available':'N',
                             'tab':df2.loc[df2['link'] == subcat_link, 'tab'].values[0],
                             'department':df2.loc[df2['link'] == subcat_link, 'department'].values[0],
                             'category_raw':df2.loc[df2['link'] == subcat_link, 'category_raw'].values[0]
                             }]
                    product_type.extend(temp)
                    print(
                        "No within page categories found, extraction done by using the sub category name as product types: ", subcat_link)
                df = pd.DataFrame(product_type)
                df.to_feather(
                    'self.metadata_path/bts_product_cat_subcat_structure')
            except:
                print("Error in extracting info at ", subcat_link)

        driver.close()
        df.drop_duplicates(subset='url', inplace=True)
        df.reset_index(inplace=True, drop=True)
        df['scraped'] = 'N'
        df.to_feather(self.metadata_path/f'bts_product_type_urls_to_extract')
        return df

    def download_metadata(self, fresh_start):
        """[summary]

        Arguments:
            fresh_start {[type]} -- [description]
        """

        product_meta_data = []

        def fresh():
            """[summary]
            """
            product_type_urls = self.get_product_type_urls()
            # progress tracker: captures scraped and error desc
            progress_tracker = pd.DataFrame(index=product_type_urls.index, columns=[
                                            'product_type', 'scraped', 'error_desc'])
            progress_tracker.scraped = 'N'
            return product_type_urls, progress_tracker

        if fresh_start:
            self.logger.info('Starting Fresh Extraction.')
            product_type_urls, progress_tracker = fresh()
        else:
            if Path(self.metadata_path/'bts_product_type_urls_to_extract').exists():
                try:
                    progress_tracker = pd.read_feather(
                        self.metadata_path/'bts_metadata_progress_tracker')
                except ArrowIOError:
                    raise MeiyumeException(f"File bts_product_type_urls_to_extract can't be located in the path {self.metadata_path}.\
                                             Please put progress file in the correct path or start fresh extraction.")
                product_type_urls = pd.read_feather(
                    self.metadata_path/'bts_product_type_urls_to_extract')
                if sum(progress_tracker.scraped == 'N') > 0:
                    self.logger.info(
                        'Continuing Metadata Extraction From Last Run.')
                    product_type_urls = product_type_urls[product_type_urls.index.isin(
                        progress_tracker.index[progress_tracker.scraped == 'N'].values.tolist())]
                else:
                    self.logger.info(
                        'Previous Run Was Complete. Starting Fresh Extraction.')
                    product_type_urls, progress_tracker = fresh()
            else:
                self.logger.info(
                    'URL File Not Found. Starting Fresh Extraction.')
                product_type_urls, progress_tracker = fresh()

        driver = self.open_browser()
        for pt in product_type_urls.index:
            cat_name = product_type_urls.loc[pt, 'category_raw']
            #sub_cat_name = product_type_urls.loc[pt,'sub_category_raw']
            product_type = product_type_urls.loc[pt, 'product_type']
            product_type_link = product_type_urls.loc[pt, 'url']

            progress_tracker.loc[pt, 'product_type'] = product_type

            if 'best-selling' in product_type or 'new' in product_type:
                progress_tracker.loc[pt, 'scraped'] = 'NA'
                continue

            driver.get(product_type_link)
            time.sleep(8)
            # click and close welcome forms

            def meta_extract(product_type_link):
                driver.get(product_type_link)
                time.sleep(3)
                meta = []

                try:
                    id1 = int(driver.find_element_by_css_selector(
                        'div[class*="pageControl number"]').get_attribute("data-pages"))
                except:
                    id1 = 1

                for i in range(id1):
                    time.sleep(8)
                    x = driver.find_elements_by_css_selector(
                        'div[class*="estore_product_container"]')
                    for j in range(len(x)):
                        prod_id = "bts_" + \
                            x[j].get_attribute('data-productid').split(".")[0]
                        # print(prod_id)
                        try:
                            product_name = x[j].find_element_by_css_selector(
                                'div.product_name').text
                            # print(prod_name)
                        except:
                            product_name = "NaN"
                        try:
                            mrp = x[j].find_element_by_css_selector(
                                'div.product_price').text
                        except:
                            mrp = "NaN"
                        try:
                            product_page = x[j].find_element_by_css_selector(
                                'div.product_name>a').get_attribute('href')
                        except:
                            product_page = "NaN"
                        try:
                            rating = x[j].find_element_by_css_selector(
                                'div.product_rating>span').get_attribute('aria-label').split(" ")[0]
                        except:
                            rating = "NaN"

                        high_p = "NaN"
                        low_p = "NaN"
                        timestamp = datetime.now()
                        source = "Boots.com"
                        brand = "NaN"
                        meta.append({
                            "brand": brand,
                            "prod_id": prod_id,
                            "product_name": product_name,
                            "mrp": mrp,
                            "low_p": low_p,
                            "high_p": high_p,
                            "product_page": product_page,
                            "rating": rating,
                            "meta_date": timestamp,
                            "source": source,
                            "complete_scrape_flag": "N"
                        })
                    if id1 != 1:
                        driver.find_element_by_css_selector(
                            'a[title*="Show next"]').click()
                return(meta)

                product_meta_data.extend(meta_extract(product_type_link))
                #print("Extracted: ", i)

            if len(product_meta_data) > 0:
                product_meta_df = pd.DataFrame(product_meta_data)
                product_meta_df.to_feather(
                    self.current_progress_path/f'bts_prod_meta_extract_progress_{product_type}_{time.strftime("%Y-%m-%d-%H%M%S")}')
                self.logger.info(
                    f'Completed till IndexPosition: {pt} - ProductType: {product_type}. (URL:{product_type_link})')
                progress_tracker.loc[pt, 'scraped'] = 'Y'
                progress_tracker.to_feather(
                    self.metadata_path/'bts_metadata_progress_tracker')
                product_meta_data = []
        self.logger.info('Metadata Extraction Complete')
        print('Metadata Extraction Complete')
        #self.progress_monitor.info('Metadata Extraction Complete')
        driver.close()

    def extract(self, fresh_start=False, delete_progress=True, clean=True, download=True):
        """[summary]

        Keyword Arguments:
            fresh_start {bool} -- [description] (default: {False})
            delete_progress {bool} -- [description] (default: {True})
        Returns:
        """
        if download:
            self.download_metadata(fresh_start)
        self.logger.info('Creating Combined Metadata File')
        files = [f for f in self.current_progress_path.glob(
            "bts_prod_meta_extract_progress_*")]
        li = [pd.read_feather(file) for file in files]
        metadata_df = pd.concat(li, axis=0, ignore_index=True)
        metadata_df.reset_index(inplace=True, drop=True)
        metadata_df['source'] = self.source
        filename = f'bts_product_metadata_all_{time.strftime("%Y-%m-%d")}'
        metadata_df.to_feather(self.metadata_path/filename)
        self.logger.info(
            f'Metadata file created. Please look for file {filename} in path {self.metadata_path}')
        print(
            f'Metadata file created. Please look for file {filename} in path {self.metadata_path}')
        if delete_progress:
            shutil.rmtree(
                f'{self.metadata_path}\\current_progress', ignore_errors=True)
            self.logger.info('Progress files deleted')
        if clean:
            cleaner = Cleaner()
            metadata_df_clean_no_cat = cleaner.clean_data(
                data=metadata_df, filename=filename)
            self.logger.info(
                'Metadata Cleaned and Removed Duplicates for Details Extraction.')
        self.logger.handlers.clear()
        self.prod_meta_log.stop_log()
