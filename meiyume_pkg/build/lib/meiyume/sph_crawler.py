
"""[summary]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import concurrent.futures
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract
from pyarrow.lib import ArrowIOError
from selenium.common.exceptions import (NoSuchElementException,
                                        StaleElementReferenceException)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.keys import Keys

from .sph_cleaner import Cleaner
from .utils import Browser, Logger, MeiyumeException, Sephora, chunks


class Metadata(Sephora):
    """[summary]
    
    Arguments:
        Sephora {[type]} -- [description]
    """
    base_url="https://www.sephora.com"
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
        self.curnet_progress_path = self.metadata_path/'current_progress'
        self.curnet_progress_path.mkdir(parents=True, exist_ok=True)
        if log:
            self.prod_meta_log = Logger("sph_prod_metadata_extraction", path=self.crawl_log_path)
            self.logger, _ = self.prod_meta_log.start_log()

    def get_product_type_urls(self):
        """[summary]
        """
        drv  = self.open_browser()
        drv.get(self.base_url)
        cats = drv.find_elements_by_class_name("css-1t5gbpr")
        cat_urls = []
        for c in cats:
            cat_urls.append((c.get_attribute("href").split("/")[-1], c.get_attribute("href")))
            self.logger.info(str.encode(f'Category:- name:{c.get_attribute("href").split("/")[-1]}, \
                                          url:{c.get_attribute("href")}', "utf-8", "ignore"))
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
                    self.logger.info(str.encode(f'SubCategory:- name:{s.get_attribute("href").split("/")[-1]},\
                                                  url:{s.get_attribute("href")}', "utf-8", "ignore"))
            else:
                sub_cat_urls.append((cat_name, cat_url.split('/')[-1], cat_url))
        product_type_urls = []
        for su in sub_cat_urls:
            cat_name = su[0]
            sub_cat_name = su[1]
            sub_cat_url = su[2]
            drv.get(sub_cat_url)
            time.sleep(3)
            product_types = drv.find_elements_by_class_name('css-h6ss0r')
            if len(product_types)>0:
                for item in product_types:
                    product_type_urls.append((cat_name, sub_cat_name, item.get_attribute("href").split("/")[-1],
                                              item.get_attribute("href")))
                    self.logger.info(str.encode(f'ProductType:- name:{item.get_attribute("href").split("/")[-1]},\
                                                  url:{item.get_attribute("href")}', "utf-8", "ignore"))
            else:
                product_type_urls.append((cat_name, sub_cat_name, sub_cat_url.split('/')[-1], sub_cat_url))
        df = pd.DataFrame(product_type_urls, columns = ['category_raw', 'sub_category_raw', 'product_type', 'url'])
        df['scraped'] = 'N'
        df.to_feather(self.metadata_path/'sph_product_cat_subcat_structure')
        drv.close()
        df.drop_duplicates(subset='product_type', inplace=True)
        df.reset_index(inplace=True, drop=True)
        df.to_feather(self.metadata_path/f'sph_product_type_urls_to_extract')
        return df

    def download_metadata(self,fresh_start):
        """[summary]

        Arguments:
            fresh_start {[type]} -- [description]
        """

        product_meta_data = []

        def fresh_ext():
            """[summary]
            """
            product_type_urls = self.get_product_type_urls()
            # progress tracker: captures scraped and error desc 
            progress_tracker = pd.DataFrame(index=product_type_urls.index, columns=['product_type', 'scraped', 'error_desc'])
            progress_tracker.scraped = 'N'
            return product_type_urls, progress_tracker

        if fresh_start:
            self.logger.info('Starting Fresh Extraction.')
            product_type_urls, progress_tracker = fresh_ext()
        else:
            if Path(self.metadata_path/'sph_product_type_urls_to_extract').exists():
                try:
                    progress_tracker = pd.read_feather(self.metadata_path/'sph_metadata_progress_tracker')
                except ArrowIOError:
                    raise MeiyumeException(f"File sph_product_type_urls_to_extract can't be located in the path {self.metadata_path}.\
                                             Please put progress file in the correct path or start fresh extraction.")
                product_type_urls = pd.read_feather(self.metadata_path/'sph_product_type_urls_to_extract')
                if sum(progress_tracker.scraped=='N')>0:
                    self.logger.info('Continuing Metadata Extraction From Last Run.')
                    product_type_urls = product_type_urls[product_type_urls.index.isin(progress_tracker.index[progress_tracker.scraped=='N'].values.tolist())]
                else:
                    self.logger.info('Previous Run Was Complete. Starting Fresh Extraction.')
                    product_type_urls, progress_tracker = fresh_ext()
            else:
                self.logger.info('URL File Not Found. Starting Fresh Extraction.')
                product_type_urls, progress_tracker = fresh_ext()

        drv  = self.open_browser()
        for pt in product_type_urls.index:
            cat_name = product_type_urls.loc[pt,'category_raw']
            #sub_cat_name = product_type_urls.loc[pt,'sub_category_raw']
            product_type = product_type_urls.loc[pt,'product_type']
            product_type_link = product_type_urls.loc[pt,'url']

            progress_tracker.loc[pt,'product_type'] = product_type

            if 'best-selling' in product_type or 'new' in product_type:
                progress_tracker.loc[pt,'scraped'] = 'NA'
                continue
            
            drv.get(product_type_link)
            time.sleep(5)
            #click and close welcome forms
            try:
                drv.find_element_by_xpath('/html/body/div[8]/div/div/div[1]/div/div/button').click()
            except:
                pass
            try:
                drv.find_element_by_xpath('/html/body/div[5]/div/div/div/div[1]/div/div/button').click()
            except:
                pass
            #sort by NEW products
            try:
                drv.find_element_by_class_name('css-1gw67j0').click()
                drv.find_element_by_xpath('//*[@id="cat_sort_menu"]/button[3]').click()
                time.sleep(2)
            except:
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType {product_type} cannot sort by NEW.(page link: {product_type_link})', 'utf-8', 'ignore'))
                pass
            #load all the products
            self.scroll_down_page(drv)
            #check whether on the first page of product type
            try:
                current_page = drv.find_element_by_class_name('css-x544ax').text
            except NoSuchElementException:
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType {product_type} has\
                only one page of products.(page link: {product_type_link})', 'utf-8', 'ignore'))
                one_page = True
                current_page = 1
            except:
                product_type_urls.loc[pt,'scraped'] = 'NA' 
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType {product_type} page not found.(page link: {product_type_link})', 'utf-8', 'ignore'))
            else:
                #get a list of all available pages
                one_page = False
                pages =  []
                for page in drv.find_elements_by_class_name('css-1f9ivf5'):
                    pages.append(page.text)

            #start getting product form each page
            while True: 
                cp = 0
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type}\
                                  getting product from page {current_page}.(page link: {product_type_link})','utf-8', 'ignore')) 
                time.sleep(3)
                products = drv.find_elements_by_class_name('css-12egk0t')
                for p in products:
                    time.sleep(3)
                    try:
                       product_name = p.find_element_by_class_name('css-ix8km1').get_attribute('aria-label')
                    except NoSuchElementException or StaleElementReferenceException:
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(p)} metadata extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                        continue
                    try:
                        new_f = p.find_element_by_class_name("css-8o71lk").text
                        product_new_flag = 'NEW'
                    except NoSuchElementException or StaleElementReferenceException:
                        product_new_flag = ''
                        # self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                        #                              product {products.index(p)} product_new_flag extraction failed.\
                        #                         (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    try:
                        product_page = p.find_element_by_class_name('css-ix8km1').get_attribute('href')
                    except NoSuchElementException or StaleElementReferenceException:
                        product_page = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(p)} product_page extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    try:
                        brand = p.find_element_by_class_name('css-ktoumz').text
                    except NoSuchElementException or StaleElementReferenceException:
                        brand = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(p)} brand extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    try:
                        rating = p.find_element_by_class_name('css-1adflzz').get_attribute('aria-label')
                    except NoSuchElementException or StaleElementReferenceException:
                        rating = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(p)} rating extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    try:
                        price = p.find_element_by_class_name('css-68u28a').text
                    except NoSuchElementException or StaleElementReferenceException:
                        price = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                      product {products.index(p)} price extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))

                    d = {"product_name":product_name,"product_page":product_page,"brand":brand,"price":price,"rating":rating,
                         "category":cat_name,"product_type": product_type, "new_flag":product_new_flag, "complete_scrape_flag":"N",
                         "timestamp": time.strftime("%Y-%m-%d-%H-%M")}
                    cp += 1
                    self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                 Product: {product_name} - {cp} extracted successfully.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    product_meta_data.append(d)

                if one_page:
                    break
                else:
                    if int(current_page) == int(pages[-1]):
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} extraction complete.\
                                                    (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                        break
                    else:
                        try:
                            drv.find_element_by_css_selector('body > div.css-o44is > div.css-138ub37 > div > div > div >\
                                                            div.css-1o80i28 > div > main > div.css-1aj5qq4 > div > div.css-1cepc9v >\
                                                            div.css-6su6fj > nav > ul > button').click()
                            time.sleep(5)
                            self.scroll_down_page(drv)
                            current_page = drv.find_element_by_class_name('css-x544ax').text
                        except:
                            self.logger.info(str.encode(f'Page navigation issue occurred for Category: {cat_name} - \
                                                          ProductType: {product_type} (page_link: {product_type_link} \
                                                          - page_no: {current_page})', 'utf-8', 'ignore'))
                            break

            if len(product_meta_data)>0:
                product_meta_df = pd.DataFrame(product_meta_data)
                product_meta_df.to_feather(self.curnet_progress_path/f'sph_prod_meta_extract_progress_{product_type}_{time.strftime("%Y-%m-%d-%H%M%S")}')
                self.logger.info(f'Completed till IndexPosition: {pt} - ProductType: {product_type}. (URL:{product_type_link})')
                progress_tracker.loc[pt,'scraped'] = 'Y'
                progress_tracker.to_feather(self.metadata_path/'sph_metadata_progress_tracker')
                product_meta_data = []
        self.logger.info('Metadata Extraction Complete')
        print('Metadata Extraction Complete')
        #self.progress_monitor.info('Metadata Extraction Complete')
        drv.close()

    def extract(self, fresh_start=False, delete_progress=True, clean=True):
        """[summary]

        Keyword Arguments:
            fresh_start {bool} -- [description] (default: {False})
            delete_progress {bool} -- [description] (default: {True})
        Returns: 
        """
        self.download_metadata(fresh_start)
        self.logger.info('Creating Combined Metadata File')
        files = [f for f in self.curnet_progress_path.glob("sph_prod_meta_extract_progress_*")]
        li = [pd.read_feather(file) for file in files]
        metadata_df = pd.concat(li, axis=0, ignore_index=True)
        metadata_df.reset_index(inplace=True, drop=True)
        metadata_df['source'] = self.source
        file_name = f'sph_product_metadata_all_{time.strftime("%Y-%m-%d")}'
        metadata_df.to_feather(self.metadata_path/file_name)
        self.logger.info(f'Metadata file created. Please look for file sph_product_metadata_all in path {self.metadata_path}')
        print(f'Metadata file created. Please look for file sph_product_metadata_all in path {self.metadata_path}')
        if delete_progress:
            print('Deleting Progress Files')
            shutil.rmtree(f'{self.metadata_path}\\current_progress', ignore_errors=True)
            self.logger.info('Progress files deleted')
        if clean:
            cleaner = Cleaner()
            metadata_df_clean_no_cat = cleaner.clean_data(data=metadata_df, file_name=file_name)
            self.logger.info('Metadata Cleaned and Removed Duplicates for Details Extraction.')
        self.logger.handlers.clear()
        self.prod_meta_log.stop_log()
        return metadata_df

class Detail(Sephora):
    """
    [summary]

    Arguments:
        Browser {[type]} -- [description]
    """
    def __init__(self, driver_path, path=Path.cwd(), show=True, log=True):
        """[summary]

        Arguments:
            driver_path {[type]} -- [description]

        Keyword Arguments:
            path {[type]} -- [description] (default: {Path.cwd()})
            show {bool} -- [description] (default: {True})
            log {bool} -- [description] (default: {True})
        """
        super().__init__(driver_path=driver_path, show=show, path=path, data_def='detail')
        self.curnet_progress_path = self.detail_path/'current_progress'
        self.curnet_progress_path.mkdir(parents=True, exist_ok=True)
        #set logger
        if log:
            self.prod_detail_log = Logger("sph_prod_detail_extraction",
                                           path=self.crawl_log_path)
            self.logger, _ = self.prod_detail_log.start_log()

    def extract(self, start_idx=None, end_idx=None, list_of_index=None, fresh_start=False, delete_progress=False, clean=True, n_workers=5):
        """[summary]

        Keyword Arguments:
            start_idx {[type]} -- [description] (default: {None})
            end_idx {[type]} -- [description] (default: {None})
            list_of_index {[type]} -- [description] (default: {None})
            fresh_start {bool} -- [description] (default: {False})
            delete_progress {bool} -- [description] (default: {False})
            clean {bool} -- [description] (default: {True})
        """
        def fresh():
            list_of_files = self.metadata_clean_path.glob('no_cat_cleaned_sph_product_metadata_all*')
            self.meta = pd.read_feather(max(list_of_files, key=os.path.getctime))[['prod_id', 'product_name', 'product_page']]
            self.meta['detail_scraped'] = 'N'

        if fresh_start:
            fresh()
        else:
            if Path(self.detail_path/'sph_detail_progress_tracker').exists():
                self.meta = pd.read_feather(self.detail_path/'sph_detail_progress_tracker')
                if sum(self.meta.detail_scraped=='N')==0:
                    self.fresh()
                    self.logger.info('Last Run was Completed. Starting Fresh Extraction.')
                self.logger.info('Continuing Detail Extraction From Last Run.')
            else:
                fresh()
                self.logger.info('Detail Progress Tracker does not exist. Starting Fresh Extraction.')

        #set list or range of product indices to crawl
        if list_of_index: lst = list_of_index
        elif start_idx and end_idx is None: lst = range(start_idx, len(self.meta))
        elif start_idx is None and end_idx: lst = range(0, end_idx)
        elif start_idx is not None and end_idx is not None: lst = range(start_idx, end_idx)
        else: lst = range(len(self.meta))
        print(lst)

        #By default the code will with 5 concurrent threads. you can change this behaviour by changing n_workers
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.download_detail, list(chunks(lst, len(lst)//5)))

        self.download_detail(lst=lst)

        self.logger.info('Creating Combined Detail File')
        det_li = []
        self.bad_det_li = []
        detail_files = [f for f in self.curnet_progress_path.glob("sph_prod_detail_extract_progress_*")]
        for file in detail_files:
            try: df = pd.read_csv(file)
            except: self.bad_det_li.append(file)
            else: det_li.append(df)

        detail_df = pd.concat(det_li, axis=0, ignore_index=True)
        detail_df.reset_index(inplace=True, drop=True)
        detail_df.to_csv(self.detail_path/f'sph_product_detail_all_{time.strftime("%Y-%m-%d")}', index=None)

        self.logger.info('Creating Combined Item File')
        item_li = []
        self.bad_item_li = []
        item_files = [f for f in self.curnet_progress_path.glob("sph_prod_item_extract_progress_*")]
        for file in item_files:
            try: idf = pd.read_csv(file)
            except: self.bad_item_li.append(file)
            else: item_li.append(idf)
        item_df = pd.concat(item_li, axis=0, ignore_index=True)
        item_df.reset_index(inplace=True, drop=True)
        item_df.to_csv(self.detail_path/f'sph_product_item_all_{time.strftime("%Y-%m-%d")}', index=None)

        self.logger.info(f'Detail and Item files created. Please look for file sph_product_detail_all and sph_product_item_all in path {self.detail_path}')
        print(f'Detail and Item files created. Please look for file sph_product_detail_all and sph_product_item_all in path {self.detail_path}')

        if delete_progress:
            print('Deleting Progress Files')
            shutil.rmtree(f'{self.detail_path}\\current_progress', ignore_errors=True)
            self.logger.info('Progress files deleted')

        self.logger.handlers.clear()
        self.prod_detail_log.stop_log()

    def download_detail(self, lst):
        """[summary]

        Arguments:
            lst {[type]} -- [description]
        """
        detail_data = []
        item_data = []
        item_df = pd.DataFrame(columns=['prod_id','product_name','item_name','item_size','item_price','item_ingredients'])

        def store_data_refresh_mem(detail_data, item_df):
            """[summary]

            Arguments:
                detail_data {[type]} -- [description]
                item_df {[type]} -- [description]
            """
            pd.DataFrame(detail_data).to_csv(self.curnet_progress_path/f'sph_prod_detail_extract_progress_{time.strftime("%Y-%m-%d-%H%M%S")}', index=None)
            detail_data = []
            item_df.reset_index(inplace=True, drop=True)
            item_df.to_csv(self.curnet_progress_path/f'sph_prod_item_extract_progress_{time.strftime("%Y-%m-%d-%H%M%S")}', index=None)
            item_df = pd.DataFrame(columns=['prod_id','product_name','item_name','item_size','item_price','item_ingredients'])
            self.meta.to_feather(self.detail_path/'sph_detail_progress_tracker')

        def get_product_attributes():
            """[summary]
            """
            #get all the variation of product
            product_variety = []
            try:
                product_variety = drv.find_elements_by_class_name('css-1j1jwa4')
            except:
                product_variety.append(drv.find_element_by_class_name('css-cl742e'))
            else:
                try:
                    product_variety.append(drv.find_element_by_class_name('css-cl742e'))
                except:
                    pass

            product_attributes = []

            if len(product_variety)>0:
                for typ in product_variety:
                    try:
                        typ.click()
                    except:
                        continue
                    time.sleep(2)
                    item_name, item_size, item_price, item_ingredients = get_item_attributes(multi_variety=True, typ=typ)
                    product_attributes.append({"prod_id":prod_id, "product_name":product_name, "item_name":item_name, "item_size":item_size, "item_price":item_price, "item_ingredients":item_ingredients})
            else:
                item_name, item_size, item_price, item_ingredients = get_item_attributes()
                product_attributes.append({"prod_id":prod_id, "product_name":product_name, "item_name":item_name, "item_size":item_size, "item_price":item_price, "item_ingredients":item_ingredients})

            return product_attributes

        def get_item_attributes(multi_variety=False, typ=None):
            """[summary]

            Keyword Arguments:
                multi_variety {bool} -- [description] (default: {False})
                typ {[type]} -- [description] (default: {None})
            """
            item_price = drv.find_element_by_class_name('css-14hdny6 ').text

            if multi_variety:
                try:
                    item_name = typ.get_attribute('aria-label')
                except NoSuchElementException or StaleElementReferenceException:
                    item_name = ""
                    self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) item_name does not exist.', 'utf-8', 'ignore'))
            else:
                item_name = ""

            try:
                item_size = drv.find_element_by_xpath("/html/body/div[2]/div[5]/main/div[2]/div[1]/div/div/div[2]/div[1]/div[1]/div[1]/span").text
            except NoSuchElementException:
                item_size = ""
                self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) item_size does not exist.', 'utf-8', 'ignore'))

            #get all tabs
            prod_tabs = []
            prod_tabs.append(drv.find_element_by_class_name('css-jpw3l4'))
            prod_tabs.extend(drv.find_elements_by_class_name('css-1r1pql5'))

            tab_names = []
            for t in prod_tabs:
                tab_names.append(t.text.lower())

            if 'ingredients' in tab_names:
                if len(tab_names) ==5:
                    try:
                        tab_num = 2
                        ing_button = drv.find_element_by_id(f'tab{tab_num}')
                        ing_button.click()
                        item_ing = drv.find_element_by_xpath(f'//*[@id="tabpanel{tab_num}"]/div').text
                    except:
                        item_ing = ""
                        self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) item_ingredients extraction failed', 'utf-8', 'ignore'))
                elif len(tab_names) <5:
                    try:
                        tab_num = 1
                        ing_button = drv.find_element_by_id(f'tab{tab_num}')
                        ing_button.click()
                        item_ing = drv.find_element_by_xpath(f'//*[@id="tabpanel{tab_num}"]/div').text
                    except:
                        item_ing = ""
                        self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) item_ingredients extraction failed.', 'utf-8', 'ignore'))
            else:
                item_ing = ""
                self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) item_ingredients does not exist.', 'utf-8', 'ignore'))

            return item_name, item_size, item_price, item_ing

        drv = self.open_browser()

        for prod in self.meta.index[self.meta.index.isin(lst)]:
            if self.meta.loc[prod, 'detail_scraped'] in ['Y','NA']:
                continue
            prod_id = self.meta.loc[prod, 'prod_id']
            product_name = self.meta.loc[prod, 'product_name']
            product_page = self.meta.loc[prod, 'product_page']

            #open product page
            drv.get(product_page)
            time.sleep(2)

            #close popup windows
            try:
                 drv.find_element_by_xpath('/html/body/div[8]/div/div/div[1]/div/div/button').click()
            except:
                pass
            try:
                drv.find_element_by_xpath('/html/body/div[5]/div/div/div/div[1]/div/div/button').click()
            except:
                pass

            #check product page is valid and exists
            try:
                drv.find_element_by_class_name('css-14hdny6').text
            except NoSuchElementException:
                self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) no longer exists in the previously fetched link.(link:{product_page})', 'utf-8', 'ignore'))
                self.meta.loc[prod, 'detail_scraped'] = 'NA'
                continue

            #get all product info tabs such as how-to-use, about-brand, ingredients
            prod_tabs = []
            prod_tabs.append(drv.find_element_by_class_name('css-jpw3l4'))
            prod_tabs.extend(drv.find_elements_by_class_name('css-1r1pql5'))

            tab_names = []
            for t in prod_tabs:
                tab_names.append(t.text.lower())

            #no. of votes
            try:
                votes = drv.find_element_by_xpath('/html/body/div[2]/div[5]/main/div[2]/div[1]/div/div/div[2]/div[1]/div[1]/div[2]/div[2]/span/span').text
            except NoSuchElementException:
                votes = ""
                self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) votes does not exist.', 'utf-8', 'ignore'))

            # product details
            if 'detail' in tab_names:
                try:
                    webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()
                    webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()
                    tab_num = tab_names.index('details')
                    #random button click
                    drv.find_element_by_id(f'tab{1}').click()
                    detail_button = drv.find_element_by_id(f'tab{tab_num}')
                    detail_button.click()
                    time.sleep(1)
                    detail = drv.find_element_by_xpath(f'//*[@id="tabpanel{tab_num}"]/div').text
                except NoSuchElementException:
                    detail = ""
                    self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) product detail extraction failed', 'utf-8', 'ignore'))
            else:
                detail = ""
                self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) product detail does not exist.', 'utf-8', 'ignore'))

            #how to use
            if 'how to use' in tab_names:
                try:
                    webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()
                    webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()
                    tab_num = tab_names.index('how to use')
                    how_to_use_button = drv.find_element_by_id(f'tab{tab_num}')
                    how_to_use_button.click()
                    time.sleep(1)
                    how_to_use = drv.find_element_by_xpath(f'//*[@id="tabpanel{tab_num}"]/div').text
                except NoSuchElementException:
                    how_to_use = ""
                    self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) how_to_use extraction failed', 'utf-8', 'ignore'))
            else:
                how_to_use = ""
                self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) how_to_use does not exist.', 'utf-8', 'ignore'))

            #about the brand
            if 'about the brand' in tab_names:
                try:
                    webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()
                    webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()
                    tab_num = tab_names.index('about the brand')
                    about_the_brand_button = drv.find_element_by_id(f'tab{tab_num}')
                    about_the_brand_button.click()
                    time.sleep(1)
                    about_the_brand = drv.find_element_by_xpath(f'//*[@id="tabpanel{tab_num}"]/div').text
                except NoSuchElementException:
                    about_the_brand = ""
                    self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) about_the_brand extraction failed', 'utf-8', 'ignore'))
            else:
                about_the_brand = ""
                self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) about_the_brand does not exist.', 'utf-8', 'ignore'))

            #no. of reviews
            self.scroll_down_page(drv, h2=0.4)

            try:
                reviews = int(drv.find_element_by_class_name('css-mzsag6').text.split()[0])
            except NoSuchElementException:
                reviews = ""
                self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) reviews does not exist.', 'utf-8', 'ignore'))

            try:
                rating_distribution = drv.find_element_by_class_name('css-960eb6').text.split('\n')
            except:
                rating_distribution = ""
                self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) rating_distribution does not exist.', 'utf-8', 'ignore'))

            try:
                would_recommend = drv.find_element_by_class_name('css-1heqyf0').text
            except:
                would_recommend = ""
                self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) would_recommend does not exist.', 'utf-8', 'ignore'))

            product_attributes = pd.DataFrame(get_product_attributes())
            item_df = pd.concat([item_df, pd.DataFrame(product_attributes)], axis=0)

            detail_data.append({'prod_id':prod_id, 'product_name':product_name, 'abt_product':detail, 'how_to_use':how_to_use, 'abt_brand':about_the_brand,
                                'reviews':reviews, 'votes':votes, 'rating_dist':rating_distribution, 'would_recommend':would_recommend})
            item_data.append(product_attributes)
            self.logger.info(str.encode(f'product: {product_name} (prod_id: {prod_id}) details extracted successfully', 'utf-8', 'ignore'))
            self.meta.loc[prod, 'detail_scraped'] = 'Y'
            
            if prod !=0 and prod%20==0:
                if len(detail_data)>0:
                    store_data_refresh_mem(detail_data, item_df)
        store_data_refresh_mem(detail_data, item_df)
        drv.close()
        self.logger.info(f'Detail Extraction Complete for start_idx: (lst[0]) to end_idx: {lst[-1]}. Or for list of values.')
