
"""[summary]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from typing import *
import concurrent.futures
import os
import shutil
import time
from datetime import datetime, timedelta
import warnings
from pathlib import Path
from typing import *
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
from selenium.webdriver.common.action_chains import ActionChains

from meiyume.cleaner_plus import Cleaner
from meiyume.utils import Browser, Logger, MeiyumeException, Sephora, chunks, convert_ago_to_date

warnings.simplefilter(action='ignore', category=FutureWarning)


class Metadata(Sephora):
    """[summary]

    Arguments:
        Sephora {[type]} -- [description]
    """
    base_url = "https://www.sephora.com"
    info = tldextract.extract(base_url)
    source = info.registered_domain

    @classmethod
    def update_base_url(cls, url: str)->None:
        """[summary]

        Arguments:
            url {[type]} -- [description]
        """
        cls.base_url = url
        cls.info = tldextract.extract(cls.base_url)
        cls.source = cls.info.registered_domain

    def __init__(self, log: bool = True, path: Path = Path.cwd()):
        """[summary]

        Arguments:
            driver_path {[type]} -- [description]

        Keyword Arguments:
            log {bool} -- [description] (default: {True})
            path {[type]} -- [description] (default: {Path.cwd()})
            show {bool} -- [description] (default: {True})
        """
        super().__init__(path=path, data_def='meta')
        self.current_progress_path = self.metadata_path/'current_progress'
        self.current_progress_path.mkdir(parents=True, exist_ok=True)
        if log:
            self.prod_meta_log = Logger(
                "sph_prod_metadata_extraction", path=self.crawl_log_path)
            self.logger, _ = self.prod_meta_log.start_log()

    def get_product_type_urls(self, open_headless: bool)-> pd.DataFrame:
        """[summary]
        """
        # create webdriver instance
        drv = self.open_browser(
            open_headless=open_headless, open_with_proxy_server=False, path=self.metadata_path)

        drv.get(self.base_url)
        cats = drv.find_elements_by_class_name("css-1b5x40g")
        cat_urls = []
        for c in cats:
            cat_urls.append((c.get_attribute("href").split("/")
                             [-1], c.get_attribute("href")))
            self.logger.info(str.encode(f'Category:- name:{c.get_attribute("href").split("/")[-1]}, \
                                          url:{c.get_attribute("href")}', "utf-8", "ignore"))

        sub_cat_urls = []
        for cu in cat_urls:
            cat_name = cu[0]
            cat_url = cu[1]
            drv.get(cat_url)
            time.sleep(8)
            sub_cats = drv.find_elements_by_class_name("css-1leg7f4")
            sub_cats.extend(drv.find_elements_by_class_name("css-or7ouu"))
            if len(sub_cats) > 0:
                for s in sub_cats:
                    sub_cat_urls.append((cat_name, s.get_attribute(
                        "href").split("/")[-1], s.get_attribute("href")))
                    self.logger.info(str.encode(f'SubCategory:- name:{s.get_attribute("href").split("/")[-1]},\
                                                  url:{s.get_attribute("href")}', "utf-8", "ignore"))
            else:
                sub_cat_urls.append(
                    (cat_name, cat_url.split('/')[-1], cat_url))

        product_type_urls = []
        for su in sub_cat_urls:
            cat_name = su[0]
            sub_cat_name = su[1]
            sub_cat_url = su[2]
            drv.get(sub_cat_url)
            time.sleep(6)
            product_types = drv.find_elements_by_class_name('css-1lp76tk')
            if len(product_types) > 0:
                for item in product_types:
                    product_type_urls.append((cat_name, sub_cat_name, item.get_attribute("href").split("/")[-1],
                                              item.get_attribute("href")))
                    self.logger.info(str.encode(f'ProductType:- name:{item.get_attribute("href").split("/")[-1]},\
                                                  url:{item.get_attribute("href")}', "utf-8", "ignore"))
            else:
                product_type_urls.append(
                    (cat_name, sub_cat_name, sub_cat_url.split('/')[-1], sub_cat_url))

        df = pd.DataFrame(product_type_urls, columns=[
                          'category_raw', 'sub_category_raw', 'product_type', 'url'])

        df_clean = pd.DataFrame(sub_cat_urls, columns=[
                                'category_raw', 'product_type', 'url'])
        df_clean['sub_category_raw'] = 'CLEAN'
        df_clean = df_clean[(df_clean.url.apply(
            lambda x: True if 'clean' in x else False)) & (df_clean.product_type != 'cleanser')]

        df_vegan = pd.DataFrame(sub_cat_urls, columns=[
                                'category_raw', 'product_type', 'url'])
        df_vegan['sub_category_raw'] = 'VEGAN'
        df_vegan = df_vegan[df_vegan.url.apply(
            lambda x: True if 'vegan' in x.lower() else False)]

        df = pd.concat([df, df_clean, df_vegan], axis=0)
        df.reset_index(inplace=True, drop=True)
        df.to_feather(self.metadata_path/'sph_product_cat_subcat_structure')
        drv.quit()

        df.drop_duplicates(subset='url', inplace=True)
        df.reset_index(inplace=True, drop=True)
        df['scraped'] = 'N'
        df.to_feather(self.metadata_path/f'sph_product_type_urls_to_extract')
        return df

    def get_metadata(self, product_type_urls: pd.DataFrame, progress_tracker: pd.DataFrame, open_headless: bool)->None:
        """[summary]

        Arguments:
            fresh_start {[type]} -- [description]
        """

        product_meta_data = []

        # create webdriver instance
        drv = self.open_browser(
            open_headless=open_headless, open_with_proxy_server=False, path=self.metadata_path)

        for pt in product_type_urls.index:
            cat_name = product_type_urls.loc[pt, 'category_raw']

            # sub_cat_name = product_type_urls.loc[pt,'sub_category_raw']
            product_type = product_type_urls.loc[pt, 'product_type']
            product_type_link = product_type_urls.loc[pt, 'url']

            progress_tracker.loc[pt, 'product_type'] = product_type

            if 'best-selling' in product_type or 'new' in product_type:
                progress_tracker.loc[pt, 'scraped'] = 'NA'
                continue

            drv.get(product_type_link)
            time.sleep(8)
            # click and close welcome forms
            webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()
            time.sleep(1)
            webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()

            # sort by new products (required to get all new products properly)
            try:
                drv.find_element_by_class_name('css-qv1cc8').click()
                button = drv.find_element_by_xpath(
                    '//*[@id="cat_sort_menu"]/button[3]')
                drv.implicitly_wait(5)
                ActionChains(drv).move_to_element(
                    button).click(button).perform()
                time.sleep(5)
            except Exception:
                self.logger.info(str.encode(
                    f'Category: {cat_name} - ProductType {product_type} cannot sort by NEW.(page link: {product_type_link})', 'utf-8', 'ignore'))
                pass

            # load all the products
            self.scroll_down_page(drv)

            # check whether on the first page of product type
            try:
                current_page = drv.find_element_by_class_name(
                    'css-nnom91').text
            except NoSuchElementException:
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType {product_type} has\
                only one page of products.(page link: {product_type_link})', 'utf-8', 'ignore'))
                one_page = True
                current_page = 1
            except Exception:
                product_type_urls.loc[pt, 'scraped'] = 'NA'
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType {product_type} page not found.(page link: {product_type_link})',
                                            'utf-8', 'ignore'))
            else:
                # get a list of all available pages
                one_page = False
                pages = []
                for page in drv.find_elements_by_class_name('css-17n3p1l'):
                    pages.append(page.text)

            # start getting product form each page
            while True:
                cp = 0
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type}\
                                  getting product from page {current_page}.(page link: {product_type_link})',
                                            'utf-8', 'ignore'))
                time.sleep(6)
                products = drv.find_elements_by_class_name('css-12egk0t')
                for p in products:
                    time.sleep(6)
                    try:
                        product_name = p.find_element_by_class_name(
                            'css-ix8km1').get_attribute('aria-label')
                    except NoSuchElementException or StaleElementReferenceException:
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(p)} metadata extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})',
                                                    'utf-8', 'ignore'))
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
                        product_page = p.find_element_by_class_name(
                            'css-ix8km1').get_attribute('href')
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
                        rating = p.find_element_by_class_name(
                            'css-ychh9y').get_attribute('aria-label')
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

                    product_data_dict = {"product_name": product_name, "product_page": product_page, "brand": brand, "price": price,
                                         "rating": rating, "category": cat_name, "product_type": product_type, "new_flag": product_new_flag,
                                         "complete_scrape_flag": "N", "meta_date": time.strftime("%Y-%m-%d")}
                    cp += 1
                    self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                 Product: {product_name} - {cp} extracted successfully.\
                                                (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    product_meta_data.append(product_data_dict)

                if one_page:
                    break
                elif int(current_page) == int(pages[-1]):
                    self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} extraction complete.\
                                                (page_link: {product_type_link} - page_no: {current_page})',
                                                'utf-8', 'ignore'))
                    break
                else:
                    # go to next page
                    try:
                        next_page_button = drv.find_element_by_class_name(
                            'css-4ktkov')
                        ActionChains(drv).move_to_element(
                            next_page_button).click(next_page_button).perform()
                        time.sleep(10)
                        self.scroll_down_page(drv)
                        current_page = drv.find_element_by_class_name(
                            'css-nnom91').text
                    except Exception:
                        self.logger.info(str.encode(f'Page navigation issue occurred for Category: {cat_name} - \
                                                        ProductType: {product_type} (page_link: {product_type_link} \
                                                        - page_no: {current_page})', 'utf-8', 'ignore'))
                        break

            if len(product_meta_data) > 0:
                product_meta_df = pd.DataFrame(product_meta_data)
                product_meta_df.to_feather(
                    self.current_progress_path/f'sph_prod_meta_extract_progress_{product_type}_{time.strftime("%Y-%m-%d-%H%M%S")}')
                self.logger.info(
                    f'Completed till IndexPosition: {pt} - ProductType: {product_type}. (URL:{product_type_link})')
                progress_tracker.loc[pt, 'scraped'] = 'Y'
                progress_tracker.to_feather(
                    self.metadata_path/'sph_metadata_progress_tracker')
                product_meta_data = []
        self.logger.info('Metadata Extraction Complete')
        print('Metadata Extraction Complete')

        # self.progress_monitor.info('Metadata Extraction Complete')
        drv.quit()

    def extract(self,  open_headless: bool = False, download: bool = True, fresh_start: bool = False, clean: bool = True,
                delete_progress: bool = True)->None:
        """[summary]

        Keyword Arguments:
            fresh_start {bool} -- [description] (default: {False})
            delete_progress {bool} -- [description] (default: {True})
        Returns:
        """
        def fresh():
            """[summary]
            """
            product_type_urls = self.get_product_type_urls(open_headless)
            # progress tracker: captures scraped and error desc
            progress_tracker = pd.DataFrame(index=product_type_urls.index, columns=[
                                            'product_type', 'scraped', 'error_desc'])
            progress_tracker.scraped = 'N'
            return product_type_urls, progress_tracker

        if fresh_start:
            self.logger.info('Starting Fresh Extraction.')
            product_type_urls, progress_tracker = fresh()
        else:
            if Path(self.metadata_path/'sph_product_type_urls_to_extract').exists():
                try:
                    progress_tracker = pd.read_feather(
                        self.metadata_path/'sph_metadata_progress_tracker')
                except ArrowIOError:
                    raise MeiyumeException(f"File sph_product_type_urls_to_extract can't be located in the path {self.metadata_path}.\
                                             Please put progress file in the correct path or start fresh extraction.")
                product_type_urls = pd.read_feather(
                    self.metadata_path/'sph_product_type_urls_to_extract')
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

        if download:
            self.get_metadata(product_type_urls=product_type_urls,
                              progress_tracker=progress_tracker, open_headless=open_headless)

        self.logger.info('Creating Combined Metadata File')
        files = [f for f in self.current_progress_path.glob(
            "sph_prod_meta_extract_progress_*")]
        li = [pd.read_feather(file) for file in files]
        metadata_df = pd.concat(li, axis=0, ignore_index=True)
        metadata_df.reset_index(inplace=True, drop=True)
        metadata_df['source'] = self.source
        filename = f'sph_product_metadata_all_{time.strftime("%Y-%m-%d")}'
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
            _ = cleaner.clean(
                data=self.metadata_path/filename)
            self.logger.info(
                'Metadata Cleaned and Removed Duplicates for Details Extraction.')

        self.logger.handlers.clear()
        self.prod_meta_log.stop_log()
        # return metadata_df


class Detail(Sephora):
    """Detail [summary]

    [extended_summary]

    Args:
        Sephora ([type]): [description]
    """

    def __init__(self, path: Path = Path.cwd(), log: bool = True):
        """__init__ [summary]

        [extended_summary]

        Args:
            path (Path, optional): [description]. Defaults to Path.cwd().
            log (bool, optional): [description]. Defaults to True.
        """
        super().__init__(path=path, data_def='detail')
        self.current_progress_path = self.detail_path/'current_progress'
        self.current_progress_path.mkdir(parents=True, exist_ok=True)
        # set logger
        if log:
            self.prod_detail_log = Logger("sph_prod_detail_extraction",
                                          path=self.crawl_log_path)
            self.logger, _ = self.prod_detail_log.start_log()

    def download_detail(self, open_headless: bool, indices: list,
                        detail_data=[], item_data=[],
                        item_df=pd.DataFrame(columns=['prod_id', 'product_name', 'item_name',
                                                      'item_size', 'item_price',
                                                      'item_ingredients']))->None:
        """download_detail [summary]

        [extended_summary]

        Args:
            open_headless (bool): [description]
            indices (list): [description]
            detail_data (list, optional): [description]. Defaults to [].
            item_data (list, optional): [description]. Defaults to [].
            item_df ([type], optional): [description]. Defaults to pd.DataFrame(columns=['prod_id', 'product_name', 'item_name',
                                                                                'item_size', 'item_price', 'item_ingredients']).
        """
        def store_data_refresh_mem(detail_data: list, item_df: pd.DataFrame)->Tuple[list, pd.DataFrame]:
            """[summary]

            Arguments:
                detail_data {[type]} -- [description]
                item_df {[type]} -- [description]

            Returns:
                [type] -- [description]
            """
            pd.DataFrame(detail_data).to_csv(self.current_progress_path /
                                             f'sph_prod_detail_extract_progress_{time.strftime("%Y-%m-%d-%H%M%S")}.csv',
                                             index=None)
            item_df.reset_index(inplace=True, drop=True)
            item_df.to_csv(self.current_progress_path /
                           f'sph_prod_item_extract_progress_{time.strftime("%Y-%m-%d-%H%M%S")}.csv',
                           index=None)
            item_df = pd.DataFrame(columns=[
                                   'prod_id', 'product_name', 'item_name', 'item_size', 'item_price', 'item_ingredients'])
            self.meta.to_feather(
                self.detail_path/'sph_detail_progress_tracker')
            return [], item_df

        def get_item_attributes(drv: webdriver.Chrome, product_name: str, prod_id: str, use_button: bool = False,
                                multi_variety: bool = False, typ=None, )->Tuple[str, str, str, str]:
            """get_item_attributes [summary]

            [extended_summary]

            Args:
                drv (webdriver.Chrome): [description]
                product_name (str): [description]
                prod_id (str): [description]
                use_button (bool, optional): [description]. Defaults to False.
                multi_variety (bool, optional): [description]. Defaults to False.
                typ ([type], optional): [description]. Defaults to None.

            Returns:
                Tuple[str, str, str, str]: [description]
            """
            # drv # type: webdriver.Chrome
            item_price = drv.find_element_by_class_name('css-slwsq8').text

            if multi_variety:
                try:
                    if use_button:
                        item_name = typ.find_element_by_tag_name(
                            'button').get_attribute('aria-label')
                    else:
                        item_name = typ.get_attribute('aria-label')
                except NoSuchElementException or StaleElementReferenceException:
                    item_name = ""
                    self.logger.info(str.encode(
                        f'product: {product_name} (prod_id: {prod_id}) item_name does not exist.', 'utf-8', 'ignore'))
            else:
                item_name = ""

            try:
                item_size = drv.find_element_by_class_name('css-v7k1z0').text
            except NoSuchElementException:
                item_size = ""
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) item_size does not exist.', 'utf-8', 'ignore'))

            # get all tabs
            prod_tabs = []
            prod_tabs = drv.find_elements_by_class_name('css-1h2tppq ')
            prod_tabs.extend(drv.find_elements_by_class_name('css-18ih2xz'))

            tab_names = []
            for t in prod_tabs:
                tab_names.append(t.text.lower())

            if 'ingredients' in tab_names:
                if len(tab_names) == 5:
                    try:
                        tab_num = 2
                        ing_button = drv.find_element_by_id(f'tab{tab_num}')
                        ActionChains(drv).move_to_element(
                            ing_button).click(ing_button).perform()
                        item_ing = drv.find_element_by_xpath(
                            f'//*[@id="tabpanel{tab_num}"]/div').text
                    except Exception:
                        item_ing = ""
                        self.logger.info(str.encode(
                            f'product: {product_name} (prod_id: {prod_id}) item_ingredients extraction failed',
                            'utf-8', 'ignore'))
                elif len(tab_names) == 4:
                    try:
                        tab_num = 1
                        ing_button = drv.find_element_by_id(f'tab{tab_num}')
                        ActionChains(drv).move_to_element(
                            ing_button).click(ing_button).perform()
                        item_ing = drv.find_element_by_xpath(
                            f'//*[@id="tabpanel{tab_num}"]/div').text
                    except Exception:
                        item_ing = ""
                        self.logger.info(str.encode(
                            f'product: {product_name} (prod_id: {prod_id}) item_ingredients extraction failed.',
                            'utf-8', 'ignore'))
                elif len(tab_names) < 4:
                    try:
                        tab_num = 0
                        ing_button = drv.find_element_by_id(f'tab{tab_num}')
                        ActionChains(drv).move_to_element(
                            ing_button).click(ing_button).perform()
                        item_ing = drv.find_element_by_xpath(
                            f'//*[@id="tabpanel{tab_num}"]/div').text
                    except Exception:
                        item_ing = ""
                        self.logger.info(str.encode(
                            f'product: {product_name} (prod_id: {prod_id}) item_ingredients extraction failed.',
                            'utf-8', 'ignore'))
            else:
                item_ing = ""
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) item_ingredients does not exist.', 'utf-8', 'ignore'))

            return item_name, item_size, item_price, item_ing

        def get_product_attributes(drv: webdriver.Chrome, product_name: str, prod_id: str)->list:
            """get_product_attributes [summary]

            [extended_summary]

            Args:
                drv (webdriver.Chrome): [description]
                product_name (str): [description]
                prod_id (str): [description]

            Returns:
                list: [description]
            """
            # get all the variation of product
            product_variety = []
            try:
                product_variety = drv.find_elements_by_class_name(
                    'css-1j1jwa4')
                product_variety.extend(
                    drv.find_elements_by_class_name('css-cl742e'))
                use_button = False
            except Exception:
                pass
            try:
                if len(product_variety) < 1:
                    product_variety = drv.find_elements_by_class_name(
                        'css-5jqxch')
                    use_button = True
            except Exception:
                pass

            product_attributes = []

            if len(product_variety) > 0:
                for typ in product_variety:
                    try:
                        ActionChains(drv).move_to_element(
                            typ).click(typ).perform()
                    except Exception:
                        continue
                    time.sleep(8)
                    item_name, item_size, item_price, item_ingredients = get_item_attributes(drv, product_name, prod_id,
                                                                                             multi_variety=True, typ=typ,
                                                                                             use_button=use_button)
                    product_attributes.append({"prod_id": prod_id, "product_name": product_name,
                                               "item_name": item_name, "item_size": item_size,
                                               "item_price": item_price, "item_ingredients": item_ingredients})
            else:
                item_name, item_size, item_price, item_ingredients = get_item_attributes(drv,
                                                                                         product_name, prod_id)
                product_attributes.append({"prod_id": prod_id, "product_name": product_name, "item_name": item_name,
                                           "item_size": item_size, "item_price": item_price, "item_ingredients": item_ingredients})

            return product_attributes

        def get_first_review_date(drv: webdriver.Chrome)->str:
            """get_first_review_date [summary]

            [extended_summary]

            Args:
                drv (webdriver.Chrome): [description]

            Returns:
                str: [description]
            """
            drv.find_element_by_id('review_filter_sort_trigger').click()
            for btn in drv.find_elements_by_class_name('css-1khw9z2'):
                if btn.text.lower() == 'oldest':
                    ActionChains(drv).move_to_element(
                        btn).click(btn).perform()
                    break
            time.sleep(1.5)
            rev = drv.find_elements_by_class_name('css-1ecc607')[2:]
            return convert_ago_to_date(rev[0].find_element_by_class_name('css-1t84k9w').text)

        for prod in self.meta.index[self.meta.index.isin(indices)]:
            #  ignore already extracted products
            if self.meta.loc[prod, 'detail_scraped'] in ['Y', 'NA']:
                continue
            # create webdriver
            use_proxy = np.random.choice([True, False])
            if use_proxy:
                drv = self.open_browser(open_headless=open_headless, open_with_proxy_server=True,
                                        path=self.detail_path)
            else:
                drv = self.open_browser(open_headless=open_headless, open_with_proxy_server=False,
                                        path=self.detail_path)

            prod_id = self.meta.loc[prod, 'prod_id']
            product_name = self.meta.loc[prod, 'product_name']
            product_page = self.meta.loc[prod, 'product_page']

            # open product page
            drv.get(product_page)
            time.sleep(8)

            # close popup windows
            ActionChains(drv).send_keys(Keys.ESCAPE).perform()
            time.sleep(1)
            ActionChains(drv).send_keys(Keys.ESCAPE).perform()

            # check product page is valid and exists
            try:
                drv.find_element_by_class_name('css-slwsq8').text
            except NoSuchElementException:
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) no longer exists in the previously fetched link.\
                        (link:{product_page})', 'utf-8', 'ignore'))
                self.meta.loc[prod, 'detail_scraped'] = 'NA'
                continue

            # get all product info tabs such as how-to-use, about-brand, ingredients
            prod_tabs = []
            prod_tabs = drv.find_elements_by_class_name('css-1h2tppq ')
            prod_tabs.extend(drv.find_elements_by_class_name('css-18ih2xz'))

            tab_names = []
            for t in prod_tabs:
                tab_names.append(t.text.lower())

            # no. of votes
            try:
                votes = drv.find_element_by_xpath(
                    '/html/body/div[3]/div[5]/main/div[2]/div[1]/div/div/div[2]/div[1]/div[1]/div[2]/div[2]/span/span').text
            except NoSuchElementException:
                votes = ""
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) votes does not exist.', 'utf-8', 'ignore'))

            # product details
            if 'details' in tab_names:
                try:
                    webdriver.ActionChains(drv).send_keys(
                        Keys.ESCAPE).perform()
                    webdriver.ActionChains(drv).send_keys(
                        Keys.ESCAPE).perform()
                    tab_num = tab_names.index('details')
                    detail_button = drv.find_element_by_id(f'tab{tab_num}')
                    try:
                        time.sleep(1)
                        ActionChains(drv).move_to_element(
                            detail_button).click(detail_button).perform()
                    except ElementClickInterceptedException:
                        details = ""
                    else:
                        details = drv.find_element_by_xpath(
                            f'//*[@id="tabpanel{tab_num}"]/div').text
                except NoSuchElementException:
                    details = ""
                    self.logger.info(str.encode(
                        f'product: {product_name} (prod_id: {prod_id}) product detail extraction failed', 'utf-8', 'ignore'))
            else:
                detail = ""
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) product detail does not exist.', 'utf-8', 'ignore'))

            # how to use
            if 'how to use' in tab_names:
                try:
                    webdriver.ActionChains(drv).send_keys(
                        Keys.ESCAPE).perform()
                    webdriver.ActionChains(drv).send_keys(
                        Keys.ESCAPE).perform()
                    tab_num = tab_names.index('how to use')
                    how_to_use_button = drv.find_element_by_id(f'tab{tab_num}')
                    try:
                        time.sleep(1)
                        ActionChains(drv).move_to_element(
                            how_to_use_button).click(how_to_use_button).perform()
                    except ElementClickInterceptedException:
                        how_to_use = ""
                    else:
                        how_to_use = drv.find_element_by_xpath(
                            f'//*[@id="tabpanel{tab_num}"]/div').text
                except NoSuchElementException:
                    how_to_use = ""
                    self.logger.info(str.encode(
                        f'product: {product_name} (prod_id: {prod_id}) how_to_use extraction failed', 'utf-8', 'ignore'))
            else:
                how_to_use = ""
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) how_to_use does not exist.', 'utf-8', 'ignore'))

            # about the brand
            if 'about the brand' in tab_names:
                try:
                    webdriver.ActionChains(drv).send_keys(
                        Keys.ESCAPE).perform()
                    webdriver.ActionChains(drv).send_keys(
                        Keys.ESCAPE).perform()
                    tab_num = tab_names.index('about the brand')
                    about_the_brand_button = drv.find_element_by_id(
                        f'tab{tab_num}')
                    try:
                        time.sleep(1)
                        ActionChains(drv).move_to_element(
                            about_the_brand_button).click(about_the_brand_button).perform()
                    except ElementClickInterceptedException:
                        about_the_brand = ""
                    else:
                        about_the_brand = drv.find_element_by_xpath(
                            f'//*[@id="tabpanel{tab_num}"]/div').text
                except NoSuchElementException:
                    about_the_brand = ""
                    self.logger.info(str.encode(
                        f'product: {product_name} (prod_id: {prod_id}) about_the_brand extraction failed', 'utf-8', 'ignore'))
            else:
                about_the_brand = ""
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) about_the_brand does not exist.', 'utf-8', 'ignore'))

            self.scroll_down_page(drv, h2=0.4)
            # click no. of reviews
            drv.find_element_by_class_name('css-1pjru6n').click()

            try:
                first_review_date = get_first_review_date(drv)
            except Exception:
                first_review_date = ""
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) first_review_date scrape failed.', 'utf-8', 'ignore'))

            try:
                reviews = int(drv.find_element_by_class_name(
                    'css-tc6qfq').text.split()[0])
            except NoSuchElementException:
                reviews = 0
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) reviews does not exist.', 'utf-8', 'ignore'))

            try:
                rating_distribution = drv.find_element_by_class_name(
                    'css-960eb6').text.split('\n')
            except Exception:
                rating_distribution = ""
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) rating_distribution does not exist.', 'utf-8', 'ignore'))

            try:
                would_recommend = drv.find_element_by_class_name(
                    'css-k9ne19').text
            except Exception:
                would_recommend = ""
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) would_recommend does not exist.', 'utf-8', 'ignore'))

            product_attributes = pd.DataFrame(
                get_product_attributes(drv, product_name, prod_id))
            item_df = pd.concat(
                [item_df, pd.DataFrame(product_attributes)], axis=0)

            detail_data.append({'prod_id': prod_id, 'product_name': product_name, 'abt_product': details,
                                'how_to_use': how_to_use, 'abt_brand': about_the_brand,
                                'reviews': reviews, 'votes': votes, 'rating_dist': rating_distribution,
                                'would_recommend': would_recommend, 'first_review_date': first_review_date})
            item_data.append(product_attributes)
            self.logger.info(str.encode(
                f'product: {product_name} (prod_id: {prod_id}) details extracted successfully', 'utf-8', 'ignore'))
            self.meta.loc[prod, 'detail_scraped'] = 'Y'
            if prod != 0 and prod % 10 == 0:
                if len(detail_data) > 0:
                    detail_data, item_df = store_data_refresh_mem(
                        detail_data, item_df)
            drv.quit()
        # save the final file
        detail_data, item_df = store_data_refresh_mem(detail_data, item_df)

        self.logger.info(
            f'Detail Extraction Complete for start_idx: (indices[0]) to end_idx: {indices[-1]}. Or for list of values.')

    def extract(self, metadata: pd.DataFrame, download: bool = True, open_headless: bool = False,
                start_idx: Optional[int] = None, end_idx: Optional[int] = None,
                list_of_index=None, fresh_start: bool = False, delete_progress: bool = False,
                clean: bool = True, n_workers: int = 5):
        """extract [summary]

        [extended_summary]

        Args:
            metadata (pd.DataFrame): [description]
            download (bool, optional): [description]. Defaults to True.
            open_headless (bool, optional): [description]. Defaults to False.
            start_idx (Optional[int], optional): [description]. Defaults to None.
            end_idx (Optional[int], optional): [description]. Defaults to None.
            list_of_index (Optional[list], optional): [description]. Defaults to None.
            fresh_start (bool, optional): [description]. Defaults to False.
            delete_progress (bool, optional): [description]. Defaults to False.
            clean (bool, optional): [description]. Defaults to True.
            n_workers (int, optional): [description]. Defaults to 5.
        """
        '''
        change metadata read logic
        add logic to look for metadata in a folder path
        if metadata is found in the folder path
        detail data crawler is triggered
        '''
        def fresh():
            list_of_files = self.metadata_clean_path.glob(
                'no_cat_cleaned_sph_product_metadata_all*')
            self.meta = pd.read_feather(max(list_of_files, key=os.path.getctime))[
                ['prod_id', 'product_name', 'product_page', 'meta_date']]
            self.meta['detail_scraped'] = 'N'
        if download:
            if fresh_start:
                fresh()
            else:
                if Path(self.detail_path/'sph_detail_progress_tracker').exists():
                    self.meta = pd.read_feather(
                        self.detail_path/'sph_detail_progress_tracker')
                    if sum(self.meta.detail_scraped == 'N') == 0:
                        fresh()
                        self.logger.info(
                            'Last Run was Completed. Starting Fresh Extraction.')
                    else:
                        self.logger.info(
                            'Continuing Detail Extraction From Last Run.')
                else:
                    fresh()
                    self.logger.info(
                        'Detail Progress Tracker does not exist. Starting Fresh Extraction.')

            # set list or range of product indices to crawl
            if list_of_index:
                indices = list_of_index
            elif start_idx and end_idx is None:
                indices = range(start_idx, len(self.meta))
            elif start_idx is None and end_idx:
                indices = range(0, end_idx)
            elif start_idx is not None and end_idx is not None:
                indices = range(start_idx, end_idx)
            else:
                indices = range(len(self.meta))
            print(indices)

            if list_of_index:
                self.download_detail(indices=list_of_index,
                                     open_headless=open_headless)
            else:  # By default the code will with 5 concurrent threads. you can change this behaviour by changing n_workers
                lst_of_lst = list(chunks(indices, len(indices)//n_workers))
                # detail_Data and item_data are lists of empty lists so that each namepace of function call will have its separate detail_data
                # list to strore scraped dictionaries. will save memory(ram/hard-disk) consumption. will stop data duplication
                detail_data = [[] for i in lst_of_lst]  # type: List
                item_data = [[] for i in lst_of_lst]  # type: List
                item_df = [pd.DataFrame(columns=['prod_id', 'product_name', 'item_name',
                                                 'item_size', 'item_price', 'item_ingredients'])
                           for i in lst_of_lst]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # but each of the function namespace will be modifying only one metadata tracing file so that progress saving
                    # is tracked correctly. else multiple progress tracker file will be created with difficulty to combine correct
                    # progress information
                    executor.map(self.download_detail, lst_of_lst,
                                 detail_data, item_data, item_df)

        self.logger.info('Creating Combined Detail File')
        det_li = []
        self.bad_det_li = []
        detail_files = [f for f in self.current_progress_path.glob(
            "sph_prod_detail_extract_progress_*")]
        for file in detail_files:
            try:
                df = pd.read_csv(file)
            except Exception:
                self.bad_det_li.append(file)
            else:
                det_li.append(df)

        detail_df = pd.concat(det_li, axis=0, ignore_index=True)
        detail_df.drop_duplicates(inplace=True)
        detail_df.reset_index(inplace=True, drop=True)
        detail_df['meta_date'] = self.meta.meta_date.max()
        detail_filename = f'sph_product_detail_all_{time.strftime("%Y-%m-%d")}.csv'
        detail_df.to_csv(self.detail_path/detail_filename, index=None)
        # detail_df.to_feather(self.detail_path/detail_filename)
        self.logger.info('Creating Combined Item File')
        item_li = []
        self.bad_item_li = []
        item_files = [f for f in self.current_progress_path.glob(
            "sph_prod_item_extract_progress_*")]
        for file in item_files:
            try:
                idf = pd.read_csv(file)
            except Exception:
                self.bad_item_li.append(file)
            else:
                item_li.append(idf)

        item_dataframe = pd.concat(item_li, axis=0, ignore_index=True)
        item_dataframe.drop_duplicates(inplace=True)
        item_dataframe.reset_index(inplace=True, drop=True)
        item_dataframe['meta_date'] = self.meta.meta_date.max()
        item_filename = f'sph_product_item_all_{time.strftime("%Y-%m-%d")}.csv'
        item_dataframe.to_csv(self.detail_path/item_filename, index=None)
        # item_df.to_feather(self.detail_path/item_filename)

        self.logger.info(
            f'Detail and Item files created. Please look for file sph_product_detail_all and\
                 sph_product_item_all in path {self.detail_path}')
        print(
            f'Detail and Item files created. Please look for file sph_product_detail_all and\
                 sph_product_item_all in path {self.detail_path}')

        if clean:
            cleaner = Cleaner()
            self.detail_clean_df = cleaner.clean(
                self.detail_path/detail_filename)
            self.item_clean_df, self.ing_clean_df = cleaner.clean(
                self.detail_path/item_filename)

        if delete_progress:
            shutil.rmtree(
                f'{self.detail_path}\\current_progress', ignore_errors=True)
            self.logger.info('Progress files deleted')

        self.logger.handlers.clear()
        self.prod_detail_log.stop_log()


class Review(Sephora):
    """[summary]

    Arguments:
        Sephora {[type]} -- [description]
    """

    def __init__(self, log: bool = True, path: Path = Path.cwd()):
        """__init__ [summary]

        [extended_summary]

        Args:
            log (bool, optional): [description]. Defaults to True.
        """
        super().__init__(path=path, data_def='review')
        self.current_progress_path = self.review_path/'current_progress'
        self.current_progress_path.mkdir(parents=True, exist_ok=True)
        if log:
            self.prod_review_log = Logger(
                "sph_prod_review_extraction", path=self.crawl_log_path)
            self.logger, _ = self.prod_review_log.start_log()

    def get_reviews(self, indices: list, open_headless: bool = False,
                    review_data: list = [], incremental: bool = True):
        """get_reviews [summary]

        [extended_summary]

        Args:
            indices (list): [description]
            open_headless (bool, optional): [description]. Defaults to False.
            review_data (list, optional): [description]. Defaults to [].
            incremental (bool, optional): [description]. Defaults to True.
        """
        def store_data_refresh_mem(review_data: list)->list:
            """[summary]

            Arguments:
                review_data {[type]} -- [description]

            Returns:
                [type] -- [description]
            """
            pd.DataFrame(review_data).to_csv(self.current_progress_path /
                                             f'sph_prod_review_extract_progress_{time.strftime("%Y-%m-%d-%H%M%S")}.csv',
                                             index=None)

            self.meta.to_csv(
                self.review_path/'sph_review_progress_tracker.csv', index=None)
            return []

        for prod in self.meta.index[self.meta.index.isin(indices)]:
            if self.meta.loc[prod, 'review_scraped'] in ['Y', 'NA']:
                continue
            prod_id = self.meta.loc[prod, 'prod_id']
            product_name = self.meta.loc[prod, 'product_name']
            product_page = self.meta.loc[prod, 'product_page']

            # change it to correct table read from database
            last_scraped_review_date = self.meta.loc[prod,
                                                     'last_scraped_review_date']
            # print(last_scraped_review_date)

            use_proxy = np.random.choice([True, False])
            if use_proxy:
                drv = self.open_browser(open_headless=open_headless, open_with_proxy_server=True,
                                        path=self.detail_path)
            else:
                drv = self.open_browser(open_headless=open_headless, open_with_proxy_server=False,
                                        path=self.detail_path)

            drv.get(product_page)
            time.sleep(8)

            # close popup windows
            webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()
            time.sleep(1)
            webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()

            self.scroll_down_page(drv, speed=20, h2=0.6)

            try:
                no_of_reviews = int(drv.find_element_by_class_name(
                    'css-tc6qfq').text.split()[0])
            except Exception:
                self.logger.info(str.encode(f'Product: {product_name} prod_id: {prod_id} reviews extraction failed.\
                                              Either product has no reviews or not\
                                              available for sell currently.(page: {product_page})', 'utf-8', 'ignore'))
                no_of_reviews = 0
                self.meta.loc[prod, 'review_scraped'] = 'NA'
                drv.quit()
                # print('in except - continue')
                continue

            # print(no_of_reviews)
            # drv.find_element_by_class_name('css-2rg6q7').click()
            if incremental and last_scraped_review_date != '':
                for n in range(no_of_reviews//6):
                    if n > 250:
                        break

                    time.sleep(0.4)
                    revs = drv.find_elements_by_class_name(
                        'css-1ecc607')[2:]

                    try:
                        if pd.to_datetime(convert_ago_to_date(revs[-1].find_element_by_class_name('css-1t84k9w').text),
                                          infer_datetime_format=True)\
                                < pd.to_datetime(last_scraped_review_date, infer_datetime_format=True):
                            # print('breaking incremental')
                            break
                    except Exception:
                        try:
                            if pd.to_datetime(convert_ago_to_date(revs[-2].find_element_by_class_name('css-1t84k9w').text),
                                              infer_datetime_format=True)\
                                    < pd.to_datetime(last_scraped_review_date, infer_datetime_format=True):
                                # print('breaking incremental')
                                break
                        except Exception:
                            self.logger.info(str.encode(f'Product: {product_name} prod_id: {prod_id} \
                                                         last_scraped_review_date to current review date \
                                                         comparision failed.(page: {product_page})',
                                                        'utf-8', 'ignore'))
                            # print('in second except block')
                            continue
                    try:
                        show_more_review_button = drv.find_element_by_class_name(
                            'css-frqcui')
                        ActionChains(drv).move_to_element(
                            show_more_review_button).click(show_more_review_button).perform()
                    except Exception:
                        webdriver.ActionChains(drv).send_keys(
                            Keys.ESCAPE).perform()
                        time.sleep(0.5)
                        webdriver.ActionChains(drv).send_keys(
                            Keys.ESCAPE).perform()
                        try:
                            show_more_review_button = drv.find_element_by_class_name(
                                'css-frqcui')
                            ActionChains(drv).move_to_element(
                                show_more_review_button).click(show_more_review_button).perform()
                        except Exception:
                            pass

            else:
                print('inside get all reviews')
                # 6 because for click sephora shows 6 reviews. additional 25 no. of clicks for buffer.
                for n in range(no_of_reviews//6+25):
                    '''
                    code will stop after getting 1800 reviews of one particular product
                    when crawling all reviews. By default it will get latest 1800 reviews.
                    then in subsequent incremental runs it will get al new reviews on weekly basis
                    '''
                    if n >= 300:  # 200:
                        break
                    time.sleep(0.4)
                    # close any opened popups by escape
                    try:
                        # drv.find_element_by_css_selector('#ratings-reviews > div.css-ilr0fu > button').click()
                        show_more_review_button = drv.find_element_by_class_name(
                            'css-frqcui')
                        ActionChains(drv).move_to_element(
                            show_more_review_button).click(show_more_review_button).perform()
                    except Exception:
                        webdriver.ActionChains(drv).send_keys(
                            Keys.ESCAPE).perform()
                        time.sleep(0.5)
                        webdriver.ActionChains(drv).send_keys(
                            Keys.ESCAPE).perform()
                        try:
                            show_more_review_button = drv.find_element_by_class_name(
                                'css-frqcui')
                            ActionChains(drv).move_to_element(
                                show_more_review_button).click(show_more_review_button).perform()
                        except Exception:
                            webdriver.ActionChains(drv).send_keys(
                                Keys.ESCAPE).perform()
                            try:
                                show_more_review_button = drv.find_element_by_class_name(
                                    'css-frqcui')
                                ActionChains(drv).move_to_element(
                                    show_more_review_button).click(show_more_review_button).perform()
                            except Exception:
                                webdriver.ActionChains(drv).send_keys(
                                    Keys.ESCAPE).perform()
                                time.sleep(1)
                                webdriver.ActionChains(drv).send_keys(
                                    Keys.ESCAPE).perform()
                                try:
                                    show_more_review_button = drv.find_element_by_class_name(
                                        'css-frqcui')
                                    ActionChains(drv).move_to_element(
                                        show_more_review_button).click(show_more_review_button).perform()
                                except Exception:
                                    if n < (no_of_reviews//6):
                                        self.logger.info(str.encode(f'Product: {product_name} - prod_id {prod_id} breaking click next review loop.\
                                                                    [total_reviews:{no_of_reviews} loaded_reviews:{n}]\
                                                                    (page link: {product_page})', 'utf-8', 'ignore'))
                                        self.logger.info(str.encode(f'Product: {product_name} - prod_id {prod_id} cant load all reviews. \
                                            Check click next 6 reviews\
                                                                    code section(page link: {product_page})', 'utf-8', 'ignore'))
                                    break

            webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()
            time.sleep(1)
            webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()

            product_reviews = drv.find_elements_by_class_name(
                'css-1ecc607')[2:]

            # print('starting extraction')
            r = 0
            for rev in product_reviews:
                r += 1
                if r % 20 == 0:
                    webdriver.ActionChains(drv).send_keys(
                        Keys.ESCAPE).perform()
                    time.sleep(1)
                    webdriver.ActionChains(drv).send_keys(
                        Keys.ESCAPE).perform()
                try:
                    review_text = rev.find_element_by_class_name(
                        'css-7rv8g1').text
                except NoSuchElementException or StaleElementReferenceException:
                    continue

                try:
                    review_date = convert_ago_to_date(
                        rev.find_element_by_class_name('css-1t84k9w').text)
                    if pd.to_datetime(review_date, infer_datetime_format=True) < \
                            pd.to_datetime(last_scraped_review_date, infer_datetime_format=True):
                        continue
                except NoSuchElementException or StaleElementReferenceException:
                    review_date = ''

                try:
                    review_title = rev.find_element_by_class_name(
                        'css-85k18d').text
                except NoSuchElementException or StaleElementReferenceException:
                    review_title = ''

                try:
                    product_variant = rev.find_element_by_class_name(
                        'css-gjgyg3').text
                except Exception:
                    product_variant = ''

                try:
                    user_rating = rev.find_element_by_class_name(
                        'css-3z5ot7').get_attribute('aria-label')
                except NoSuchElementException or StaleElementReferenceException:
                    user_rating = ''

                try:
                    user_attribute = [{'_'.join(u.lower().split()[0:-1]): u.lower().split()[-1]}
                                      for u in rev.find_element_by_class_name('css-j5yt83').text.split('\n')]
                    # user_attribute = []
                    # for u in rev.find_elements_by_class_name('css-j5yt83'):
                    #     user_attribute.append(
                    #         {'_'.join(u.text.lower().split()[0:-1]): u.text.lower().split()[-1]})
                except NoSuchElementException or StaleElementReferenceException:
                    user_attribute = []

                try:
                    recommend = rev.find_element_by_class_name(
                        'css-1nv53ng').text
                except NoSuchElementException or StaleElementReferenceException:
                    recommend = ''

                try:
                    helpful = rev.find_element_by_class_name('css-bsl1yh').text
                    # helpful = []
                    # for h in rev.find_elements_by_class_name('css-39esqn'):
                    #     helpful.append(h.text)
                except NoSuchElementException or StaleElementReferenceException:
                    helpful = []

                review_data.append({'prod_id': prod_id, 'product_name': product_name,
                                    'user_attribute': user_attribute, 'product_variant': product_variant,
                                    'review_title': review_title, 'review_text': review_text,
                                    'review_rating': user_rating, 'recommend': recommend,
                                    'review_date': review_date,   'helpful': helpful})
            drv.quit()
            self.meta.loc[prod, 'review_scraped'] = 'Y'
            review_data = store_data_refresh_mem(review_data)
            if not incremental:
                self.logger.info(str.encode(
                    f'Product_name: {product_name} prod_id:{prod_id} reviews extracted successfully.(total_reviews: {no_of_reviews}, \
                    extracted_reviews: {len(product_reviews)}, page: {product_page})', 'utf-8', 'ignore'))
            else:
                self.logger.info(str.encode(
                    f'Product_name: {product_name} prod_id:{prod_id} new reviews extracted successfully.\
                        (no_of_new_extracted_reviews: {len(product_reviews)},\
                         page: {product_page})', 'utf-8', 'ignore'))
        # save the final review file
        review_data = store_data_refresh_mem(review_data)

    def extract(self, metadata: pd.DataFrame, open_headless: bool = False, download: bool = True,
                start_idx: Optional[int] = None, end_idx: Optional[int] = None, list_of_index=None,
                fresh_start: bool = False, incremental: bool = True, delete_progress: bool = False,
                clean: bool = True, n_workers: int = 5,)->None:
        """extract [summary]

        [extended_summary]

        Args:
            metadata (pd.DataFrame): [description]
            open_headless (bool, optional): [description]. Defaults to False.
            download (bool, optional): [description]. Defaults to True.
            start_idx (Optional[int], optional): [description]. Defaults to None.
            end_idx (Optional[int], optional): [description]. Defaults to None.
            list_of_index (Optional[list], optional): [description]. Defaults to None.
            fresh_start (bool, optional): [description]. Defaults to False.
            incremental (bool, optional): [description]. Defaults to True.
            delete_progress (bool, optional): [description]. Defaults to False.
            clean (bool, optional): [description]. Defaults to True.
            n_workers (int, optional): [description]. Defaults to 5.
        """
        def fresh():
            list_of_files = self.metadata_clean_path.glob(
                'no_cat_cleaned_sph_product_metadata_all*')
            self.meta = pd.read_feather(max(list_of_files, key=os.path.getctime))[
                ['prod_id', 'product_name', 'product_page', 'meta_date', 'last_scraped_review_date']]
            self.meta.last_scraped_review_date.fillna('', inplace=True)
            self.meta['review_scraped'] = 'N'

        if download:
            if fresh_start:
                fresh()
            else:
                if Path(self.review_path/'sph_review_progress_tracker.csv').exists():
                    self.meta = pd.read_csv(
                        self.review_path/'sph_review_progress_tracker.csv')
                    if sum(self.meta.review_scraped == 'N') == 0:
                        fresh()
                        self.logger.info(
                            'Last Run was Completed. Starting Fresh Extraction.')
                    else:
                        self.logger.info(
                            'Continuing Review Extraction From Last Run.')
                else:
                    fresh()
                    self.logger.info(
                        'Review Progress Tracker not found. Starting Fresh Extraction.')

            # set list or range of product indices to crawl
            if list_of_index:
                indices = list_of_index
            elif start_idx and end_idx is None:
                indices = range(start_idx, len(self.meta))
            elif start_idx is None and end_idx:
                indices = range(0, end_idx)
            elif start_idx is not None and end_idx is not None:
                indices = range(start_idx, end_idx)
            else:
                indices = range(len(self.meta))
            # print(indices)

            if list_of_index:
                self.get_reviews(
                    indices=list_of_index, incremental=incremental, open_headless=open_headless)
            else:  # By default the code will with 5 concurrent threads. you can change this behaviour by changing n_workers
                lst_of_lst = list(
                    chunks(indices, len(indices)//n_workers))  # type: list
                '''
                # review_Data and item_data are lists of empty lists so that each namepace of function call will
                # have its separate detail_data
                # list to strore scraped dictionaries. will save memory(ram/hard-disk) consumption. will stop data duplication
                '''
                review_data = [[] for i in lst_of_lst]  # type: list
                inc_list = [incremental for i in lst_of_lst]  # type: list
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    '''
                    # but each of the function namespace will be modifying only one metadata tracing file so that progress saving
                    # is tracked correctly. else multiple progress tracker file will be created with difficulty to combine correct
                    # progress information
                    '''
                    executor.map(self.get_reviews, lst_of_lst,
                                 review_data, inc_list, )

        self.logger.info('Creating Combined Review File')
        rev_li = []
        self.bad_rev_li = []
        review_files = [f for f in self.current_progress_path.glob(
            "sph_prod_review_extract_progress_*")]
        for file in review_files:
            try:
                df = pd.read_csv(file)
            except Exception:
                self.bad_rev_li.append(file)
            else:
                rev_li.append(df)
        rev_df = pd.concat(rev_li, axis=0, ignore_index=True)
        rev_df.drop_duplicates(inplace=True)
        rev_df.reset_index(inplace=True, drop=True)
        rev_df['meta_date'] = self.meta.meta_date.max()
        review_filename = f'sph_product_review_all_{time.strftime("%Y-%m-%d")}'
        rev_df.to_feather(self.review_path/review_filename)  # , index=None)

        self.logger.info(
            f'Review file created. Please look for file sph_product_review_all in path {self.review_path}')
        print(
            f'Review file created. Please look for file sph_product_review_all in path {self.review_path}')

        if clean:
            cleaner = Cleaner()
            self.review_clean_df = cleaner.clean(
                self.review_path/review_filename)

        if delete_progress:
            shutil.rmtree(
                f'{self.review_path}\\current_progress', ignore_errors=True)
            self.logger.info('Progress files deleted')

        self.logger.handlers.clear()
        self.prod_review_log.stop_log()

        '''
        sort by NEW reviews
        try:
            drv.find_element_by_class_name('css-2rg6q7').click()
            drv.find_element_by_id('review_filter_sort_trigger').click()
            for btn in drv.find_elements_by_class_name('css-a2osvj'):
                if btn.text == 'Newest':
                    drv.find_element_by_id(
                        'review_filter_sort_trigger').click()
                    btn.click()
                    break
        except:
            try:
                drv.find_element_by_id(
                    'review_filter_sort_trigger').click()
                drv.find_element_by_xpath(
                    '/html/body/div[2]/div[5]/main/div[2]/div[2]/div/div[1]/div/div[3]/div[2]/div/div[1]/div/div[2]/div[2]/div/div/div/div[2]/div/span/span').click()
                time.sleep(1)
            except:
                try:
                    drv.find_element_by_id(
                        'review_filter_sort_trigger').click()
                    drv.find_element_by_xpath(
                        '/html/body/div[2]/div[5]/main/div[2]/div[2]/div/div[1]/div/div[3]/div[2]/div/div[1]/div/div[4]/div[2]/div/div/div/div[2]/div/span/span').click()
                    time.sleep(1)
                except:
                    try:
                        drv.find_element_by_class_name(
                            'css-2rg6q7').click()
                        drv.find_element_by_id(
                            'review_filter_sort_trigger').click()
                        drv.find_element_by_xpath(
                            '/html/body/div[2]/div[5]/main/div[2]/div[2]/div/div[1]/div/div[3]/div[2]/div/div[1]/div/div[2]/div[2]/div/div/div/div[2]/div/span/span').click()
                        time.sleep(1)
                    except:
                        try:
                            drv.find_element_by_class_name(
                                'css-2rg6q7').click()
                            drv.find_element_by_id(
                                'review_filter_sort_trigger').click()
                            drv.find_element_by_xpath(
                                '/html/body/div[2]/div[5]/main/div[2]/div[2]/div/div[1]/div/div[3]/div[2]/div/div[1]/div/div[4]/div[2]/div/div/div/div[2]/div/span/span').click()
                            time.sleep(1)
                        except:
                            self.logger.info(str.encode(
                                f'Product: {product_name} - prod_id {prod_id} reviews can not sort by NEW.(page link: {product_page})',
        'utf-8', 'ignore'))
                        try:
                            drv.find_element_by_id('review_filter_sort_trigger').click()
                            drv.find_element_by_css_selector('#review_filter_sort > div > div > div:nth-child(2) > div > span > span').click()
                            time.sleep(1)
                        except:
                            self.logger.info(str.encode(f'Product: {product_name} - prod_id {prod_id} reviews can not sort by NEW.(page link:
        {product_page})', 'utf-8', 'ignore'))
        '''


class Image(Sephora):

    """Image [summary]

    [extended_summary]

    Args:
        Sephora ([type]): [description]
    """

    def __init__(self, path: Path = Path.cwd(), log: bool = True):
        """__init__ [summary]

        [extended_summary]

        Args:
            path (Path, optional): [description]. Defaults to Path.cwd().
            log (bool, optional): [description]. Defaults to True.
        """
        super().__init__(path=path, data_def='image')

        if log:
            self.prod_image_log = Logger(
                "sph_prod_image_extraction", path=self.crawl_log_path)
            self.logger, _ = self.prod_image_log.start_log()

    def get_images(self, indices: list, open_headless: bool):
        """get_images [summary]

        [extended_summary]

        Args:
            indices (list): list of product indices to iterate over
        """

        for prod in self.meta.index[self.meta.index.isin(indices)]:
            if self.meta.loc[prod, 'image_scraped'] in ['Y', 'NA']:
                continue
            prod_id = self.meta.loc[prod, 'prod_id']
            product_name = self.meta.loc[prod, 'product_name']
            product_page = self.meta.loc[prod, 'product_page']
            drv = self.open_browser(
                open_headless=open_headless, open_for_screenshot=True, path=self.image_path)
            drv.get(product_page)
            time.sleep(8)

            # close popup windows
            webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()
            time.sleep(0.5)
            webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()

            try:
                price = drv.find_element_by_class_name('css-slwsq8')
            except Exception:
                self.logger.info(str.encode(f'Product: {product_name} prod_id: {prod_id} image extraction failed.\
                                            Product may not be available for sell currently.(page: {product_page})',
                                            'utf-8', 'ignore'))
                self.meta.loc[prod, 'image_scraped'] = 'NA'
                self.meta.to_csv(
                    self.image_path/'sph_image_progress_tracker.csv', index=None)
                drv.quit()
                continue

            # get image elements
            try:
                images = drv.find_elements_by_class_name('css-od26es')
            except Exception:
                continue

            if len(images) == 0:
                try:
                    images = drv.find_elements_by_class_name('css-od26es')
                except Exception:
                    self.logger.info(str.encode(f'Product: {product_name} prod_id: {prod_id} image extraction failed.\
                                                 (page: {product_page})', 'utf-8', 'ignore'))
                    self.meta.loc[prod, 'image_scraped'] = 'NA'
                    self.meta.to_csv(
                        self.image_path/'sph_image_progress_tracker.csv', index=None)
                    drv.quit()
                    continue
            # get image urls
            sources = [i.find_element_by_tag_name(
                'img').get_attribute('src') for i in images]
            sources = [i.split('?')[0] for i in sources]

            if len(sources) > 4:
                sources = sources[:3]

            self.current_image_path = self.image_path/prod_id
            if not self.current_image_path.exists():
                self.current_image_path.mkdir(parents=True, exist_ok=True)

            # download images
            image_count = 0
            try:
                for src in sources:
                    #     src = i.find_element_by_tag_name('img').get_attribute('src')
                    #     srcset = i.find_element_by_tag_name('img').get_attribute('srcset')
                    drv.get(src)
                    time.sleep(3)
                    image_count += 1
                    image_name = f'{prod_id}_image_{image_count}.jpg'
                    drv.save_screenshot(
                        str(self.current_image_path/image_name))
                self.meta.loc[prod, 'image_scraped'] = 'Y'
                drv.quit()

            except Exception:
                if image_count <= 1:
                    self.logger.info(str.encode(f'Product: {product_name} prod_id: {prod_id} image extraction failed.\
                                                    (page: {product_page})', 'utf-8', 'ignore'))
                    self.meta.loc[prod, 'image_scraped'] = 'NA'
                    self.meta.to_csv(
                        self.image_path/'sph_image_progress_tracker.csv', index=None)
                drv.quit()
                continue

            if prod % 10 == 0:
                self.meta.to_csv(
                    self.image_path/'sph_image_progress_tracker.csv', index=None)
        self.meta.to_csv(
            self.image_path/'sph_image_progress_tracker.csv', index=None)

    def extract(self, start_idx: int = None, end_idx: int = None, list_of_index=None, fresh_start: bool = False,
                n_workers: int = 5, download: bool = True, open_headless: bool = True):
        """extract [summary]

        [extended_summary]

        Args:
            start_idx (int, optional): [description]. Defaults to None.
            end_idx (int, optional): [description]. Defaults to None.
            list_of_index (list, optional): [description]. Defaults to None.
            fresh_start (bool, optional): [description]. Defaults to False.
            n_workers (int, optional): [description]. Defaults to 5.
            download (bool, optional): [description]. Defaults to True.
        """
        def fresh():
            list_of_files = self.metadata_clean_path.glob(
                'no_cat_cleaned_sph_product_metadata_all*')

            self.meta = pd.read_feather(max(list_of_files, key=os.path.getctime))[
                ['prod_id', 'product_name', 'product_page', 'meta_date']]

            self.meta['image_scraped'] = 'N'

        if download:
            if fresh_start:
                fresh()
            else:
                if Path(self.image_path/'sph_image_progress_tracker.csv').exists():
                    self.meta = pd.read_csv(
                        self.image_path/'sph_image_progress_tracker.csv')
                    if sum(self.meta.image_scraped == 'N') == 0:
                        fresh()
                        self.logger.info(
                            'Last Run was Completed. Starting Fresh Extraction.')
                    else:
                        self.logger.info(
                            'Continuing Image Extraction From Last Run.')
                else:
                    fresh()
                    self.logger.info(
                        'Image Progress Tracker not found. Starting Fresh Extraction.')

            self.meta = self.meta[~self.meta.image_scraped.isin(['Y', 'NA'])]
            self.meta.to_csv(
                self.image_path/'sph_image_progress_tracker.csv', index=None)
            self.meta.reset_index(inplace=True, drop=True)

            # set list or range of product indices to crawl
            if list_of_index:
                indices = list_of_index
            elif start_idx and end_idx is None:
                indices = range(start_idx, len(self.meta))
            elif start_idx is None and end_idx:
                indices = range(0, end_idx)
            elif start_idx is not None and end_idx is not None:
                indices = range(start_idx, end_idx)
            else:
                indices = range(len(self.meta))
            # print(indices)
            self.get_images(indices=indices, open_headless=open_headless)

            # if list_of_index:
            #     self.get_images(
            #         indices=list_of_index)
            # else:  # By default the code will with 5 concurrent threads. you can change this behaviour by changing n_workers
            #     lst_of_lst = list(chunks(indices, len(indices)//n_workers))

            #     with concurrent.futures.ThreadPoolExecutor() as executor:
            #         executor.map(self.get_images, lst_of_lst)

        self.logger.info(
            f'Image files are downloaded to product specific folders. \
            Please look for file sph_product_review_all in path {self.image_path}')
        print(
            f'Image files are downloaded to product specific folders. \
            Please look for file sph_product_review_all in path {self.image_path}')

        self.logger.handlers.clear()
        self.prod_image_log.stop_log()
