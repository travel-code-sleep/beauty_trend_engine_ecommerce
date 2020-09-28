
"""[summary]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from typing import *
import sys
import types
import pdb
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
from selenium import webdriver
from selenium.common.exceptions import (ElementClickInterceptedException,
                                        NoSuchElementException,
                                        StaleElementReferenceException,
                                        TimeoutException)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.alert import Alert

from meiyume.cleaner_plus import Cleaner
from meiyume.utils import (Browser, Logger, MeiyumeException, Boots,
                           accept_alert, close_popups, log_exception,
                           chunks, ranges, convert_ago_to_date)

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
    def update_base_url(cls, url: str)->None:
        """update_base_url [summary]

        [extended_summary]

        Args:
            url (str): [description]
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
        self.path = path
        self.current_progress_path = self.metadata_path/'current_progress'
        self.current_progress_path.mkdir(parents=True, exist_ok=True)

        # move old raw and clean files to old folder
        old_metadata_files = list(self.metadata_path.glob(
            'bts_product_metadata_all*'))
        for f in old_metadata_files:
            shutil.move(str(f), str(self.old_metadata_files_path))

        old_clean_metadata_files = os.listdir(self.metadata_clean_path)
        for f in old_clean_metadata_files:
            shutil.move(str(self.metadata_clean_path/f),
                        str(self.old_metadata_clean_files_path))
        # set logger
        if log:
            self.prod_meta_log = Logger(
                "bts_prod_metadata_extraction", path=self.crawl_log_path)
            self.logger, _ = self.prod_meta_log.start_log()

    def get_product_type_urls(self, open_headless: bool, open_with_proxy_server: bool) -> pd.DataFrame:
        """get_product_type_urls [summary]

        [extended_summary]

        Args:
            open_headless (bool): [description]
            open_with_proxy_server (bool): [description]

        Returns:
            pd.DataFrame: [description]
        """
        # create webdriver instance
        drv = self.open_browser(
            open_headless=open_headless, open_with_proxy_server=open_with_proxy_server, path=self.metadata_path)

        drv.get(self.base_url)
        time.sleep(15)
        # click and close welcome forms
        accept_alert(drv, 10)
        close_popups(drv)

        try:
            country_popup = WebDriverWait(drv, 3).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.estores_overlay_content > a:nth-child(1)')))
            pop_close_button = drv.find_element_by_css_selector(
                '.estores_overlay_content > a:nth-child(1)')
            Browser().scroll_to_element(drv, pop_close_button)
            ActionChains(drv).move_to_element(
                pop_close_button).click(pop_close_button).perform()
        except Exception as ex:
            pass

        allowed_categories = ['beauty', 'fragrance',
                              'toiletries', 'mens', 'wellness']
        exclude_categories = ['health-pharmacy', 'advice', 'luxury-beauty-premium-beauty-book-an-appointment',
                              'health-value-packs-and-bundles', 'all-luxury-skincare', 'wellness-supplements', 'travel-essentials',
                              'opticians', 'feminine-hygiene', 'travel-health', 'sunglasses', 'inspiration', 'food-and-drink',
                              'nutrition', 'vitaminsandsupplements', 'condoms-sexual-health', 'mens-health-information',
                              'weightloss', 'menshealth', 'recommended', 'offers', 'all-', 'male-incontinence', 'sustainable-living',
                              'bootsdental', 'back-to-school-and-nursery', 'luxury-beauty-skincare', 'mens-gift-sets',
                              'all-face', 'all-eyes', 'all-lips', 'gift/him/mens-aftershave', 'gift/her/luxury-beauty-gift',
                              'christmas/christmas-3-for-2', 'christmas/gifts-for-her', 'christmas/gifts-for-him',
                              'christmas/advent-calendars', 'beauty-expert-skincare-expert-skincare-shop-all',
                              'black-afro-and-textured-hair-straight-hair', 'black-afro-and-textured-hair-wavy',
                              'black-afro-and-textured-hair-straight-hair',
                              ]
        # Extracting the category-sub category structure
        cat_urls = []
        for i in drv.find_elements_by_css_selector('a[id*="subcategoryLink"]'):
            cat_urls.append(i.get_attribute('href'))

        cat_urls = [u for u in cat_urls if any(
            cat in str(u) for cat in allowed_categories)]
        cat_urls = [u for u in cat_urls if all(
            cat not in str(u) for cat in exclude_categories)]

        prod_type_urls = []

        for url in cat_urls:
            drv.get(url)

            time.sleep(6)
            accept_alert(drv, 10)
            close_popups(drv)

            subcats = drv.find_elements_by_css_selector('div.category-link>a')
            if len(subcats) > 0:
                for i in drv.find_elements_by_css_selector('div.category-link>a'):
                    prod_type_urls.append(i.get_attribute('href'))
            else:
                prod_type_urls.append(url)

        prod_type_urls = [u for u in prod_type_urls if any(
            cat in str(u) for cat in allowed_categories)]
        prod_type_urls = [u for u in prod_type_urls if all(
            cat not in str(u) for cat in exclude_categories)]

        drv.quit()

        df = pd.DataFrame(prod_type_urls, columns=['url'])
        df['dept'], df['category_raw'], df['subcategory_raw'], df['product_type'] = zip(
            *df.url.str.split('/', expand=True).loc[:, 3:].values)

        def set_cat_subcat_ptype(x):
            if x.product_type is None:
                return x.dept, x.category_raw, x.subcategory_raw
            else:
                return x.category_raw, x.subcategory_raw, x.product_type

        df.category_raw, df.subcategory_raw, df.product_type = zip(
            *df.apply(set_cat_subcat_ptype, axis=1))
        df.drop(columns=['dept'], inplace=True)
        df.drop_duplicates(subset='url', inplace=True)
        df.reset_index(inplace=True, drop=True)
        df['scraped'] = 'N'
        df.to_feather(self.metadata_path/f'bts_product_type_urls_to_extract')
        return df

    def get_metadata(self, indices: Union[list, range],
                     open_headless: bool, open_with_proxy_server: bool,
                     randomize_proxy_usage: bool,
                     product_meta_data: list = []):
        """get_metadata [summary]

        [extended_summary]

        Args:
            indices (Union[list, range]): [description]
            open_headless (bool): [description]
            open_with_proxy_server (bool): [description]
            randomize_proxy_usage (bool): [description]
            product_meta_data (list, optional): [description]. Defaults to [].
        """
        for pt in self.product_type_urls.index[self.product_type_urls.index.isin(indices)]:
            cat_name = self.product_type_urls.loc[pt, 'category_raw']
            product_type = self.product_type_urls.loc[pt, 'product_type']
            product_type_link = self.product_type_urls.loc[pt, 'url']

            self.progress_tracker.loc[pt, 'product_type'] = product_type
            # print(self.progress_tracker.loc[pt, 'product_type'])
            # print(product_type_link)
            if 'best-selling' in product_type.lower() or 'new' in product_type.lower():
                self.progress_tracker.loc[pt, 'scraped'] = 'NA'
                # print(self.progress_tracker.loc[pt, 'scraped'])
                continue

            if randomize_proxy_usage:
                use_proxy = np.random.choice([True, False])
            else:
                use_proxy = True
            if open_with_proxy_server:
                # print(use_proxy)
                drv = self.open_browser(open_headless=open_headless, open_with_proxy_server=use_proxy,
                                        path=self.metadata_path)
            else:
                drv = self.open_browser(open_headless=open_headless, open_with_proxy_server=False,
                                        path=self.metadata_path)

            drv.get(product_type_link)
            time.sleep(15)  # 30
            accept_alert(drv, 10)
            close_popups(drv)

            # load all the products
            self.scroll_down_page(drv, h2=0.8, speed=5)
            time.sleep(5)

            try:
                pages = int(drv.find_element_by_css_selector(
                    'div[class*="pageControl number"]').get_attribute("data-pages"))
            except NoSuchElementException as ex:
                log_exception(self.logger,
                              additional_information=f'Prod Type: {product_type}')
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType {product_type} has\
                only one page of products.(page link: {product_type_link})', 'utf-8', 'ignore'))
                pages = 1
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod Type: {product_type}')
                self.progress_tracker.loc[pt, 'scraped'] = 'NA'
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType {product_type}\
                     page not found.(page link: {product_type_link})',
                                            'utf-8', 'ignore'))

            for page in range(pages):
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type}\
                                  getting product from page {page}.(page link: {product_type_link})',
                                            'utf-8', 'ignore'))

                products = drv.find_elements_by_css_selector(
                    'div[class*="estore_product_container"]')

                for product in products:
                    # prod_id = "bts_" + \
                    #     product.get_attribute('data-productid').split(".")[0]
                    self.scroll_to_element(drv, product)
                    try:
                        product_name = product.find_element_by_css_selector(
                            'div.product_name').text
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod Type: {product_type}')
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(product)} metadata extraction failed.\
                                                (page_link: {product_type_link} - page_no: {page})',
                                                    'utf-8', 'ignore'))
                        product_name = ''

                    try:
                        price = product.find_element_by_css_selector(
                            'div.product_price').text
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod Type: {product_type}')
                        price = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                      product {products.index(product)} price extraction failed.\
                                                (page_link: {product_type_link} - page_no: {page})', 'utf-8', 'ignore'))

                    try:
                        discount = product.find_element_by_css_selector(
                            'div.product_savePrice>span').text.split()[-1]
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod Type: {product_type}')
                        discount = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                      product {products.index(product)} no discount/savings.\
                                                (page_link: {product_type_link} - page_no: {page})', 'utf-8', 'ignore'))

                    try:
                        product_page = product.find_element_by_css_selector(
                            'div.product_name>a').get_attribute('href')
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod Type: {product_type}')
                        product_page = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(product)} product_page extraction failed.\
                                                (page_link: {product_type_link} - page_no: {page})', 'utf-8', 'ignore'))

                    try:
                        rating = product.find_element_by_css_selector(
                            'div.product_rating>span').get_attribute('aria-label').split(" ")[0]
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod Type: {product_type}')
                        rating = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(product)} rating extraction failed.\
                                                (page_link: {product_type_link} - page_no: {page})', 'utf-8', 'ignore'))

                    if datetime.now().day < 15:
                        meta_date = f'{time.strftime("%Y-%m")}-01'
                    else:
                        meta_date = f'{time.strftime("%Y-%m")}-15'

                    product_data_dict = {"product_name": product_name, "product_page": product_page, "brand": '',
                                         "price": price, "discount": discount, "rating": rating, "category": cat_name,
                                         "product_type": product_type, "new_flag": '', "meta_date": meta_date}
                    product_meta_data.append(product_data_dict)

                if pages != 1:
                    next_page_button = drv.find_element_by_css_selector(
                        'a[title*="Show next"]')
                    self.scroll_to_element(drv, next_page_button)
                    ActionChains(drv).move_to_element(
                        next_page_button).click(next_page_button).perform()
                    time.sleep(5)
                    accept_alert(drv, 10)
                    close_popups(drv)
                    self.scroll_down_page(drv, h2=0.8, speed=5)
                    time.sleep(5)

            drv.quit()

            if len(product_meta_data) > 0:
                product_meta_df = pd.DataFrame(product_meta_data)
                product_meta_df.to_feather(
                    self.current_progress_path/f'bts_prod_meta_extract_progress_{product_type}_{time.strftime("%Y-%m-%d-%H%M%S")}')
                self.logger.info(
                    f'Completed till IndexPosition: {pt} - ProductType: {product_type}. (URL:{product_type_link})')
                self.progress_tracker.loc[pt, 'scraped'] = 'Y'
                # print(self.progress_tracker.loc[pt, 'scraped'])
                self.progress_tracker.to_feather(
                    self.metadata_path/'bts_metadata_progress_tracker')
                # print(self.progress_tracker)
                product_meta_data = []
        self.logger.info('Metadata Extraction Complete')
        print('Metadata Extraction Complete')
        # self.progress_monitor.info('Metadata Extraction Complete')

    def extract(self, download: bool = True, fresh_start: bool = False, auto_fresh_start: bool = False, n_workers: int = 5,
                open_headless: bool = False, open_with_proxy_server: bool = True, randomize_proxy_usage: bool = True,
                start_idx: Optional[int] = None, end_idx: Optional[int] = None, list_of_index=None,
                clean: bool = True, compile_progress_files: bool = False, delete_progress: bool = False) -> None:
        """extract [summary]

        [extended_summary]

        Args:
            download (bool, optional): [description]. Defaults to True.
            fresh_start (bool, optional): [description]. Defaults to False.
            auto_fresh_start (bool, optional): [description]. Defaults to False.
            n_workers (int, optional): [description]. Defaults to 5.
            open_headless (bool, optional): [description]. Defaults to False.
            open_with_proxy_server (bool, optional): [description]. Defaults to True.
            randomize_proxy_usage (bool, optional): [description]. Defaults to True.
            start_idx (Optional[int], optional): [description]. Defaults to None.
            end_idx (Optional[int], optional): [description]. Defaults to None.
            list_of_index ([type], optional): [description]. Defaults to None.
            clean (bool, optional): [description]. Defaults to True.
            compile_progress_files (bool, optional): [description]. Defaults to False.
            delete_progress (bool, optional): [description]. Defaults to False.
        """
        def fresh():
            """[summary]
            """
            self.product_type_urls = self.get_product_type_urls(open_headless=open_headless,
                                                                open_with_proxy_server=open_with_proxy_server)
            # progress tracker: captures scraped and error desc
            self.progress_tracker = pd.DataFrame(index=self.product_type_urls.index, columns=[
                'product_type', 'scraped', 'error_desc'])
            self.progress_tracker.scraped = 'N'

        if fresh_start:
            self.logger.info('Starting Fresh Extraction.')
            fresh()
        else:
            if Path(self.metadata_path/'bts_product_type_urls_to_extract').exists():
                self.product_type_urls = pd.read_feather(
                    self.metadata_path/'bts_product_type_urls_to_extract')
                if Path(self.metadata_path/'bts_metadata_progress_tracker').exists():
                    self.progress_tracker = pd.read_feather(
                        self.metadata_path/'bts_metadata_progress_tracker')
                else:
                    self.progress_tracker = pd.DataFrame(index=self.product_type_urls.index, columns=[
                        'product_type', 'scraped', 'error_desc'])
                    self.progress_tracker.scraped = 'N'
                    self.progress_tracker.to_feather(
                        self.metadata_path/'bts_metadata_progress_tracker')
                if sum(self.progress_tracker.scraped == 'N') > 0:
                    self.logger.info(
                        'Continuing Metadata Extraction From Last Run.')
                    self.product_type_urls = self.product_type_urls[self.product_type_urls.index.isin(
                        self.progress_tracker.index[self.progress_tracker.scraped == 'N'].values.tolist())]
                else:
                    if auto_fresh_start:
                        self.logger.info(
                            'Previous Run Was Complete. Starting Fresh Extraction.')
                        fresh()
                    else:
                        self.logger.info(
                            'Previous Run is Complete.')
            else:
                self.logger.info(
                    'URL File Not Found. Start Fresh Extraction.')
        # print(self.progress_tracker)
        if download:
            # set list or range of product indices to crawl
            if list_of_index:
                indices = list_of_index
            elif start_idx and end_idx is None:
                indices = range(start_idx, len(self.product_type_urls))
            elif start_idx is None and end_idx:
                indices = range(0, end_idx)
            elif start_idx is not None and end_idx is not None:
                indices = range(start_idx, end_idx)
            else:
                indices = range(len(self.product_type_urls))
            # print(indices)
            if list_of_index:
                self.get_metadata(indices=list_of_index,
                                  open_headless=open_headless,
                                  open_with_proxy_server=open_with_proxy_server,
                                  randomize_proxy_usage=randomize_proxy_usage,
                                  product_meta_data=[])
            else:
                '''
                # review_Data and item_data are lists of empty lists so that each namepace of function call will
                # have its separate detail_data
                # list to strore scraped dictionaries. will save memory(ram/hard-disk) consumption. will stop data duplication
                '''
                if start_idx:
                    lst_of_lst = ranges(
                        indices[-1]+1, n_workers, start_idx=start_idx)
                else:
                    lst_of_lst = ranges(len(indices), n_workers)
                print(lst_of_lst)
                headless = [open_headless for i in lst_of_lst]
                proxy = [open_with_proxy_server for i in lst_of_lst]
                rand_proxy = [randomize_proxy_usage for i in lst_of_lst]
                product_meta_data = [[] for i in lst_of_lst]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    '''
                    # but each of the function namespace will be modifying only one metadata tracing file so that progress saving
                    # is tracked correctly. else multiple progress tracker file will be created with difficulty to combine correct
                    # progress information
                    '''
                    executor.map(self.get_metadata, lst_of_lst,
                                 headless, proxy, rand_proxy, product_meta_data)

        if compile_progress_files:
            self.logger.info('Creating Combined Metadata File')
            files = [f for f in self.current_progress_path.glob(
                "bts_prod_meta_extract_progress_*")]
            li = [pd.read_feather(file) for file in files]
            metadata_df = pd.concat(li, axis=0, ignore_index=True)
            metadata_df.reset_index(inplace=True, drop=True)
            metadata_df['source'] = self.source

            if datetime.now().day < 15:
                meta_date = f'{time.strftime("%Y-%m")}-01'
            else:
                meta_date = f'{time.strftime("%Y-%m")}-15'
            filename = f'bts_product_metadata_all_{meta_date}'
            metadata_df.to_feather(self.metadata_path/filename)

            self.logger.info(
                f'Metadata file created. Please look for file {filename} in path {self.metadata_path}')
            print(
                f'Metadata file created. Please look for file {filename} in path {self.metadata_path}')

            if clean:
                cleaner = Cleaner(path=self.path)
                _ = cleaner.clean(
                    data=self.metadata_path/filename)
                self.logger.info(
                    'Metadata Cleaned and Removed Duplicates for Details/Review/Image Extraction.')

            if delete_progress:
                shutil.rmtree(
                    f'{self.metadata_path}\\current_progress', ignore_errors=True)
                self.logger.info('Progress files deleted')

    def terminate_logging(self):
        """terminate_logging [summary]

        [extended_summary]
        """
        self.logger.handlers.clear()
        self.prod_meta_log.stop_log()


class DetailReview(Boots):
    """
    [summary]

    Arguments:
        Browser {[type]} -- [description]
    """

    def __init__(self, log: bool = True, path: Path = Path.cwd()):
        """__init__ [summary]

        [extended_summary]

        Args:
            log (bool, optional): [description]. Defaults to True.
            path (Path, optional): [description]. Defaults to Path.cwd().
        """
        super().__init__(path=path, data_def='detail_review_image')
        self.path = Path(path)
        self.detail_current_progress_path = self.detail_path/'current_progress'
        self.detail_current_progress_path.mkdir(parents=True, exist_ok=True)

        self.review_current_progress_path = self.review_path/'current_progress'
        self.review_current_progress_path.mkdir(parents=True, exist_ok=True)

        old_detail_files = list(self.detail_path.glob(
            'bts_product_detail_all*')) + list(self.detail_path.glob(
                'bts_product_item_all*'))
        for f in old_detail_files:
            shutil.move(str(f), str(self.old_detail_files_path))

        old_clean_detail_files = os.listdir(self.detail_clean_path)
        for f in old_clean_detail_files:
            shutil.move(str(self.detail_clean_path/f),
                        str(self.old_detail_clean_files_path))

        old_review_files = list(self.review_path.glob(
            'bts_product_review_all*'))
        for f in old_review_files:
            shutil.move(str(f), str(self.old_review_files_path))

        old_clean_review_files = os.listdir(self.review_clean_path)
        for f in old_clean_review_files:
            shutil.move(str(self.review_clean_path/f),
                        str(self.old_review_clean_files_path))
        if log:
            self.prod_detail_review_image_log = Logger(
                "bts_prod_review_image_extraction", path=self.crawl_log_path)
            self.logger, _ = self.prod_detail_review_image_log.start_log()

    def get_details(self, drv: webdriver.Firefox, prod_id: str, product_name: str) -> Tuple[dict, pd.DataFrame]:
        """get_detail [summary]

        [extended_summary]

        Args:
            drv (webdriver.Firefox): [description]
            prod_id (str): [description]
            product_name (str): [description]

        Returns:
            Tuple[dict, pd.DataFrame]: [description]
        """
        def get_product_attributes(drv: webdriver.Firefox, prod_id: str, product_name: str):
            """[summary]
            """
            # get all the variation of product
            product_attributes = []

            # product_variety = drv.find_elements_by_css_selector(
            #     'li[id*="size_combo_button_pdp"]')

            try:
                item_ingredients = drv.find_element_by_xpath(
                    '//div[h3[@id="product_ingredients"]]').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')
                item_ingredients = ''
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) item_ingredients does not exist.', 'utf-8', 'ignore'))

            try:
                item_price = drv.find_element_by_xpath(
                    '//div[@id="PDP_productPrice"]').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')
                item_price = ''

            try:
                item_size_price = drv.find_element_by_css_selector(
                    'div.details').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')
                item_size_price = ''
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) item_size does not exist.', 'utf-8', 'ignore'))

            item_size_price = item_price + " per " + item_size_price

            if len(item_size_price.split('|')) > 0:

                for i in item_size_price.split('|'):
                    ps = i.split('per')
                    item_price = ps[0].strip()
                    item_size = ps[1].strip()
                    product_attributes.append(
                        {"prod_id": prod_id, "product_name": product_name,
                         "item_name": '', "item_size": item_size,
                         "item_price": item_price, "item_ingredients": item_ingredients
                         }
                    )

            else:
                product_attributes.append(
                    {"prod_id": prod_id, "product_name": product_name,
                     "item_name": '', "item_size": item_size_price,
                     "item_price": item_price, "item_ingredients": item_ingredients
                     }
                )

            return product_attributes

        try:
            abt_product = drv.find_element_by_css_selector(
                'div[id="contentOmnipresent"]').text
        except Exception as ex:
            log_exception(self.logger,
                          additional_information=f'Prod ID: {prod_id}')
            abt_product = ''
            self.logger.info(str.encode(
                f'product: {product_name} (prod_id: {prod_id}) product detail does not exist.', 'utf-8', 'ignore'))

        try:
            how_to_use = drv.find_element_by_xpath(
                '//div[h3[@id="product_how_to_use"]]').text
        except Exception as ex:
            log_exception(self.logger,
                          additional_information=f'Prod ID: {prod_id}')
            how_to_use = ''
            self.logger.info(str.encode(
                f'product: {product_name} (prod_id: {prod_id}) how_to_use does not exist.', 'utf-8', 'ignore'))

        try:
            reviews = drv.find_element_by_css_selector(
                'span[itemprop="reviewCount"]').text
        except Exception as ex:
            log_exception(self.logger,
                          additional_information=f'Prod ID: {prod_id}')
            reviews = 0
            self.logger.info(str.encode(
                f'product: {product_name} (prod_id: {prod_id}) reviews does not exist.', 'utf-8', 'ignore'))

        try:
            ratings = drv.find_elements_by_css_selector(
                'div.bv-inline-histogram-ratings-score>span:nth-child(1)')
            five_star = ratings[0].text
            four_star = ratings[1].text
            three_star = ratings[2].text
            two_star = ratings[3].text
            one_star = ratings[4].text

        except Exception as ex:
            log_exception(self.logger,
                          additional_information=f'Prod ID: {prod_id}')
            five_star = ''
            four_star = ''
            three_star = ''
            two_star = ''
            one_star = ''
            self.logger.info(str.encode(
                f'product: {product_name} (prod_id: {prod_id}) rating_distribution does not exist.', 'utf-8', 'ignore'))

        detail = {"prod_id": prod_id,
                  "product_name": product_name,
                  "abt_product": abt_product,
                  "abt_brand": '',
                  "how_to_use": how_to_use,
                  "reviews": reviews,
                  "votes": '',
                  "five_star": five_star,
                  "four_star": four_star,
                  "three_star": three_star,
                  "two_star": two_star,
                  "one_star": one_star,
                  "would_recommend": '',
                  "first_review_date": ''
                  }

        item = pd.DataFrame(
            get_product_attributes(drv, prod_id, product_name))

        return detail, item

    def get_reviews(self,  drv: webdriver.Firefox, prod_id: str, product_name: str,
                    last_scraped_review_date: str, no_of_reviews: int,
                    incremental: bool = True, reviews: list = [])-> list:
        """get_reviews [summary]

        [extended_summary]

        Args:
            drv (webdriver.Firefox): [description]
            prod_id (str): [description]
            product_name (str): [description]
            last_scraped_review_date (str): [description]
            no_of_reviews (int): [description]
            incremental (bool, optional): [description]. Defaults to True.
            reviews (list, optional): [description]. Defaults to [].

        Returns:
            list: [description]
        """
        # print(no_of_reviews)
        # drv.find_element_by_class_name('css-2rg6q7').click()
        if incremental and last_scraped_review_date != '':
            for i in range(no_of_reviews//30):
                if i >= 100:
                    break
                time.sleep(0.4)
                revs = drv.find_elements_by_css_selector(
                    'ol[class="bv-content-list bv-content-list-reviews"]>li')
                date1 = convert_ago_to_date(revs[-1].find_element_by_css_selector('div.bv-content-datetime>meta[itemprop="dateCreated"]'
                                                                                  ).get_attribute('content'))
                date2 = convert_ago_to_date(revs[-2].find_element_by_css_selector('div.bv-content-datetime>meta[itemprop="dateCreated"]'
                                                                                  ).get_attribute('content'))
                try:
                    if pd.to_datetime(date1, infer_datetime_format=True)\
                            < pd.to_datetime(last_scraped_review_date, infer_datetime_format=True):
                        # print('breaking incremental')
                        break
                except Exception as ex:
                    log_exception(self.logger,
                                  additional_information=f'Prod ID: {prod_id}')
                    try:
                        if pd.to_datetime(date2, infer_datetime_format=True)\
                                < pd.to_datetime(last_scraped_review_date, infer_datetime_format=True):
                            # print('breaking incremental')
                            break
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod ID: {prod_id}')
                        self.logger.info(str.encode(f'Product: {product_name} prod_id: {prod_id} \
                                                        last_scraped_review_date to current review date \
                                                        comparision failed.(page: {product_page})',
                                                    'utf-8', 'ignore'))
                        # print('in second except block')
                        continue

                if len(revs) >= 2000:
                    break
                try:
                    accept_alert(drv, 1)
                    close_popups(drv)
                    show_more_review_button = drv.find_element_by_css_selector(
                        'span[class="bv-content-btn-pages-load-more-text"]')
                except Exception as ex:
                    log_exception(self.logger,
                                  additional_information=f'Prod ID: {prod_id}. Loaded all the reviews. No more reviews exist.')
                    show_more_review_button = ''
                    break
                if show_more_review_button != '':
                    self.scroll_to_element(
                        drv, show_more_review_button)
                    ActionChains(drv).move_to_element(
                        show_more_review_button).click(show_more_review_button).perform()
                    # print('loading more reviews')
                time.sleep(0.4)
        else:
            for n in range(no_of_reviews//30+5):
                '''
                code will stop after getting 1800 reviews of one particular product
                when crawling all reviews. By default it will get latest 1800 reviews.
                then in subsequent incremental runs it will get al new reviews on weekly basis
                '''
                if n >= 100:  # 200:
                    break
                time.sleep(1)

                # close any opened popups by escape
                accept_alert(drv, 1)
                close_popups(drv)
                try:
                    show_more_review_button = drv.find_element_by_css_selector(
                        'span[class="bv-content-btn-pages-load-more-text"]')
                except Exception as ex:
                    log_exception(self.logger,
                                  additional_information=f'Prod ID: {prod_id}. Loaded all the reviews. No more reviews exist.')
                    show_more_review_button = ''
                    break
                if show_more_review_button != '':
                    self.scroll_to_element(
                        drv, show_more_review_button)
                    ActionChains(drv).move_to_element(
                        show_more_review_button).click(show_more_review_button).perform()
                    # print('loading more reviews')

        accept_alert(drv, 2)
        close_popups(drv)

        product_reviews = drv.find_elements_by_css_selector(
            'ol[class="bv-content-list bv-content-list-reviews"]>li')

        r = 0
        for rev in product_reviews:
            accept_alert(drv, 0.5)
            close_popups(drv)
            try:
                self.scroll_to_element(drv, rev)
                ActionChains(drv).move_to_element(rev).perform()
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to scroll to review.')
                pass

            try:
                review_text = rev.find_element_by_css_selector(
                    'div.bv-content-summary-body-text').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract review_text. Skip review.')
                continue

            try:
                review_date = convert_ago_to_date(
                    rev.find_element_by_css_selector('div.bv-content-datetime>meta[itemprop="dateCreated"]').get_attribute('content'))
                if pd.to_datetime(review_date, infer_datetime_format=True) <= \
                        pd.to_datetime(last_scraped_review_date, infer_datetime_format=True):
                    continue
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract review_date.')
                review_date = ''

            try:
                review_title = rev.find_element_by_css_selector(
                    'h3.bv-content-title').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract review_title.')
                review_title = ''

            try:
                product_variant = rev.find_element_by_class_name(
                    'css-1op1cn7').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract product_variant.')
                product_variant = ''

            try:
                user_rating = rev.find_element_by_css_selector(
                    'span.bv-rating-stars-container>span.bv-off-screen').text.split(" ")[0]
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract user_rating.')
                user_rating = ''

            try:
                helpful_yes = rev.find_element_by_css_selector(
                    'div.bv-content-feedback-btn-container>button:nth-child(1) span.bv-content-btn-count').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Helpful Yes not found.')
                helpful_yes = ''

            try:
                helpful_no = rev.find_element_by_css_selector(
                    'div.bv-content-feedback-btn-container>button:nth-child(2) span.bv-content-btn-count').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Helpful No not found.')
                helpful_no = ''

            try:
                recommend = rev.find_element_by_css_selector(
                    'dl[class*="recommend"] span.bv-content-data-label').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract recommend.')
                recommend = ''

            reviews.append(
                {
                    'prod_id': prod_id, 'product_name': product_name,
                    'user_attribute': '', 'product_variant': product_variant,
                    'review_title': review_title, 'review_text': review_text,
                    'review_rating': user_rating, 'recommend': recommend,
                    'review_date': review_date, "helpful_y": helpful_yes,
                    "helpful_n": helpful_no
                }
            )
        return reviews

    def crawl_page(self, indices: list, open_headless: bool, open_with_proxy_server: bool,
                   randomize_proxy_usage: bool, detail_data: list = [],
                   item_df=pd.DataFrame(columns=['prod_id', 'product_name', 'item_name',
                                                 'item_size', 'item_price',
                                                 'item_ingredients']),
                   review_data: list = [], incremental: bool = True):
        """crawl_page

        [extended_summary]

        Args:
            indices (list): [description]
            open_headless (bool): [description]
            open_with_proxy_server (bool): [description]
            randomize_proxy_usage (bool): [description]
            detail_data (list, optional): [description]. Defaults to [].
            item_df ([type], optional): [description]. Defaults to pd.DataFrame(columns=['prod_id', 'product_name', 'item_name',
            'item_size', 'item_price', 'item_ingredients']).
            review_data (list, optional): [description]. Defaults to [].
            incremental (bool, optional): [description]. Defaults to True.
        """

        def store_data_refresh_mem(detail_data: list, item_df: pd.DataFrame,
                                   review_data: list) -> Tuple[list, pd.DataFrame, list]:
            """store_data_refresh_mem [summary]

            [extended_summary]

            Args:
                detail_data (list): [description]
                item_df (pd.DataFrame): [description]
                review_data (list): [description]

            Returns:
                Tuple[list, pd.DataFrame, list]: [description]
            """
            pd.DataFrame(detail_data).to_csv(self.detail_current_progress_path /
                                             f'bts_prod_detail_extract_progress_{time.strftime("%Y-%m-%d-%H%M%S")}.csv',
                                             index=None)
            item_df.reset_index(inplace=True, drop=True)
            item_df.to_csv(self.detail_current_progress_path /
                           f'bts_prod_item_extract_progress_{time.strftime("%Y-%m-%d-%H%M%S")}.csv',
                           index=None)
            item_df = pd.DataFrame(columns=[
                                   'prod_id', 'product_name', 'item_name', 'item_size', 'item_price', 'item_ingredients'])

            pd.DataFrame(review_data).to_csv(self.review_current_progress_path /
                                             f'bts_prod_review_extract_progress_{time.strftime("%Y-%m-%d-%H%M%S")}.csv',
                                             index=None)
            self.meta.to_csv(
                self.path/'boots/bts_detail_review_image_progress_tracker.csv', index=None)
            return [], item_df, []

        for prod in self.meta.index[self.meta.index.isin(indices)]:
            #  ignore already extracted products
            if self.meta.loc[prod, 'scraped'] in ['Y', 'NA']:
                continue
            # print(prod, self.meta.loc[prod, 'detail_scraped'])
            prod_id = self.meta.loc[prod, 'prod_id']
            product_name = self.meta.loc[prod, 'product_name']
            product_page = self.meta.loc[prod, 'product_page']
            last_scraped_review_date = self.meta.loc[prod,
                                                     'last_scraped_review_date']
            # create webdriver
            if randomize_proxy_usage:
                use_proxy = np.random.choice([True, False])
            else:
                use_proxy = True
            if open_with_proxy_server:
                # print(use_proxy)
                drv = self.open_browser(open_headless=open_headless, open_with_proxy_server=use_proxy,
                                        path=self.detail_path)
            else:
                drv = self.open_browser(open_headless=open_headless, open_with_proxy_server=False,
                                        path=self.detail_path)
            # open product page
            drv.get(product_page)
            time.sleep(15)  # 30
            accept_alert(drv, 5)
            close_popups(drv)

            # check product page is valid and exists
            try:
                close_popups(drv)
                accept_alert(drv, 2)
                price = drv.find_element_by_xpath(
                    '//div[@id="PDP_productPrice"]').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')
                drv.quit()
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) no longer exists in the previously fetched link.\
                        (link:{product_page})', 'utf-8', 'ignore'))
                self.meta.loc[prod, 'detail_scraped'] = 'NA'
                continue

            # self.scroll_down_page(drv, speed=6, h2=0.6)
            # time.sleep(5)

            detail, item = self.get_details(drv, prod_id, product_name)

            detail_data.append(detail)
            item_df = pd.concat(
                [item_df, item], axis=0)

            # item_data.append(product_attributes)
            self.logger.info(str.encode(
                f'product: {product_name} (prod_id: {prod_id}) details extracted successfully', 'utf-8', 'ignore'))

            try:
                close_popups(drv)
                accept_alert(drv, 1)
                no_of_reviews = int(drv.find_element_by_css_selector(
                    'span[itemprop="reviewCount"]').text)
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')
                self.logger.info(str.encode(f'Product: {product_name} prod_id: {prod_id} reviews extraction failed.\
                                              Either product has no reviews or not\
                                              available for sell currently.(page: {product_page})', 'utf-8', 'ignore'))
                no_of_reviews = 0

            if no_of_reviews > 0:
                reviews = self.get_reviews(
                    drv, prod_id, product_name, last_scraped_review_date, no_of_reviews, incremental, reviews=[])

                if len(reviews) > 0:
                    review_data.extend(reviews)

                    if not incremental:
                        self.logger.info(str.encode(
                            f'Product_name: {product_name} prod_id:{prod_id} reviews extracted successfully.(total_reviews: {no_of_reviews}, \
                            extracted_reviews: {len(reviews)}, page: {product_page})', 'utf-8', 'ignore'))
                    else:
                        self.logger.info(str.encode(
                            f'Product_name: {product_name} prod_id:{prod_id} new reviews extracted successfully.\
                                (no_of_new_extracted_reviews: {len(reviews)},\
                                page: {product_page})', 'utf-8', 'ignore'))
            else:
                reviews = 0

            if prod != 0 and prod % 5 == 0:
                detail_data, item_df, review_data = store_data_refresh_mem(
                    detail_data, item_df, review_data)

            self.meta.loc[prod, 'scraped'] = 'Y'
            drv.quit()

        detail_data, item_df, review_data = store_data_refresh_mem(
            detail_data, item_df, review_data)
        self.logger.info(
            f'Extraction Complete for start_idx: {indices[0]} to end_idx: {indices[-1]}. Or for list of values.')

    def extract(self, metadata: Union[pd.DataFrame, str, Path], download: bool = True, n_workers: int = 5,
                fresh_start: bool = False, auto_fresh_start: bool = False, incremental: bool = True,
                open_headless: bool = False, open_with_proxy_server: bool = True, randomize_proxy_usage: bool = True,
                start_idx: Optional[int] = None, end_idx: Optional[int] = None, list_of_index=None,
                compile_progress_files: bool = False, clean: bool = True, delete_progress: bool = False):
        """extract [summary]

        [extended_summary]

        Args:
            metadata (Union[pd.DataFrame, str, Path]): [description]
            download (bool, optional): [description]. Defaults to True.
            n_workers (int, optional): [description]. Defaults to 5.
            fresh_start (bool, optional): [description]. Defaults to False.
            auto_fresh_start (bool, optional): [description]. Defaults to False.
            incremental (bool, optional): [description]. Defaults to True.
            open_headless (bool, optional): [description]. Defaults to False.
            open_with_proxy_server (bool, optional): [description]. Defaults to True.
            randomize_proxy_usage (bool, optional): [description]. Defaults to True.
            start_idx (Optional[int], optional): [description]. Defaults to None.
            end_idx (Optional[int], optional): [description]. Defaults to None.
            list_of_index ([type], optional): [description]. Defaults to None.
            compile_progress_files (bool, optional): [description]. Defaults to False.
            clean (bool, optional): [description]. Defaults to True.
            delete_progress (bool, optional): [description]. Defaults to False.
        """
        def fresh():
            if not isinstance(metadata, pd.core.frame.DataFrame):
                list_of_files = self.metadata_clean_path.glob(
                    'no_cat_cleaned_bts_product_metadata_all*')
                self.meta = pd.read_feather(max(list_of_files, key=os.path.getctime))[
                    ['prod_id', 'product_name', 'product_page', 'meta_date', 'last_scraped_review_date']]
            else:
                self.meta = metadata[[
                    'prod_id', 'product_name', 'product_page', 'meta_date', 'last_scraped_review_date']]
            self.meta.last_scraped_review_date.fillna('', inplace=True)
            self.meta['scraped'] = 'N'

        if download:
            if fresh_start:
                fresh()
                self.logger.info(
                    'Starting Fresh Deatil Review Image Extraction.')
            else:
                if Path(self.path/'boots/bts_detail_review_image_progress_tracker.csv').exists():
                    self.meta = pd.read_csv(
                        self.path/'boots/bts_detail_review_image_progress_tracker.csv')
                    if sum(self.meta.scraped == 'N') == 0:
                        if auto_fresh_start:
                            fresh()
                            self.logger.info(
                                'Last Run was Completed. Starting Fresh Extraction.')
                        else:
                            self.logger.info(
                                'Deatil Review Image extraction for this cycle is complete.')
                    else:
                        self.logger.info(
                            'Continuing Deatil Review Image Extraction From Last Run.')
                else:
                    if auto_fresh_start:
                        fresh()
                        self.logger.info(
                            'Deatil Review Image Progress Tracker does not exist. Starting Fresh Extraction.')

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
                self.crawl_page(indices=list_of_index, incremental=incremental,
                                open_headless=open_headless,
                                open_with_proxy_server=open_with_proxy_server,
                                randomize_proxy_usage=randomize_proxy_usage)
            else:  # By default the code will with 5 concurrent threads. you can change this behaviour by changing n_workers
                if start_idx:
                    lst_of_lst = ranges(
                        indices[-1]+1, n_workers, start_idx=start_idx)
                else:
                    lst_of_lst = ranges(len(indices), n_workers)
                print(lst_of_lst)
                # detail_Data and item_data are lists of empty lists so that each namepace of function call will have its separate detail_data
                # list to strore scraped dictionaries. will save memory(ram/hard-disk) consumption. will stop data duplication
                headless = [open_headless for i in lst_of_lst]
                proxy = [open_with_proxy_server for i in lst_of_lst]
                rand_proxy = [randomize_proxy_usage for i in lst_of_lst]
                detail_data = [[] for i in lst_of_lst]  # type: List
                # item_data=[[] for i in lst_of_lst]  # type: List
                item_df = [pd.DataFrame(columns=['prod_id', 'product_name', 'item_name',
                                                 'item_size', 'item_price', 'item_ingredients'])
                           for i in lst_of_lst]
                review_data = [[] for i in lst_of_lst]  # type: list
                inc_list = [incremental for i in lst_of_lst]  # type: list
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # but each of the function namespace will be modifying only one metadata tracing file so that progress saving
                    # is tracked correctly. else multiple progress tracker file will be created with difficulty to combine correct
                    # progress information
                    print('inside executor')
                    executor.map(self.crawl_page, lst_of_lst,
                                 headless, proxy, rand_proxy,
                                 detail_data, item_df, review_data,
                                 inc_list)
        try:
            if compile_progress_files:
                self.logger.info('Creating Combined Detail and Item File')
                if datetime.now().day < 15:
                    meta_date = f'{time.strftime("%Y-%m")}-01'
                else:
                    meta_date = f'{time.strftime("%Y-%m")}-15'

                det_li = []
                self.bad_det_li = []
                detail_files = [f for f in self.detail_current_progress_path.glob(
                    "bts_prod_detail_extract_progress_*")]
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
                detail_df['meta_date'] = meta_date
                detail_filename = f'bts_product_detail_all_{meta_date}.csv'
                detail_df.to_csv(self.detail_path/detail_filename, index=None)
                # detail_df.to_feather(self.detail_path/detail_filename)

                item_li = []
                self.bad_item_li = []
                item_files = [f for f in self.detail_current_progress_path.glob(
                    "bts_prod_item_extract_progress_*")]
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
                item_dataframe['meta_date'] = meta_date
                item_filename = f'bts_product_item_all_{meta_date}.csv'
                item_dataframe.to_csv(
                    self.detail_path/item_filename, index=None)
                # item_df.to_feather(self.detail_path/item_filename)

                self.logger.info(
                    f'Detail and Item files created. Please look for file bts_product_detail_all and\
                        bts_product_item_all in path {self.detail_path}')
                print(
                    f'Detail and Item files created. Please look for file bts_product_detail_all and\
                        bts_product_item_all in path {self.detail_path}')

                self.logger.info('Creating Combined Review File')

                rev_li = []
                self.bad_rev_li = []
                review_files = [f for f in self.review_current_progress_path.glob(
                    "bts_prod_review_extract_progress_*")]
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
                rev_df['meta_date'] = pd.to_datetime(meta_date).date()
                review_filename = f'bts_product_review_all_{meta_date}'
                # , index=None)
                rev_df.to_feather(self.review_path/review_filename)

                self.logger.info(
                    f'Review file created. Please look for file bts_product_review_all in path {self.review_path}')
                print(
                    f'Review file created. Please look for file bts_product_review_all in path {self.review_path}')

                if clean:
                    detail_cleaner = Cleaner(path=self.path)
                    self.detail_clean_df = detail_cleaner.clean(
                        self.detail_path/detail_filename)

                    item_cleaner = Cleaner(path=self.path)
                    self.item_clean_df, self.ing_clean_df = item_cleaner.clean(
                        self.detail_path/item_filename)

                    review_cleaner = Cleaner(path=self.path)
                    self.review_clean_df = review_cleaner.clean(
                        self.review_path/review_filename)

                    file_creation_status = True
            else:
                file_creation_status = False
        except Exception as ex:
            log_exception(
                self.logger, additional_information=f'Detail Item Review Combined File Creation Failed.')
            file_creation_status = False

        if delete_progress and file_creation_status:
            shutil.rmtree(
                f'{self.detail_path}\\current_progress', ignore_errors=True)
            shutil.rmtree(
                f'{self.review_path}\\current_progress', ignore_errors=True)
            self.logger.info('Progress files deleted')

    def terminate_logging(self):
        """terminate_logging [summary]

        [extended_summary]
        """
        self.logger.handlers.clear()
        self.prod_detail_review_image_log.stop_log()
