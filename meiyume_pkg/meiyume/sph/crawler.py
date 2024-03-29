"""The module to crawl Sephora website data."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from typing import *
import sys
import types
import pdb
import gc
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
from meiyume.utils import (Browser, Logger, MeiyumeException, Sephora,
                           accept_alert, close_popups, log_exception,
                           chunks, ranges, convert_ago_to_date)

warnings.simplefilter(action='ignore', category=FutureWarning)


class Metadata(Sephora):
    """Metadata extracts product metadata such as product page url, prices and brand from Sephora website.

    The Metadata class begins the data crawling process and all other stages depend on the product urls extracted by Metadata class.

    Args:
        Sephora (Browser): Class that initializes folder paths and selenium webdriver for data scraping.

    """

    base_url = "https://www.sephora.com"
    info = tldextract.extract(base_url)
    source = info.registered_domain

    @classmethod
    def update_base_url(cls, url: str) -> None:
        """update_base_url defines the parent url from where the data scraping process will begin.

        Args:
            url (str): The URL from which the spider will enter the website.

        """
        cls.base_url = url
        cls.info = tldextract.extract(cls.base_url)
        cls.source = cls.info.registered_domain

    def __init__(self, log: bool = True, path: Path = Path.cwd()):
        """__init__ Metadata class instace initializer.

        This method sets all the folder paths required for Metadata crawler to work.
        If the paths does not exist the paths get automatically created depending on current directory
        or provided directory.

        Args:
            log (bool, optional): Whether to create crawling exception and progess log. Defaults to True.
            path (Path, optional): Folder path where the Metadata will be extracted. Defaults to current directory(Path.cwd()).

        """

        super().__init__(path=path, data_def='meta')
        self.path = path
        self.current_progress_path = self.metadata_path/'current_progress'
        self.current_progress_path.mkdir(parents=True, exist_ok=True)

        # move old raw and clean files to old folder
        old_metadata_files = list(self.metadata_path.glob(
            'sph_product_metadata_all*'))
        for f in old_metadata_files:
            shutil.move(str(f), str(self.old_metadata_files_path))

        old_clean_metadata_files = os.listdir(self.metadata_clean_path)
        for f in old_clean_metadata_files:
            shutil.move(str(self.metadata_clean_path/f),
                        str(self.old_metadata_clean_files_path))
        # set logger
        if log:
            self.prod_meta_log = Logger(
                "sph_prod_metadata_extraction", path=self.crawl_log_path)
            self.logger, _ = self.prod_meta_log.start_log()

    def get_product_type_urls(self, open_headless: bool, open_with_proxy_server: bool) -> pd.DataFrame:
        """get_product_type_urls Extract the category/subcategory structure and urls to extract the products of those category/subcategory.

        Extracts the links of pages containing the list of all products structured into
        category/subcategory/product type to effectively stored in relational database.
        Defines the structure of data extraction that helps store unstructured data in a structured manner.

        Args:
            open_headless (bool): Whether to open browser headless.
            open_with_proxy_server (bool): Whether to use proxy server.

        Returns:
            pd.DataFrame: returns pandas dataframe containing urls for getting list of products, category, subcategory etc.

        """
        # create webdriver instance
        drv = self.open_browser(
            open_headless=open_headless, open_with_proxy_server=open_with_proxy_server, path=self.metadata_path)

        drv.get(self.base_url)
        time.sleep(15)
        # click and close welcome forms
        accept_alert(drv, 10)
        close_popups(drv)

        cats = drv.find_elements_by_css_selector('a[id^="top_nav_drop_"]')
        cat_urls = []
        for c in cats:
            if c.get_attribute('href') is not None:
                cat_name, url = (c.get_attribute("href").split("/")
                                 [-1], c.get_attribute("href"))
                cat_urls.append((cat_name, url))
                self.logger.info(str.encode(f'Category:- name:{cat_name}, \
                                          url:{url}', "utf-8", "ignore"))

        sub_cat_urls = []
        for cu in cat_urls:
            cat_name = cu[0]
            if cat_name in ['brands-list', 'new-beauty-products', 'sephora-collection']:
                continue
            cat_url = cu[1]
            drv.get(cat_url)

            time.sleep(10)
            accept_alert(drv, 10)
            close_popups(drv)

            sub_cats = drv.find_elements_by_css_selector(
                'a[data-at*="top_level_category"]')
            # sub_cats.extend(drv.find_elements_by_class_name("css-or7ouu"))
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
            if any(name in sub_cat_name for name in ['best-selling', 'new', 'mini']):
                continue
            sub_cat_url = su[2]
            drv.get(sub_cat_url)

            time.sleep(10)
            accept_alert(drv, 10)
            close_popups(drv)

            product_types = drv.find_elements_by_css_selector(
                'a[data-at*="nth_level"]')
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
        df.drop(columns='sub_category_raw', inplace=True)
        df.reset_index(inplace=True, drop=True)
        df['scraped'] = 'N'
        df.to_feather(self.metadata_path/f'sph_product_type_urls_to_extract')
        return df

    def get_metadata(self, indices: Union[list, range],
                     open_headless: bool, open_with_proxy_server: bool,
                     randomize_proxy_usage: bool,
                     product_meta_data: list = []) -> None:
        """get_metadata Crawls product listing pages for price, name, brand etc.

        Get Metadata crawls a product type page for example lipstick.
        The function gets individual product urls, names, brands and prices etc. and stores
        in a relational table structure to use later to download product images, scrape reviews and
        other specific information.

        Args:
            indices (Union[list, range]): list of indices or range of indices of product urls to scrape.
            open_headless (bool): Whether to open browser headless.
            open_with_proxy_server (bool): Whether to use proxy server.
            randomize_proxy_usage (bool): Whether to use both proxy and native network in tandem to decrease proxy requests.
            product_meta_data (list, optional): Empty intermediate list to store product metadata during parallel crawl. Defaults to [].

        """
        for pt in self.product_type_urls.index[self.product_type_urls.index.isin(indices)]:
            cat_name = self.product_type_urls.loc[pt, 'category_raw']
            product_type = self.product_type_urls.loc[pt, 'product_type']
            product_type_link = self.product_type_urls.loc[pt, 'url']

            # print(product_type_link)
            if 'best-selling' in product_type.lower() or 'new' in product_type.lower():
                self.product_type_urls.loc[pt, 'scraped'] = 'NA'
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

            try:
                chat_popup_button = WebDriverWait(drv, 3).until(
                    EC.presence_of_element_located((By.XPATH, '//*[@id="divToky"]/img[3]')))
                chat_popup_button = drv.find_element_by_xpath(
                    '//*[@id="divToky"]/img[3]')
                self.scroll_to_element(drv, chat_popup_button)
                ActionChains(drv).move_to_element(
                    chat_popup_button).click(chat_popup_button).perform()
            except TimeoutException:
                pass

            # sort by new products (required to get all new products properly)
            try:
                sort_dropdown = drv.find_element_by_css_selector(
                    'button[id="cat_sort_menu_trigger"]')
                Browser().scroll_to_element(drv, sort_dropdown)
                ActionChains(drv).move_to_element(
                    sort_dropdown).click(sort_dropdown).perform()
                drv.find_elements_by_css_selector(
                    'div[id="cat_sort_menu"]>button')[2].click()
                time.sleep(10)
                # sort_dropdown = drv.find_element_by_class_name('css-16tfpwn')
                # self.scroll_to_element(drv, sort_dropdown)
                # ActionChains(drv).move_to_element(
                #     sort_dropdown).click(sort_dropdown).perform()
                # button = drv.find_element_by_xpath(
                #     '//*[@id="cat_sort_menu"]/button[3]')
                # drv.implicitly_wait(4)
                # self.scroll_to_element(drv, button)
                # ActionChains(drv).move_to_element(
                #     button).click(button).perform()
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod Type: {product_type}')
                self.logger.info(str.encode(
                    f'Category: {cat_name} - ProductType {product_type} cannot sort by NEW.(page link: {product_type_link})',
                    'utf-8', 'ignore'))
                pass

            # load all the products
            self.scroll_down_page(drv, h2=0.8, speed=3)
            time.sleep(8)

            # check whether on the first page of product type
            try:
                close_popups(drv)
                accept_alert(drv, 2)
                current_page = drv.find_element_by_class_name(
                    'css-g48inl').text
            except NoSuchElementException as ex:
                log_exception(self.logger,
                              additional_information=f'Prod Type: {product_type}')
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType {product_type} has\
                only one page of products.(page link: {product_type_link})', 'utf-8', 'ignore'))
                one_page = True
                current_page = 1
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod Type: {product_type}')
                self.progress_tracker.loc[pt, 'scraped'] = 'NA'
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType {product_type}\
                     page not found.(page link: {product_type_link})',
                                            'utf-8', 'ignore'))
            else:
                # get a list of all available pages
                one_page = False
                # get next page button
                next_page_button = drv.find_element_by_class_name(
                    'css-1lkjxdl')
                pages = []
                for page in drv.find_elements_by_class_name('css-1lk9n5p'):
                    pages.append(page.text)

            # start getting product form each page
            while True:
                cp = 0
                self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type}\
                                  getting product from page {current_page}.(page link: {product_type_link})',
                                            'utf-8', 'ignore'))
                time.sleep(5)
                close_popups(drv)
                accept_alert(drv, 2)
                products = drv.find_elements_by_css_selector(
                    'div[data-comp="ProductGrid "]>div>div>a')
                # print(len(products))

                for p in products:
                    time.sleep(0.5)

                    close_popups(drv)
                    accept_alert(drv, 0.5)

                    self.scroll_to_element(drv, p)
                    ActionChains(drv).move_to_element(p).perform()

                    try:
                        product_name = p.get_attribute('aria-label')
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod Type: {product_type}')
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(p)} metadata extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})',
                                                    'utf-8', 'ignore'))
                        continue

                    try:
                        product_new_flag = p.find_element_by_css_selector(
                            'div[data-at="product_badges"]').text
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod Type: {product_type}')
                        product_new_flag = ''
                        # self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                        #                              product {products.index(p)} product_new_flag extraction failed.\
                        #                         (page_link: {product_type_link} - page_no: {current_page})', 'utf-8', 'ignore'))
                    try:
                        product_page = p.get_attribute('href')
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod Type: {product_type}')
                        product_page = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(p)} product_page extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})',
                                                    'utf-8', 'ignore'))
                    try:
                        brand = p.find_element_by_css_selector(
                            'span[data-at="sku_item_brand"]').text
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod Type: {product_type}')
                        brand = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(p)} brand extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})',
                                                    'utf-8', 'ignore'))
                    try:
                        rating = p.find_element_by_css_selector(
                            'div[data-comp="StarRating "]').get_attribute('aria-label')
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod Type: {product_type}')
                        rating = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                     product {products.index(p)} rating extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})',
                                                    'utf-8', 'ignore'))
                    try:
                        price = p.find_element_by_css_selector(
                            'span[data-at="sku_item_price_list"]').text
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod Type: {product_type}')
                        price = ''
                        self.logger.info(str.encode(f'Category: {cat_name} - ProductType: {product_type} -\
                                                      product {products.index(p)} price extraction failed.\
                                                (page_link: {product_type_link} - page_no: {current_page})',
                                                    'utf-8', 'ignore'))

                    if datetime.now().day < 15:
                        meta_date = f'{time.strftime("%Y-%m")}-01'
                    else:
                        meta_date = f'{time.strftime("%Y-%m")}-15'

                    product_data_dict = {"product_name": product_name, "product_page": product_page, "brand": brand, "price": price,
                                         "rating": rating, "category": cat_name, "product_type": product_type, "new_flag": product_new_flag,
                                         "complete_scrape_flag": "N", "meta_date": meta_date}
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
                        self.scroll_to_element(drv, next_page_button)
                        ActionChains(drv).move_to_element(
                            next_page_button).click(next_page_button).perform()
                        time.sleep(15)
                        accept_alert(drv, 10)
                        close_popups(drv)
                        self.scroll_down_page(drv, h2=0.8, speed=3)
                        time.sleep(10)
                        current_page = drv.find_element_by_class_name(
                            'css-g48inl').text
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod Type: {product_type}')
                        self.logger.info(str.encode(f'Page navigation issue occurred for Category: {cat_name} - \
                                                        ProductType: {product_type} (page_link: {product_type_link} \
                                                        - page_no: {current_page})', 'utf-8', 'ignore'))
                        break
            drv.quit()

            if len(product_meta_data) > 0:
                product_meta_df = pd.DataFrame(product_meta_data)
                product_meta_df.to_feather(
                    self.current_progress_path/f'sph_prod_meta_extract_progress_{product_type}_{time.strftime("%Y-%m-%d-%H%M%S")}')
                self.logger.info(
                    f'Completed till IndexPosition: {pt} - ProductType: {product_type}. (URL:{product_type_link})')
                self.product_type_urls.loc[pt, 'scraped'] = 'Y'
                self.product_type_urls.to_feather(
                    self.metadata_path/'sph_product_type_urls_to_extract')
                product_meta_data = []
        self.logger.info('Metadata Extraction Complete')
        print('Metadata Extraction Complete')

    def extract(self, download: bool = True, fresh_start: bool = False, auto_fresh_start: bool = False, n_workers: int = 5,
                open_headless: bool = False, open_with_proxy_server: bool = True, randomize_proxy_usage: bool = True,
                start_idx: Optional[int] = None, end_idx: Optional[int] = None, list_of_index=None,
                compile_progress_files: bool = False, clean: bool = True, delete_progress: bool = False) -> None:
        """extract method controls all properties of the spiders and runs multi-threaded web crawling.

        Extract is exposes all functionality of the spiders and user needs to run this method to begin data crawling from web.
        This method has four major functionality:
        * 1. Run the spider
        * 2. Store data in regular intervals to free up ram
        * 3. Compile all crawled data into one file.
        * 4. Clean and push cleaned data to S3 storage for further algorithmic processing.

        Args:
            download (bool, optional): Whether to crawl data from or compile crawled data into one file. Defaults to True.
            fresh_start (bool, optional): Whether to continue last crawl job or start new one. Defaults to False.
            auto_fresh_start (bool, optional): Whether to automatically start a new crawl job if last job was finished.
                                               Defaults to False.
            n_workers (int, optional): No. of parallel threads to run. Defaults to 5.
            open_headless (bool, optional): Whether to open browser headless. Defaults to False.
            open_with_proxy_server (bool, optional): Whether to use ip rotation service. Defaults to True.
            randomize_proxy_usage (bool, optional): Whether to use both proxy and native network in tandem to decrease proxy requests.
                                                    Defaults to True.
            start_idx (Optional[int], optional): Starting index of the links to crawl. Defaults to None.
            end_idx (Optional[int], optional): Ending index of the links to crawl. Defaults to None.
            list_of_index ([type], optional): List of indices or range of indices of product urls to scrape. Defaults to None.
            compile_progress_files (bool, optional): Whether to combine crawled data into one file. Defaults to False.
            clean (bool, optional): Whether to clean the compiled data. Defaults to True.
            delete_progress (bool, optional): Whether to delete intermediate data after compilation into one file. Defaults to False.
        """
        def fresh():
            """ If fresh_start is True, this function sets initial parameters for a fresh data crawl.
            """
            self.product_type_urls = self.get_product_type_urls(open_headless=open_headless,
                                                                open_with_proxy_server=open_with_proxy_server)
            # # progress tracker: captures scraped and error desc
            # self.progress_tracker = pd.DataFrame(index=self.product_type_urls.index, columns=[
            #     'product_type', 'scraped', 'error_desc'])
            # self.progress_tracker.scraped = 'N'
            self.product_type_urls.scraped = 'N'

        if fresh_start:
            self.logger.info('Starting Fresh Extraction.')
            fresh()
        else:
            if Path(self.metadata_path/'sph_product_type_urls_to_extract').exists():
                self.product_type_urls = pd.read_feather(
                    self.metadata_path/'sph_product_type_urls_to_extract')
                if sum(self.product_type_urls.scraped == 'N') > 0:
                    self.logger.info(
                        'Continuing Metadata Extraction From Last Run.')
                    self.product_type_urls = self.product_type_urls[self.product_type_urls.scraped == 'N']
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
                "sph_prod_meta_extract_progress_*")]
            li = [pd.read_feather(file) for file in files]
            metadata_df = pd.concat(li, axis=0, ignore_index=True)
            metadata_df.reset_index(inplace=True, drop=True)
            metadata_df['source'] = self.source

            if datetime.now().day < 15:
                meta_date = f'{time.strftime("%Y-%m")}-01'
            else:
                meta_date = f'{time.strftime("%Y-%m")}-15'
            filename = f'sph_product_metadata_all_{meta_date}'
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
        """terminate_logging ends log generation for the crawler and cleaner after the job finshes successfully.
        """
        self.logger.handlers.clear()
        self.prod_meta_log.stop_log()

    # def extract_failed_pages(self):
    #     """extract_failed_pages [summary]

    #     [extended_summary]
    #     """
    #     pass


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

    def get_images(self, indices: Union[list, range], open_headless: bool,
                   open_with_proxy_server: bool,
                   randomize_proxy_usage: bool) -> None:
        """get_images scrapes individual product images.

        Get Images download up to four images for on product.

        Args:
            indices (Union[list, range]): list of indices or range of indices of product urls to scrape.
            open_headless (bool): Whether to open browser headless.
            open_with_proxy_server (bool): Whether to use ip rotation service.
            randomize_proxy_usage (bool): Whether to use both proxy and native network in tandem to decrease proxy requests.
        """

        for prod in self.meta.index[self.meta.index.isin(indices)]:
            if self.meta.loc[prod, 'image_scraped'] in ['Y', 'NA']:
                continue
            prod_id = self.meta.loc[prod, 'prod_id']
            product_page = self.meta.loc[prod, 'product_page']

            # create webdriver
            if randomize_proxy_usage:
                use_proxy = np.random.choice([True, False])
            else:
                use_proxy = True
            if open_with_proxy_server:
                # print(use_proxy)
                drv = self.open_browser(open_headless=open_headless, open_with_proxy_server=use_proxy,
                                        open_for_screenshot=True, path=self.image_path)
            else:
                drv = self.open_browser(open_headless=open_headless, open_with_proxy_server=False,
                                        open_for_screenshot=True, path=self.image_path)
            # open product page
            drv.get(product_page)
            time.sleep(15)  # 30
            accept_alert(drv, 10)
            close_popups(drv)

            try:
                product_text = drv.find_element_by_class_name(
                    'css-1wag3se').text
                if 'productnotcarried' in product_text.lower():
                    self.logger.info(str.encode(f'prod_id: {prod_id} image extraction failed.\
                                            Product may not be available for sell currently.(page: {product_page})',
                                                'utf-8', 'ignore'))
                    self.meta.loc[prod, 'image_scraped'] = "NA"
                    self.meta.to_csv(
                        self.image_path/'sph_image_progress_tracker.csv', index=None)
                    drv.quit()
                    continue
            except Exception as ex:
                log_exception(
                    self.logger, additional_information=f'Prod ID: {prod_id}')

            # get image elements
            try:
                accept_alert(drv, 1)
                close_popups(drv)
                images = drv.find_elements_by_class_name('css-11rgy2w')
                if len(images) == 0:
                    try:
                        accept_alert(drv, 1)
                        close_popups(drv)
                        images = drv.find_elements_by_class_name('css-11rgy2w')
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod ID: {prod_id}')
                        self.logger.info(str.encode(f'prod_id: {prod_id} failed to get image sources.\
                                                    (page: {product_page})', 'utf-8', 'ignore'))
                        self.meta.loc[prod, 'image_scraped'] = "NA"
                        self.meta.to_csv(
                            self.image_path/'sph_image_progress_tracker.csv', index=None)
                        drv.quit()
                        continue
            except Exception:
                continue
            else:
                if len(images) == 0:
                    self.logger.info(str.encode(f'{prod_id} image extraction failed.\
                                                    (page: {product_page})', 'utf-8', 'ignore'))
                    self.meta.loc[prod, 'image_scraped'] = "NA"
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
            accept_alert(drv, 3)
            try:
                for src in sources:
                    drv.get(src)
                    time.sleep(2)
                    accept_alert(drv, 1)
                    close_popups(drv)
                    image_count += 1
                    image_name = f'{prod_id}_image_{image_count}.jpg'
                    drv.save_screenshot(
                        str(self.current_image_path/image_name))
                self.meta.loc[prod, 'image_scraped'] = 'Y'
                drv.quit()
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')
                if image_count <= 1:
                    self.logger.info(str.encode(f'prod_id: {prod_id} image extraction failed.\
                                                    (page: {product_page})', 'utf-8', 'ignore'))
                    self.meta.loc[prod, 'image_scraped'] = 'NA'
                    self.meta.to_csv(
                        self.image_path/'sph_image_progress_tracker.csv', index=None)
                drv.quit()
                continue

            if prod % 10 == 0 and prod != 0:
                self.meta.to_csv(
                    self.image_path/'sph_image_progress_tracker.csv', index=None)
        self.meta.to_csv(
            self.image_path/'sph_image_progress_tracker.csv', index=None)

    def extract(self, metadata: Union[pd.DataFrame, str, Path], download: bool = True,
                n_workers: int = 5, fresh_start: bool = False, auto_fresh_start: bool = False,
                open_headless: bool = False, open_with_proxy_server: bool = True,
                randomize_proxy_usage: bool = True,
                start_idx: Optional[int] = None, end_idx: Optional[int] = None, list_of_index=None,
                ):
        """extract method controls all properties of the spiders and runs multi-threaded web crawling.

        Extract is exposes all functionality of the spiders and user needs to run this method to begin data crawling from web.
        This method has four major functionality:
        * 1. Run the spider
        * 2. Store data in regular intervals to free up ram
        * 3. Compile all crawled data into one file.
        * 4. Clean and push cleaned data to S3 storage for further algorithmic processing.

        Args:
            metadata (pd.DataFrame): Dataframe containing product specific url, name and id of the products to be scraped.
            download (bool, optional): Whether to crawl data from or compile crawled data into one file. Defaults to True.
            n_workers (int, optional): No. of parallel threads to run. Defaults to 5.
            fresh_start (bool, optional): Whether to continue last crawl job or start new one. Defaults to False.
            auto_fresh_start (bool, optional): Whether to automatically start a new crawl job if last job was finished. Defaults to False.
            open_headless (bool, optional):  Whether to open browser headless. Defaults to False.
            open_with_proxy_server (bool, optional): Whether to use ip rotation service. Defaults to True.
            randomize_proxy_usage (bool, optional): Whether to use both proxy and native network in tandem to decrease proxy requests.
                                                    Defaults to False.
            start_idx (Optional[int], optional): Starting index of the links to crawl. Defaults to None.
            end_idx (Optional[int], optional): Ending index of the links to crawl. Defaults to None.
            list_of_index ([type], optional): List of indices or range of indices of product urls to scrape. Defaults to None.
            compile_progress_files (bool, optional): Whether to combine crawled data into one file. Defaults to False.
            clean (bool, optional): Whether to clean the compiled data. Defaults to True.
            delete_progress (bool, optional): Whether to delete intermediate data after compilation into one file. Defaults to False.
        """
        def fresh():
            self.meta = metadata[['prod_id', 'product_page']]
            self.meta['image_scraped'] = 'N'

        if download:
            if fresh_start:
                fresh()
            else:
                if Path(self.image_path/'sph_image_progress_tracker.csv').exists():
                    self.meta = pd.read_csv(
                        self.image_path/'sph_image_progress_tracker.csv', na_filter=False)
                    if sum(self.meta.image_scraped == 'N') == 0:
                        if auto_fresh_start:
                            fresh()
                            self.logger.info(
                                'Last Run was Completed. Starting Fresh Extraction.')
                        else:
                            self.logger.info(
                                f'Image extraction for this cycle is complete. \
                                    Please check files in path: {self.image_path}')
                            print(
                                f'Image extraction for this cycle is complete. \
                                    Please check files in path: {self.image_path}')
                    else:
                        self.logger.info(
                            'Continuing Image Extraction From Last Run.')

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

            if list_of_index:
                self.get_images(
                    indices=list_of_index, open_headless=open_headless,
                    open_with_proxy_server=open_with_proxy_server,
                    randomize_proxy_usage=randomize_proxy_usage)
            else:  # By default the code will with 5 concurrent threads. you can change this behaviour by changing n_workers
                if start_idx:
                    lst_of_lst = ranges(
                        indices[-1]+1, n_workers, start_idx=start_idx)
                else:
                    lst_of_lst = ranges(len(indices), n_workers)
                print(lst_of_lst)

                headless = [open_headless for i in lst_of_lst]
                proxy = [open_with_proxy_server for i in lst_of_lst]
                rand_proxy = [randomize_proxy_usage for i in lst_of_lst]

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.map(self.get_images, lst_of_lst, headless,
                                 proxy, rand_proxy)

        self.logger.info(
            f'Image files are downloaded to product specific folders. \
            Please look for file sph_product_review_all in path {self.image_path}')
        print(
            f'Image files are downloaded to product specific folders. \
            Please look for file sph_product_review_all in path {self.image_path}')

    def terminate_logging(self) -> None:
        """terminate_logging ends log generation for the crawler and cleaner after the job finshes successfully.
        """
        self.logger.handlers.clear()
        self.prod_image_log.stop_log()


class DetailReview(Sephora):
    """DetailReview class carries out scraping of Sephora Detail and Review data.

    Args:
        Sephora (Browser): Class that initializes folder paths and selenium webdriver for data scraping.

    """

    def __init__(self, log: bool = True, path: Path = Path.cwd()):
        """__init__ DetailReview class instace initializer.

        This method sets all the folder paths required for DetailReview crawler to work.
        If the paths does not exist the paths get automatically created depending on current directory
        or provided directory.

        Initializer Tasks:
        1. Set folder paths for data reading and storing.
        2. Move old scraped and cleaned data files to archive folders before starting new extraction.
        3. Create data scraping log files in log path to monitor/correct failed data extraction from website.

        Args:
            log (bool, optional): Whether to create crawling exception and progess log. Defaults to True.
            path (Path, optional): Folder path where the Detail and Review data will be extracted.
                                   Defaults to current directory(Path.cwd()).

        """
        super().__init__(path=path, data_def='detail_review_image')
        self.path = Path(path)
        self.detail_current_progress_path = self.detail_path/'current_progress'
        self.detail_current_progress_path.mkdir(parents=True, exist_ok=True)

        self.review_current_progress_path = self.review_path/'current_progress'
        self.review_current_progress_path.mkdir(parents=True, exist_ok=True)

        old_detail_files = list(self.detail_path.glob(
            'sph_product_detail_all*')) + list(self.detail_path.glob(
                'sph_product_item_all*'))
        for f in old_detail_files:
            shutil.move(str(f), str(self.old_detail_files_path))

        old_clean_detail_files = os.listdir(self.detail_clean_path)
        for f in old_clean_detail_files:
            shutil.move(str(self.detail_clean_path/f),
                        str(self.old_detail_clean_files_path))

        old_review_files = list(self.review_path.glob(
            'sph_product_review_all*'))
        for f in old_review_files:
            shutil.move(str(f), str(self.old_review_files_path))

        old_clean_review_files = os.listdir(self.review_clean_path)
        for f in old_clean_review_files:
            shutil.move(str(self.review_clean_path/f),
                        str(self.old_review_clean_files_path))
        if log:
            self.prod_detail_review_image_log = Logger(
                "sph_prod_detail_review_image_extraction", path=self.crawl_log_path)
            self.logger, _ = self.prod_detail_review_image_log.start_log()

    def get_details(self, drv: webdriver.Chrome, prod_id: str, product_name: str) -> Tuple[dict, pd.DataFrame]:
        """get_detail scrapes individual product pages for price, ingredients, color etc.

        Get Detail crawls product specific page and scrapes data such as ingredients, review rating distribution,
        size specific prices, color, product claims and other information pertaining to one individual product.

        Args:
            drv (webdriver.Chrome): Selenium webdriver with opened product page.
            prod_id (str): Id of the product from metdata.
            product_name (str): Name of the product from metadata.

        Returns:
            Tuple[dict, pd.DataFrame]: Dictionary containing details of a product and dataframe containing item
                                       attributes of a product.
        """

        def get_item_attributes(drv: webdriver.Chrome, product_name: str, prod_id: str,
                                multi_variety: bool = False, typ=None, ) -> Tuple[str, str, str, str]:
            """get_item_attributes extracts product type specific information.

            Args:
                drv (webdriver.Chrome): Selenium webdriver with opened product page.
                product_name (str): Name of the product from metadata.
                prod_id (str): Id of the product from metdata.
                multi_variety (bool, optional): Whether product has multiple colors/sizes. Defaults to False.
                typ ([type], optional): Webelement corresponding to a particular product type. Defaults to None.

            Returns:
                Tuple[str, str, str, str]: item_name, item_size, item_price, item_ingredient

            """
            # drv # type: webdriver.Chrome
            # close popup windows
            close_popups(drv)
            accept_alert(drv, 1)

            try:
                item_price = drv.find_element_by_css_selector(
                    'span[data-at="price"]').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')
                item_price = ''
            # print(item_price)

            if multi_variety:
                try:
                    item_name = typ.get_attribute('aria-label')
                    # print(item_name)
                except Exception as ex:
                    log_exception(self.logger,
                                  additional_information=f'Prod ID: {prod_id}')
                    item_name = ""
                    self.logger.info(str.encode(
                        f'product: {product_name} (prod_id: {prod_id}) item_name does not exist.', 'utf-8', 'ignore'))
            else:
                item_name = ""

            try:
                item_size = drv.find_element_by_css_selector(
                    'div[data-at="sku_size"]').text
                # print(item_size)
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')
                item_size = ""
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) item_size does not exist.', 'utf-8', 'ignore'))

            # get all tabs
            first_tab = drv.find_element_by_id(f'tab{0}')
            self.scroll_to_element(drv, first_tab)
            ActionChains(drv).move_to_element(
                first_tab).click(first_tab).perform()

            prod_tabs = []
            prod_tabs = drv.find_elements_by_css_selector(
                'div[aria-label="Product Information"]>button>span')
            # prod_tabs.extend(drv.find_elements_by_class_name('css-12vae0p'))

            tab_names = []
            for t in prod_tabs:
                tab_names.append(t.text.lower())
            # print(tab_names)

            if 'ingredients' in tab_names:
                close_popups(drv)
                accept_alert(drv, 1)

                tab_num = tab_names.index('ingredients')
                ing_button = drv.find_element_by_id(f'tab{tab_num}')
                self.scroll_to_element(drv, ing_button)
                ActionChains(drv).move_to_element(
                    ing_button).click(ing_button).perform()
                item_ing = drv.find_element_by_xpath(
                    f'//*[@id="tabpanel{tab_num}"]/div').text
            else:
                item_ing = ""
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) item_ingredients does not exist.', 'utf-8', 'ignore'))
            # print(item_ing)
            return item_name, item_size, item_price, item_ing

        def get_product_attributes(drv: webdriver.Chrome, product_name: str, prod_id: str) -> list:
            """get_product_attributes uses get_item_attribute method to scrape item details and stores
            in a list which is returned to get_detail method for storing in a product specific dataframe.

            Args:
                drv (webdriver.Chrome): Selenium webdriver with opened product page.
                product_name (str): Name of the product from metadata.
                prod_id (str): Id of the product from metdata.

            Returns:
                list: List containing all product item attributes of multiple varieties with name and id.

            """
            # get all the variation of product
            # close popup windows
            close_popups(drv)
            accept_alert(drv, 1)

            product_variety = []
            try:
                product_variety = drv.find_elements_by_css_selector(
                    'button[data-at="selected_swatch"]')
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')

            product_attributes = []

            if len(product_variety) > 0:
                for typ in product_variety:
                    close_popups(drv)
                    accept_alert(drv, 1)
                    try:
                        self.scroll_to_element(drv, typ)
                        ActionChains(drv).move_to_element(
                            typ).click(typ).perform()
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod ID: {prod_id}')
                    time.sleep(4)  # 8
                    item_name, item_size, item_price, item_ingredients = get_item_attributes(drv, product_name, prod_id,
                                                                                             multi_variety=True, typ=typ)
                    product_attributes.append({"prod_id": prod_id,
                                               "product_name": product_name,
                                               "item_name": item_name,
                                               "item_size": item_size,
                                               "item_price": item_price,
                                               "item_ingredients": item_ingredients
                                               })
            else:
                item_name, item_size, item_price, item_ingredients = get_item_attributes(drv,
                                                                                         product_name, prod_id)
                product_attributes.append({"prod_id": prod_id,
                                           "product_name": product_name,
                                           "item_name": item_name,
                                           "item_size": item_size,
                                           "item_price": item_price,
                                           "item_ingredients": item_ingredients
                                           })

            return product_attributes

        '''
        def get_first_review_date(drv: webdriver.Chrome) -> str:
            """get_first_review_date scrapes the first review data of a product.

            Args:
                drv (webdriver.Chrome): Selenium webdriver with opened product page.

            Returns:
                str: First review date.

            """
            # close popup windows
            close_popups(drv)
            accept_alert(drv, 1)

            try:
                review_sort_trigger = drv.find_element_by_id(
                    'review_filter_sort_trigger')
                self.scroll_to_element(drv, review_sort_trigger)
                ActionChains(drv).move_to_element(
                    review_sort_trigger).click(review_sort_trigger).perform()
                for btn in drv.find_elements_by_class_name('css-rfz1gg'):
                    if btn.text.lower() == 'oldest':
                        ActionChains(drv).move_to_element(
                            btn).click(btn).perform()
                        break
                time.sleep(6)
                close_popups(drv)
                accept_alert(drv, 1)
                rev = drv.find_elements_by_class_name('css-1kk8dps')[2:]
                try:
                    first_review_date = convert_ago_to_date(
                        rev[0].find_element_by_class_name('css-h2vfi1').text)
                except Exception as ex:
                    log_exception(self.logger,
                                  additional_information=f'Prod ID: {prod_id}')
                    try:
                        first_review_date = convert_ago_to_date(
                            rev[1].find_element_by_class_name('css-h2vfi1').text)
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod ID: {prod_id}')
                        print('sorted but cant get first review date value')
                        first_review_date = ''
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')
                first_review_date = ''
            return first_review_date
        '''

        # get all product info tabs such as how-to-use, about-brand, ingredients
        prod_tabs = []
        prod_tabs = prod_tabs = drv.find_elements_by_css_selector(
            'div[aria-label="Product Information"]>button>span')

        tab_names = []
        for t in prod_tabs:
            tab_names.append(t.text.lower())

        accept_alert(drv, 1)
        try:
            votes = drv.find_elements_by_class_name('css-2rg6q7')[-1].text
        except Exception as ex:
            log_exception(self.logger,
                          additional_information=f'Prod ID: {prod_id}')
            votes = ""
            self.logger.info(str.encode(
                f'product: {product_name} (prod_id: {prod_id}) votes does not exist.', 'utf-8', 'ignore'))

        # product details
        if 'details' in tab_names:
            try:
                close_popups(drv)
                accept_alert(drv, 1)
                tab_num = tab_names.index('details')
                detail_button = drv.find_element_by_id(f'tab{tab_num}')
                try:
                    time.sleep(1)
                    self.scroll_to_element(drv, detail_button)
                    ActionChains(drv).move_to_element(
                        detail_button).click(detail_button).perform()
                except Exception as ex:
                    log_exception(self.logger,
                                  additional_information=f'Prod ID: {prod_id}')
                    details = ""
                else:
                    try:
                        details = drv.find_element_by_xpath(
                            f'//*[@id="tabpanel{tab_num}"]/div').text
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod ID: {prod_id}')
                        details = ""
                        self.logger.info(str.encode(
                            f'product: {product_name} (prod_id: {prod_id}) product detail text\
                                    does not exist.', 'utf-8', 'ignore'))
                    # print(details)
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')
                details = ""
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) product detail extraction failed', 'utf-8', 'ignore'))
        else:
            details = ""
            self.logger.info(str.encode(
                f'product: {product_name} (prod_id: {prod_id}) product detail does not exist.', 'utf-8', 'ignore'))

        # how to use
        if 'how to use' in tab_names:
            try:
                close_popups(drv)
                accept_alert(drv, 1)

                tab_num = tab_names.index('how to use')
                how_to_use_button = drv.find_element_by_id(f'tab{tab_num}')
                try:
                    time.sleep(1)
                    self.scroll_to_element(drv, how_to_use_button)
                    ActionChains(drv).move_to_element(
                        how_to_use_button).click(how_to_use_button).perform()
                except Exception as ex:
                    log_exception(self.logger,
                                  additional_information=f'Prod ID: {prod_id}')
                    how_to_use = ""
                else:
                    try:
                        how_to_use = drv.find_element_by_xpath(
                            f'//*[@id="tabpanel{tab_num}"]/div').text
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod ID: {prod_id}')
                        how_to_use = ""
                        self.logger.info(str.encode(
                            f'product: {product_name} (prod_id: {prod_id}) how_to_use text\
                                    does not exist.', 'utf-8', 'ignore'))
                    # print(how_to_use)
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')
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
                close_popups(drv)
                accept_alert(drv, 1)

                tab_num = tab_names.index('about the brand')
                about_the_brand_button = drv.find_element_by_id(
                    f'tab{tab_num}')
                try:
                    time.sleep(1)
                    self.scroll_to_element(drv, about_the_brand_button)
                    ActionChains(drv).move_to_element(
                        about_the_brand_button).click(about_the_brand_button).perform()
                except Exception as ex:
                    log_exception(self.logger,
                                  additional_information=f'Prod ID: {prod_id}')
                    about_the_brand = ""
                else:
                    try:
                        about_the_brand = drv.find_element_by_xpath(
                            f'//*[@id="tabpanel{tab_num}"]/div').text
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod ID: {prod_id}')
                        about_the_brand = ""
                        self.logger.info(str.encode(
                            f'product: {product_name} (prod_id: {prod_id}) about_the_brand text\
                                does not exist', 'utf-8', 'ignore'))
                    # print(about_the_brand)
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')
                about_the_brand = ""
                self.logger.info(str.encode(
                    f'product: {product_name} (prod_id: {prod_id}) about_the_brand extraction failed', 'utf-8', 'ignore'))
        else:
            about_the_brand = ""
            self.logger.info(str.encode(
                f'product: {product_name} (prod_id: {prod_id}) about_the_brand does not exist.', 'utf-8', 'ignore'))

        self.scroll_down_page(drv, h2=0.4, speed=5)
        time.sleep(5)
        accept_alert(drv, 1)
        try:
            chat_popup_button = WebDriverWait(drv, 3).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="divToky"]/img[3]')))
            chat_popup_button = drv.find_element_by_xpath(
                '//*[@id="divToky"]/img[3]')
            self.scroll_to_element(drv, chat_popup_button)
            ActionChains(drv).move_to_element(
                chat_popup_button).click(chat_popup_button).perform()
        except TimeoutException:
            pass
        # click no. of reviews
        try:
            review_button = drv.find_element_by_css_selector(
                'span[data-at="number_of_reviews"]')
            self.scroll_to_element(drv, review_button)
            ActionChains(drv).move_to_element(
                review_button).click(review_button).perform()
        except Exception as ex:
            log_exception(self.logger,
                          additional_information=f'Prod ID: {prod_id}')

        try:
            first_review_date = ''
            # print(first_review_date)
        except Exception as ex:
            log_exception(self.logger,
                          additional_information=f'Prod ID: {prod_id}')
            first_review_date = ''
            self.logger.info(str.encode(
                f'product: {product_name} (prod_id: {prod_id}) first_review_date scrape failed.', 'utf-8', 'ignore'))

        try:
            close_popups(drv)
            accept_alert(drv, 1)
            reviews = int(drv.find_element_by_class_name(
                'css-ils4e4').text.split()[0])
            # print(reviews)
        except Exception as ex:
            log_exception(self.logger,
                          additional_information=f'Prod ID: {prod_id}')
            reviews = 0
            self.logger.info(str.encode(
                f'product: {product_name} (prod_id: {prod_id}) reviews does not exist.', 'utf-8', 'ignore'))

        try:
            close_popups(drv)
            accept_alert(drv, 1)
            rating_distribution = drv.find_element_by_css_selector(
                'table[data-comp="HistogramChart "]').text.split('\n')
            # print(rating_distribution)
        except Exception as ex:
            log_exception(self.logger,
                          additional_information=f'Prod ID: {prod_id}')
            rating_distribution = ""
            self.logger.info(str.encode(
                f'product: {product_name} (prod_id: {prod_id}) rating_distribution does not exist.', 'utf-8', 'ignore'))

        try:
            close_popups(drv)
            accept_alert(drv, 1)
            would_recommend = drv.find_element_by_class_name(
                'css-k9ne19').text
            # print(would_recommend)
        except Exception as ex:
            log_exception(self.logger,
                          additional_information=f'Prod ID: {prod_id}')
            would_recommend = ""
            self.logger.info(str.encode(
                f'product: {product_name} (prod_id: {prod_id}) would_recommend does not exist.', 'utf-8', 'ignore'))

        detail = {'prod_id': prod_id, 'product_name': product_name, 'abt_product': details,
                  'how_to_use': how_to_use, 'abt_brand': about_the_brand,
                  'reviews': reviews, 'votes': votes, 'rating_dist': rating_distribution,
                  'would_recommend': would_recommend, 'first_review_date': first_review_date}

        item = pd.DataFrame(
            get_product_attributes(drv, product_name, prod_id))

        return detail, item

    def get_reviews(self,  drv: webdriver.Chrome, prod_id: str, product_name: str, product_page: str,
                    last_scraped_review_date: str, no_of_reviews: int,
                    incremental: bool = True, reviews: list = []) -> list:
        """get_reviews crawls individual product pages for review text, title, date user attributes etc.

        Args:
            drv (webdriver.Chrome): drv (webdriver.Chrome): Selenium webdriver with opened product page.
            prod_id (str): Id of the product from metdata.
            product_name (str): Name of the product from metadata.
            product_page (str): Url of the product page.
            last_scraped_review_date (str): The last review date scraped as per database.
            no_of_reviews (int): No.of reviews a product has.
            incremental (bool, optional): Whether to scrape reviews incrementally from last scraped review date. Defaults to True.
            reviews (list, optional): Empty intermediate list to store data during parallel crawl. Defaults to []

        Returns:
            list: List of dictionaries containing all the scraped reviews of a product.

        """
        # print(no_of_reviews)
        # drv.find_element_by_class_name('css-2rg6q7').click()
        if incremental and last_scraped_review_date != '':
            accept_alert(drv, 1)
            for n in range(no_of_reviews//6):
                if n > 400:
                    break

                time.sleep(0.4)
                revs = drv.find_elements_by_class_name(
                    'css-1kk8dps')[2:]

                try:
                    if pd.to_datetime(convert_ago_to_date(revs[-1].find_element_by_class_name('css-1t84k9w').text),
                                      infer_datetime_format=True)\
                            < pd.to_datetime(last_scraped_review_date, infer_datetime_format=True):
                        # print('breaking incremental')
                        break
                except Exception as ex:
                    log_exception(self.logger,
                                  additional_information=f'Prod ID: {prod_id}')
                    try:
                        if pd.to_datetime(convert_ago_to_date(revs[-2].find_element_by_class_name('css-1t84k9w').text),
                                          infer_datetime_format=True)\
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
                try:
                    show_more_review_button = drv.find_element_by_class_name(
                        'css-xswy5p')
                except Exception as ex:
                    log_exception(self.logger,
                                  additional_information=f'Prod ID: {prod_id}. Failed to get show more review button.')
                else:
                    try:
                        self.scroll_to_element(
                            drv, show_more_review_button)
                        ActionChains(drv).move_to_element(
                            show_more_review_button).click(show_more_review_button).perform()
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod ID: {prod_id}. Failed to click on show more review button.')
                        accept_alert(drv, 1)
                        close_popups(drv)
                        try:
                            self.scroll_to_element(
                                drv, show_more_review_button)
                            ActionChains(drv).move_to_element(
                                show_more_review_button).click(show_more_review_button).perform()
                        except Exception as ex:
                            log_exception(self.logger,
                                          additional_information=f'Prod ID: {prod_id}. Failed to click on show more review button.')

        else:
            # print('inside get all reviews')
            # 6 because for click sephora shows 6 reviews. additional 25 no. of clicks for buffer.
            accept_alert(drv, 1)
            for n in range(no_of_reviews//6+10):
                '''
                code will stop after getting 1800 reviews of one particular product
                when crawling all reviews. By default it will get latest 1800 reviews.
                then in subsequent incremental runs it will get al new reviews on weekly basis
                '''
                if n >= 300:  # 200:
                    break
                time.sleep(1)
                # close any opened popups by escape
                accept_alert(drv, 1)
                close_popups(drv)
                try:
                    show_more_review_button = drv.find_element_by_class_name(
                        'css-xswy5p')
                except Exception as ex:
                    log_exception(self.logger,
                                  additional_information=f'Prod ID: {prod_id}. Failed to get show more review button.')
                else:
                    try:
                        self.scroll_to_element(
                            drv, show_more_review_button)
                        ActionChains(drv).move_to_element(
                            show_more_review_button).click(show_more_review_button).perform()
                    except Exception as ex:
                        log_exception(self.logger,
                                      additional_information=f'Prod ID: {prod_id}. Failed to click on show more review button.')
                        accept_alert(drv, 1)
                        close_popups(drv)
                        try:
                            self.scroll_to_element(
                                drv, show_more_review_button)
                            ActionChains(drv).move_to_element(
                                show_more_review_button).click(show_more_review_button).perform()
                        except Exception as ex:
                            log_exception(self.logger,
                                          additional_information=f'Prod ID: {prod_id}.\
                                                Failed to click on show more review button.')
                            try:
                                self.scroll_to_element(
                                    drv, show_more_review_button)
                                ActionChains(drv).move_to_element(
                                    show_more_review_button).click(show_more_review_button).perform()
                            except Exception as ex:
                                log_exception(self.logger,
                                              additional_information=f'Prod ID: {prod_id}.\
                                                Failed to click on show more review button.')
                                accept_alert(drv, 2)
                                close_popups(drv)
                                try:
                                    self.scroll_to_element(
                                        drv, show_more_review_button)
                                    ActionChains(drv).move_to_element(
                                        show_more_review_button).click(show_more_review_button).perform()
                                except Exception as ex:
                                    log_exception(self.logger,
                                                  additional_information=f'Prod ID: {prod_id}. Failed to click on show more review button.')
                                    if n < (no_of_reviews//6):
                                        self.logger.info(str.encode(f'Product: {product_name} - prod_id \
                                            {prod_id} breaking click next review loop.\
                                                                    [total_reviews:{no_of_reviews} loaded_reviews:{n}]\
                                                                    (page link: {product_page})', 'utf-8', 'ignore'))
                                        self.logger.info(str.encode(f'Product: {product_name} - prod_id {prod_id} cant load all reviews.\
                                                                        Check click next 6 reviews\
                                                                        code section(page link: {product_page})', 'utf-8', 'ignore'))
                                    break

        accept_alert(drv, 2)
        close_popups(drv)

        product_reviews = drv.find_elements_by_css_selector(
            'div[data-comp="Review "]')

        # print('starting extraction')
        r = 0
        for rev in product_reviews:
            accept_alert(drv, 0.5)
            close_popups(drv)
            self.scroll_to_element(drv, rev)
            ActionChains(drv).move_to_element(rev).perform()

            try:
                try:
                    review_text = rev.find_element_by_class_name(
                        'css-1jg2pb9').text
                except NoSuchElementException:
                    review_text = rev.find_element_by_class_name(
                        'css-429528').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract review_text. Skip review.')
                continue

            try:
                review_date = convert_ago_to_date(
                    rev.find_element_by_css_selector('span[data-at="time_posted"]').text)
                if pd.to_datetime(review_date, infer_datetime_format=True) <= pd.to_datetime(last_scraped_review_date, infer_datetime_format=True):
                    continue
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract review_date.')
                review_date = ''

            try:
                review_title = rev.find_element_by_class_name(
                    'css-1jfmule').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract review_title.')
                review_title = ''

            try:
                accept_alert(drv, 0.5)
                product_variant = rev.find_element_by_class_name(
                    'css-1op1cn7').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract product_variant.')
                product_variant = ''

            try:
                user_rating = rev.find_element_by_css_selector(
                    'div[data-comp="StarRating "]').get_attribute('aria-label')
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract user_rating.')
                user_rating = ''

            try:
                accept_alert(drv, 0.5)
                user_attribute = [{'_'.join(u.lower().split()[0:-1]): u.lower().split()[-1]}
                                  for u in rev.find_element_by_class_name('css-ecreye').text.split('\n')]
                # user_attribute = []
                # for u in rev.find_elements_by_class_name('css-j5yt83'):
                #     user_attribute.append(
                #         {'_'.join(u.text.lower().split()[0:-1]): u.text.lower().split()[-1]})
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract user_attribute.')
                user_attribute = []

            try:
                recommend = rev.find_element_by_class_name(
                    'css-1tf5yph').text
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract recommend.')
                recommend = ''

            try:
                helpful = rev.find_element_by_class_name('css-b7zg5r').text
                # helpful = []
                # for h in rev.find_elements_by_class_name('css-39esqn'):
                #     helpful.append(h.text)
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}. Failed to extract helpful.')
                helpful = ''

            reviews.append({'prod_id': prod_id,
                            'product_name': product_name,
                            'user_attribute': user_attribute,
                            'product_variant': product_variant,
                            'review_title': review_title,
                            'review_text': review_text,
                            'review_rating': user_rating,
                            'recommend': recommend,
                            'review_date': review_date,
                            'helpful': helpful
                            })
        return reviews

    def crawl_page(self, indices: list, open_headless: bool, open_with_proxy_server: bool,
                   randomize_proxy_usage: bool, detail_data: list = [],
                   item_df=pd.DataFrame(columns=['prod_id', 'product_name', 'item_name',
                                                 'item_size', 'item_price',
                                                 'item_ingredients']),
                   review_data: list = [], incremental: bool = True):
        """crawl_page opens each product url in a sepearate browser to scrape detail and review data together.

        The scraping happens in a multi-threaded manner to extract product information of up to 20 products simultaneously.

        Args:
            indices (list): list of indices or range of indices of product urls to scrape.
            open_headless (bool): Whether to open browser headless.
            open_with_proxy_server (bool): Whether to use ip rotation service.
            randomize_proxy_usage (bool): Whether to use both proxy and native network in tandem to decrease proxy requests.
            detail_data (list, optional): Empty intermediate list to store data during parallel crawl. Defaults to [].
            item_df ([type], optional): Empty intermediate dataframe to store data during parallel crawl.
                                        Defaults to []. Defaults to pd.DataFrame(columns=['prod_id', 'product_name', 'item_name',
                                                                                         'item_size', 'item_price', 'item_ingredients']).
            review_data (list, optional): Empty intermediate list to store data during parallel crawl. Defaults to [].
            incremental (bool, optional): Whether to scrape reviews incrementally from last scraped review date. Defaults to True.

        """

        def store_data_refresh_mem(detail_data: list, item_df: pd.DataFrame,
                                   review_data: list) -> Tuple[list, pd.DataFrame, list]:
            """store_data_refresh_mem method stores crawled data in regular interval to free up system memory.

            Store data after five products are extracted every time to free up RAM.

            Args:
                detail_data (list): List containing crawled detail data to store.
                item_df (pd.DataFrame): Dataframe containing scraped item data to store.
                review_data (list): List containing scraped Review data to store.

            Returns:
                Tuple[list, pd.DataFrame, list]: Empty list, dataframe and list to accumulate data from next product scraping.

            """
            pd.DataFrame(detail_data).to_csv(self.detail_current_progress_path /
                                             f'sph_prod_detail_extract_progress_{time.strftime("%Y-%m-%d-%H%M%S")}.csv',
                                             index=None)
            item_df.reset_index(inplace=True, drop=True)
            item_df.to_csv(self.detail_current_progress_path /
                           f'sph_prod_item_extract_progress_{time.strftime("%Y-%m-%d-%H%M%S")}.csv',
                           index=None)
            item_df = pd.DataFrame(columns=[
                                   'prod_id', 'product_name', 'item_name', 'item_size', 'item_price', 'item_ingredients'])

            pd.DataFrame(review_data).to_csv(self.review_current_progress_path /
                                             f'sph_prod_review_extract_progress_{time.strftime("%Y-%m-%d-%H%M%S")}.csv',
                                             index=None)
            self.meta.to_csv(
                self.path/'sephora/sph_detail_review_image_progress_tracker.csv', index=None)
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

            try:
                # open product page
                drv.get(product_page)
                time.sleep(20)  # 30
                accept_alert(drv, 10)
                close_popups(drv)
                price = drv.find_element_by_css_selector(
                    'span[data-at="price"]').text
            except NoSuchElementException:
                try:
                    product_text = drv.find_element_by_class_name(
                        'css-1wag3se').text
                    if 'productnotcarried' in product_text.lower():
                        self.logger.info(str.encode(f'Product Name: {product_name}, Product ID: {prod_id} extraction failed.\
                                                Product may not be available for sell currently.(Page: {product_page})',
                                                    'utf-8', 'ignore'))
                        self.meta.loc[prod, 'scraped'] = 'NA'
                        self.meta.to_csv(
                            self.path/'sephora/sph_detail_review_image_progress_tracker.csv', index=None)
                        drv.quit()
                        continue
                except Exception as ex:
                    log_exception(
                        self.logger, additional_information=f'Prod ID: {prod_id}')
                    try:
                        drv.get(product_page)
                        time.sleep(20)  # 30
                        accept_alert(drv, 10)
                        close_popups(drv)
                        price = drv.find_element_by_css_selector(
                            'span[data-at="price"]').text
                    except Exception as ex:
                        try:
                            drv = self.open_browser(open_headless=open_headless, open_with_proxy_server=False,
                                                    path=self.detail_path)
                            drv.get(product_page)
                            time.sleep(20)  # 30
                            accept_alert(drv, 10)
                            close_popups(drv)
                            price = drv.find_element_by_css_selector(
                                'span[data-at="price"]').text
                        except Exception as ex:
                            log_exception(self.logger,
                                          additional_information=f'Prod ID: {prod_id}')
                            drv.quit()
                            self.logger.info(str.encode(
                                f'product: {product_name} (prod_id: {prod_id}) no longer exists in the previously fetched link \
                                    or loaded new style page. (link:{product_page})', 'utf-8', 'ignore'))
                            self.meta.loc[prod, 'scraped'] = 'NA'
                            continue

            self.scroll_down_page(drv, speed=6, h2=0.6)
            time.sleep(5)

            try:
                chat_popup_button = WebDriverWait(drv, 3).until(
                    EC.presence_of_element_located((By.XPATH, '//*[@id="divToky"]/img[3]')))
                chat_popup_button = drv.find_element_by_xpath(
                    '//*[@id="divToky"]/img[3]')
                self.scroll_to_element(drv, chat_popup_button)
                ActionChains(drv).move_to_element(
                    chat_popup_button).click(chat_popup_button).perform()
            except TimeoutException:
                pass

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
                no_of_reviews = int(drv.find_element_by_class_name(
                    'css-ils4e4').text.split()[0])
            except Exception as ex:
                log_exception(self.logger,
                              additional_information=f'Prod ID: {prod_id}')
                self.logger.info(str.encode(f'Product: {product_name} prod_id: {prod_id} reviews extraction failed.\
                                              Either product has no reviews or not\
                                              available for sell currently.(page: {product_page})', 'utf-8', 'ignore'))
                no_of_reviews = 0

            if no_of_reviews > 0:
                reviews = self.get_reviews(
                    drv, prod_id, product_name, product_page, last_scraped_review_date, no_of_reviews, incremental, reviews=[])
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
                reviews = []

            if prod != 0 and prod % 10 == 0:
                detail_data, item_df, review_data = store_data_refresh_mem(
                    detail_data, item_df, review_data)

            self.meta.loc[prod, 'scraped'] = 'Y'
            drv.quit()

        detail_data, item_df, review_data = store_data_refresh_mem(
            detail_data, item_df, review_data)
        self.logger.info(
            f'Extraction Complete for start_idx: (indices[0]) to end_idx: {indices[-1]}. Or for list of values.')

    def extract(self, metadata: Union[pd.DataFrame, str, Path], download: bool = True, n_workers: int = 5,
                fresh_start: bool = False, auto_fresh_start: bool = False, incremental: bool = True,
                open_headless: bool = False, open_with_proxy_server: bool = True, randomize_proxy_usage: bool = True,
                start_idx: Optional[int] = None, end_idx: Optional[int] = None, list_of_index=None,
                compile_progress_files: bool = False, clean: bool = True, delete_progress: bool = False):
        """Extract method controls all properties of the spiders and runs multi-threaded web crawling.

        Extract is exposes all functionality of the spiders and user needs to run this method to begin data crawling from web.
        This method has four major functionality:
        * 1. Run the spider
        * 2. Store data in regular intervals to free up ram
        * 3. Compile all crawled data into one file.
        * 4. Clean and push cleaned data to S3 storage for further algorithmic processing.

        Args:
            metadata (Union[pd.DataFrame, str, Path]): Dataframe containing product specific url, name and id of the products to be scraped.
            download (bool, optional): Whether to crawl data from or compile crawled data into one file. Defaults to True.
            n_workers (int, optional): No. of parallel threads to run. Defaults to 5.
            fresh_start (bool, optional): Whether to continue last crawl job or start new one. Defaults to False.
            auto_fresh_start (bool, optional): Whether to automatically start a new crawl job if last job was finished. Defaults to False.
            incremental (bool, optional): Whether to scrape reviews incrementally from last scraped review date. Defaults to True.
            open_headless (bool, optional):  Whether to open browser headless. Defaults to False.
            open_with_proxy_server (bool, optional): Whether to use ip rotation service. Defaults to True.
            randomize_proxy_usage (bool, optional): Whether to use both proxy and native network in tandem to decrease proxy requests.
            Defaults to False.
            start_idx (Optional[int], optional): Starting index of the links to crawl. Defaults to None.
            end_idx (Optional[int], optional): Ending index of the links to crawl. Defaults to None.
            list_of_index ([type], optional): List of indices or range of indices of product urls to scrape. Defaults to None.
            compile_progress_files (bool, optional): Whether to combine crawled data into one file. Defaults to False.
            clean (bool, optional): Whether to clean the compiled data. Defaults to True.
            delete_progress (bool, optional): Whether to delete intermediate data after compilation into one file. Defaults to False.

        """
        def fresh():
            if not isinstance(metadata, pd.core.frame.DataFrame):
                list_of_files = self.metadata_clean_path.glob(
                    'no_cat_cleaned_sph_product_metadata_all*')
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
                if Path(self.path/'sephora/sph_detail_review_image_progress_tracker.csv').exists():
                    self.meta = pd.read_csv(
                        self.path/'sephora/sph_detail_review_image_progress_tracker.csv')
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
                detail_df['meta_date'] = meta_date
                detail_filename = f'sph_product_detail_all_{meta_date}.csv'
                detail_df.to_csv(self.detail_path/detail_filename, index=None)
                # detail_df.to_feather(self.detail_path/detail_filename)

                item_li = []
                self.bad_item_li = []
                item_files = [f for f in self.detail_current_progress_path.glob(
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
                item_dataframe['meta_date'] = meta_date
                item_filename = f'sph_product_item_all_{meta_date}.csv'
                item_dataframe.to_csv(
                    self.detail_path/item_filename, index=None)
                # item_df.to_feather(self.detail_path/item_filename)

                self.logger.info(
                    f'Detail and Item files created. Please look for file sph_product_detail_all and\
                        sph_product_item_all in path {self.detail_path}')
                print(
                    f'Detail and Item files created. Please look for file sph_product_detail_all and\
                        sph_product_item_all in path {self.detail_path}')

                self.logger.info('Creating Combined Review File')

                rev_li = []
                self.bad_rev_li = []
                review_files = [f for f in self.review_current_progress_path.glob(
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
                rev_df['meta_date'] = pd.to_datetime(meta_date).date()
                review_filename = f'sph_product_review_all_{meta_date}'
                rev_df.to_feather(self.review_path/review_filename)

                self.logger.info(
                    f'Review file created. Please look for file sph_product_review_all in path {self.review_path}')
                print(
                    f'Review file created. Please look for file sph_product_review_all in path {self.review_path}')

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
        """terminate_logging."""
        self.logger.handlers.clear()
        self.prod_detail_review_image_log.stop_log()
