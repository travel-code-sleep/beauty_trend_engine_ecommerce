
"""Utils module contains crucial classes and functions that are used in all other modules of meiyume package."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import gc
import io
import logging
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import *

import boto3
import numpy as np
import pandas as pd
import pg8000
from retrying import retry
# import missingno as msno
from selenium import webdriver
from selenium.common.exceptions import (ElementClickInterceptedException,
                                        NoSuchElementException,
                                        StaleElementReferenceException,
                                        TimeoutException)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager

os.environ['WDM_LOG_LEVEL'] = '0'


class MeiyumeException(Exception):
    """MeiyumeException class to define custom exceptions in runtime.

    Args:
        Exception (object): Python exceptions module.

    """

    pass


class Browser(object):
    """Browser class serves selenium web-driver in head and headless mode for web scraping.

    Browser module provides methods to either use chrome or firefox browser for scraping.

    It carries out below functions:
    1. Instantiate selenium driver for scraping.
    2. Enable or disable ip rotation service
    3. Open web pages to take high resolution screenshots.
    4. Scroll web page.
    5. Scroll to particular element on a webpage.

    """

    def open_browser(self, open_headless: bool = False, open_for_screenshot: bool = False,
                     open_with_proxy_server: bool = False, path: Path = Path.cwd()) -> webdriver.Chrome:
        """open_browser instantiates selenium chrome driver with or without proxy services.

        Args:
            open_headless (bool, optional): Whether to open browser in headless mode. Defaults to False.
            open_for_screenshot (bool, optional): Whether to open browser to take screenshots. Defaults to False.
            open_with_proxy_server (bool, optional): Whether to enable ip rotation service. Defaults to False.
            path (Path, optional): Folder path where the driver software will be saved and used from.
                                   Defaults to current working directory(Path.cwd()).

        Returns:
            webdriver.Chrome: Instantiated chrome driver.

        """
        # chrome_options = Options()
        chrome_options = webdriver.ChromeOptions()
        chrome_options.set_capability('unhandledPromptBehavior', 'accept')
        chrome_options.set_capability('unexpectedAlertBehaviour', 'accept')
        # chrome_options.add_argument('--no-sandbox')

        if open_headless:
            chrome_options.add_argument('--headless')

        if open_for_screenshot:
            WINDOW_SIZE = "1920,1080"
            chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)

        if open_with_proxy_server:
            chrome_options.add_argument('--ignore-ssl-errors=yes')
            chrome_options.add_argument('--ignore-certificate-errors')
            headless_proxy = "127.0.0.1:3128"
            proxy = Proxy({
                'proxyType': ProxyType.MANUAL,
                'httpProxy': headless_proxy,
                'ftpProxy': headless_proxy,
                'sslProxy': headless_proxy,
                "noProxy": None,
                "proxyType": "MANUAL",
                "class": "org.openqa.selenium.Proxy",
                "autodetect": False,
                "acceptSslCerts": True,
                "unexpectedAlertBehaviour": "accept",
                "browser.tabs.warnOnClose": False
            })
            capabilities = dict(DesiredCapabilities.CHROME)
            proxy.add_to_capabilities(capabilities)
            driver = webdriver.Chrome(ChromeDriverManager(path=path, log_level=0).install(),
                                      desired_capabilities=capabilities, options=chrome_options)
            driver.set_page_load_timeout(600)
            return driver

        driver = webdriver.Chrome(ChromeDriverManager(path=path,
                                                      log_level=0).install(),
                                  options=chrome_options)
        driver.set_page_load_timeout(600)
        return driver

    def open_browser_firefox(self, open_headless: bool = False, open_for_screenshot: bool = False,
                             open_with_proxy_server: bool = False, path: Path = Path.cwd()) -> webdriver.Firefox:
        """open_browser instantiates selenium firefox geckodriver with or without proxy services.

        Args:
            open_headless (bool, optional): Whether to open browser in headless mode. Defaults to False.
            open_for_screenshot (bool, optional): Whether to open browser to take screenshots. Defaults to False.
            open_with_proxy_server (bool, optional): Whether to enable ip rotation service. Defaults to False.
            path (Path, optional): Folder path where the driver software will be saved and used from.
                                   Defaults to current working directory(Path.cwd()).

        Returns:
            webdriver.Firefox: Instantiated firefox driver.

        """
        # Add service path creation condition to store logs
        if not Path(path/'service').exists():
            (path/'service').mkdir(parents=True, exist_ok=True)

        binary = FirefoxBinary(
            r'C:\Program Files\Mozilla Firefox\firefox.exe')
        firefox_options = webdriver.FirefoxOptions()
        firefox_options.set_capability('unhandledPromptBehavior', 'accept')
        firefox_options.set_capability('unexpectedAlertBehaviour', 'accept')

        if open_headless:
            firefox_options.add_argument('--headless')

        if open_for_screenshot:
            WINDOW_SIZE = "1920,1080"
            firefox_options.add_argument("--window-size=%s" % WINDOW_SIZE)

        if open_with_proxy_server:
            firefox_options.add_argument('--ignore-ssl-errors=yes')
            firefox_options.add_argument('--ignore-certificate-errors')
            headless_proxy = "127.0.0.1:3128"
            proxy = Proxy({
                'proxyType': ProxyType.MANUAL,
                'httpProxy': headless_proxy,
                'ftpProxy': headless_proxy,
                'sslProxy': headless_proxy,
                "noProxy": None,
                "proxyType": "MANUAL",
                "class": "org.openqa.selenium.Proxy",
                "autodetect": False,
                "acceptSslCerts": True,
                "unexpectedAlertBehaviour": "accept",
                "browser.tabs.warnOnClose": False
            })
            capabilities = dict(DesiredCapabilities.FIREFOX)
            capabilities["marionette"] = True
            proxy.add_to_capabilities(capabilities)
            driver = webdriver.Firefox(executable_path=GeckoDriverManager(path=path, log_level=0).install(),
                                       desired_capabilities=capabilities, options=firefox_options,
                                       firefox_binary=binary, service_log_path=path/'service/geckodriver.log',
                                       log_path=path/'geckodriver.log')
            driver.set_page_load_timeout(600)
            return driver

        driver = webdriver.Firefox(executable_path=GeckoDriverManager(path=path, log_level=0).install(),
                                   options=firefox_options, firefox_binary=binary,
                                   service_log_path=path/'service/geckodriver.log', log_path=path/'geckodriver.log')
        driver.set_page_load_timeout(600)
        return driver

    @staticmethod
    def scroll_down_page(driver: Union[webdriver.Firefox, webdriver.Chrome], speed: int = 8,
                         h1: int = 0, h2: int = 1) -> None:
        """scroll_down_page scrolls up or down a page at given speed.

        Args:
            driver (Union[webdriver.Firefox, webdriver.Chrome]): The selenium driver with opened webpage.
            speed (int, optional): Scrolling speed. Defaults to 8.
            h1 (int, optional): Starting height from which to scroll. Defaults to 0.
            h2 (int, optional): Ending height to which to scroll. Defaults to 1.

        """
        current_scroll_position, new_height = h1, h2
        while current_scroll_position <= new_height:
            current_scroll_position += speed
            driver.execute_script(
                "window.scrollTo(0, {});".format(current_scroll_position))
            new_height = driver.execute_script(
                "return document.body.scrollHeight")

    @staticmethod
    def scroll_to_element(driver: Union[webdriver.Firefox, webdriver.Chrome], element: WebElement) -> None:
        """scroll_to_element scrolls to a particular element on a webpage.

        Args:
            driver (Union[webdriver.Firefox, webdriver.Chrome]): The selenium driver with opened webpage.
            element (WebElement): The web element to scroll to.

        """
        driver.execute_script(
            "arguments[0].scrollIntoView();", element)


class Sephora(Browser):
    """This object is inherited by all crawler and cleaner classes in sph.crawler module.

    Sephora class creates and sets directories for respective data definitions.

    Args:
        Browser (object): Browser class serves selenium webdriver in head or headless
                          mode. It also provides some additional utilities such as scrolling. proxies etc.

    """

    def __init__(self, data_def: str = None, path: Union[str, Path] = Path.cwd()):
        """__init__ Sephora class instacne constructor creates all the data folder paths as per data definition.

        The sub directories are created under parent directory only when the folders don't already exist.

        Args:
            data_def ([type], optional): Type of e-commerce data. Defaults to None.(Accepted values: [Metadata, Detail, Review])
            path (Union[str, Path], optional): The parent directory where the data definition specific sub-directories
                                               will be created. Defaults to Path.cwd().

        """
        super().__init__()
        self.path = Path(Path(path)/'sephora')

        # set data paths as per calls from data definition classes
        self.metadata_path = self.path/'metadata'
        self.old_metadata_files_path = self.metadata_path/'old_metadata_files'
        self.metadata_clean_path = self.metadata_path/'clean'
        self.old_metadata_clean_files_path = self.metadata_path/'cleaned_old_metadata_files'
        # set crawler trigger folders
        self.detail_crawler_trigger_path = self.path/'detail_crawler_trigger_folder'
        self.review_crawler_trigger_path = self.path/'review_crawler_trigger_folder'
        if data_def == 'meta':
            self.metadata_path.mkdir(parents=True, exist_ok=True)
            self.old_metadata_files_path.mkdir(parents=True, exist_ok=True)
            self.metadata_clean_path.mkdir(parents=True, exist_ok=True)
            self.old_metadata_clean_files_path.mkdir(
                parents=True, exist_ok=True)
            self.detail_crawler_trigger_path.mkdir(
                parents=True, exist_ok=True)
            self.review_crawler_trigger_path.mkdir(
                parents=True, exist_ok=True)

        self.detail_path = self.path/'detail'
        self.old_detail_files_path = self.detail_path/'old_detail_files'
        self.detail_clean_path = self.detail_path/'clean'
        self.old_detail_clean_files_path = self.detail_path/'cleaned_old_detail_files'
        if data_def == 'detail':
            self.detail_path.mkdir(parents=True, exist_ok=True)
            self.old_detail_files_path.mkdir(parents=True, exist_ok=True)
            self.detail_clean_path.mkdir(parents=True, exist_ok=True)
            self.old_detail_clean_files_path.mkdir(parents=True, exist_ok=True)

        self.review_path = self.path/'review'
        self.old_review_files_path = self.review_path/'old_review_files'
        self.review_clean_path = self.review_path/'clean'
        self.old_review_clean_files_path = self.review_path/'cleaned_old_review_files'
        if data_def == 'review':
            self.review_path.mkdir(parents=True, exist_ok=True)
            self.old_review_files_path.mkdir(parents=True, exist_ok=True)
            self.review_clean_path.mkdir(parents=True, exist_ok=True)
            self.old_review_clean_files_path.mkdir(parents=True, exist_ok=True)

        self.image_path = self.path/'product_images'
        self.image_processed_path = self.image_path/'processed_product_images'
        if data_def == 'image':
            self.image_path.mkdir(parents=True, exist_ok=True)
            self.image_processed_path.mkdir(parents=True, exist_ok=True)

        if data_def == 'detail_review_image':
            self.review_path.mkdir(parents=True, exist_ok=True)
            self.old_review_files_path.mkdir(parents=True, exist_ok=True)
            self.review_clean_path.mkdir(parents=True, exist_ok=True)
            self.old_review_clean_files_path.mkdir(parents=True, exist_ok=True)
            self.detail_path.mkdir(parents=True, exist_ok=True)
            self.old_detail_files_path.mkdir(parents=True, exist_ok=True)
            self.detail_clean_path.mkdir(parents=True, exist_ok=True)
            self.old_detail_clean_files_path.mkdir(parents=True, exist_ok=True)
            self.image_path.mkdir(parents=True, exist_ok=True)
            self.image_processed_path.mkdir(parents=True, exist_ok=True)
        # set universal log path for sephora
        self.crawl_log_path = self.path/'crawler_logs'
        self.crawl_log_path.mkdir(parents=True, exist_ok=True)
        self.clean_log_path = self.path/'cleaner_logs'
        self.clean_log_path.mkdir(parents=True, exist_ok=True)


class Boots(Browser):
    """This object is inherited by all crawler and cleaner classes in bts.crawler module.

    Boots class creates and sets directories for respective data definitions.

    Args:
        Browser (object): Browser class serves selenium webdriver in head or headless
                          mode. It also provides some additional utilities such as scrolling. proxies etc.

    """

    def __init__(self, data_def: str = None, path: Union[str, Path] = Path.cwd()):
        """__init__ Boots class instance constructor creates all the data folder paths as per data definition.

        The sub directories are created under parent directory only when the folders don't already exist.

        Args:
            data_def ([type], optional): Type of e-commerce data. Defaults to None.(Accepted values: [Metadata, Detail, Review])
            path (Union[str, Path], optional): The parent directory where the data definition specific sub-directories
                                               will be created. Defaults to Path.cwd().

        """
        super().__init__()
        self.path = Path(Path(path)/'boots')
        # set data paths as per calls from data definition classes
        self.metadata_path = self.path/'metadata'
        self.old_metadata_files_path = self.metadata_path/'old_metadata_files'
        self.metadata_clean_path = self.metadata_path/'clean'
        self.old_metadata_clean_files_path = self.metadata_path/'cleaned_old_metadata_files'
        # set crawler trigger folders
        self.detail_crawler_trigger_path = self.path/'detail_crawler_trigger_folder'
        self.review_crawler_trigger_path = self.path/'review_crawler_trigger_folder'
        if data_def == 'meta':
            self.metadata_path.mkdir(parents=True, exist_ok=True)
            self.old_metadata_files_path.mkdir(parents=True, exist_ok=True)
            self.metadata_clean_path.mkdir(parents=True, exist_ok=True)
            self.old_metadata_clean_files_path.mkdir(
                parents=True, exist_ok=True)
            self.detail_crawler_trigger_path.mkdir(
                parents=True, exist_ok=True)
            self.review_crawler_trigger_path.mkdir(
                parents=True, exist_ok=True)

        self.detail_path = self.path/'detail'
        self.old_detail_files_path = self.detail_path/'old_detail_files'
        self.detail_clean_path = self.detail_path/'clean'
        self.old_detail_clean_files_path = self.detail_path/'cleaned_old_detail_files'
        if data_def == 'detail':
            self.detail_path.mkdir(parents=True, exist_ok=True)
            self.old_detail_files_path.mkdir(parents=True, exist_ok=True)
            self.detail_clean_path.mkdir(parents=True, exist_ok=True)
            self.old_detail_clean_files_path.mkdir(parents=True, exist_ok=True)

        self.review_path = self.path/'review'
        self.old_review_files_path = self.review_path/'old_review_files'
        self.review_clean_path = self.review_path/'clean'
        self.old_review_clean_files_path = self.review_path/'cleaned_old_review_files'
        if data_def == 'review':
            self.review_path.mkdir(parents=True, exist_ok=True)
            self.old_review_files_path.mkdir(parents=True, exist_ok=True)
            self.review_clean_path.mkdir(parents=True, exist_ok=True)
            self.old_review_clean_files_path.mkdir(parents=True, exist_ok=True)

        self.image_path = self.path/'product_images'
        self.image_processed_path = self.image_path/'processed_product_images'
        if data_def == 'image':
            self.image_path.mkdir(parents=True, exist_ok=True)
            self.image_processed_path.mkdir(parents=True, exist_ok=True)

        if data_def == 'detail_review_image':
            self.review_path.mkdir(parents=True, exist_ok=True)
            self.old_review_files_path.mkdir(parents=True, exist_ok=True)
            self.review_clean_path.mkdir(parents=True, exist_ok=True)
            self.old_review_clean_files_path.mkdir(parents=True, exist_ok=True)
            self.detail_path.mkdir(parents=True, exist_ok=True)
            self.old_detail_files_path.mkdir(parents=True, exist_ok=True)
            self.detail_clean_path.mkdir(parents=True, exist_ok=True)
            self.old_detail_clean_files_path.mkdir(parents=True, exist_ok=True)
            self.image_path.mkdir(parents=True, exist_ok=True)
            self.image_processed_path.mkdir(parents=True, exist_ok=True)

        # set universal log path for sephora
        self.crawl_log_path = self.path/'crawler_logs'
        self.crawl_log_path.mkdir(parents=True, exist_ok=True)
        self.clean_log_path = self.path/'cleaner_logs'
        self.clean_log_path.mkdir(parents=True, exist_ok=True)


class ModelsAlgorithms(object):
    """ModelsAlgorithms creates folder structure to store outputs from several algorithms to disk."""

    def __init__(self, path: Union[str, Path] = Path.cwd()):
        """__init__ ModelsAlgorithms instance constructor will create the output paths at time of instantiation.

        Args:
            path Union[str, Path]: The parent directory where the output subdirectories will be created.
                                   Defaults to current working directory(Path.cwd()).

        """
        self.path = Path(path)
        self.output_path = self.path/'algorithm_outputs'
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.external_path = self.path/'external_data_sources'
        self.external_path.mkdir(parents=True, exist_ok=True)

        self.model_path = self.path/'dl_ml_models'
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.sph = Sephora(path='.')
        self.bts = Boots(path='.')


class Logger(object):
    """Logger creates file handlers to write program execution logs to disk."""

    def __init__(self, task_name: str, path: Path):
        """__init__ initializes the file write stream.

        Args:
            task_name (str): Name of the log file.
            path (Path): Path in which the generated logs will be stored.

        """
        self.filename = path / \
            f'{task_name}_{time.strftime("%Y-%m-%d-%H%M%S")}.log'

    def start_log(self):
        """Start writing logs."""
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
        """Stop writing logs and flush the file handlers."""
        # self.logger.removeHandler(self.file_handler)
        del self.logger, self.file_handler
        gc.collect()


'''
def show_missing_value(dataframe, viz_type=None):
    """[summary]

    Arguments:
        dataframe {[type]} -- [description]

    Keyword Arguments:
        viz_type {[type]} -- [description] (default: {None})
    """
    if viz_type == 'matrix':
        return msno.matrix(dataframe, figsize=(12, 4))
    elif viz_type == 'percentage':
        return dataframe.isna().mean() * 100
    elif viz_type == 'dendrogram':
        return msno.dendrogram(dataframe, figsize=(12, 8))
    else:
        return dataframe.isna().sum()
'''


def chunks(l, n):
    """Yield successive n-sized chunks from l.

    Arguments:
        l {[list, range, index]} -- [description]
        n {[type]} -- [description]

    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def convert_ago_to_date(x: str) -> str:
    """convert_ago_to_date removes ago from date and converts to proper date format.

    Args:
        x (str): Date data to clean. (e.g.: 11 days ago)

    Returns:
        str: Cleaned date data in dd mm yyyy format.

    """
    if 'ago' in x.lower() and x is not np.nan:
        if 'd' in x.lower():
            days = int(x.split()[0])
            date = datetime.today() - timedelta(days=days)
            return date.strftime('%d %b %Y')
        elif 'm' in x.lower():
            mins = int(x.split()[0])
            date = datetime.today()  # - timedelta(minutes=mins)
            return date.strftime('%d %b %Y')
        elif 'h' in x.lower():
            hours = int(x.split()[0])
            date = datetime.today()  # - timedelta(hours=hours)
            return date.strftime('%d %b %Y')
    else:
        return x


class S3FileManager(object):
    """S3FileManager reads from and writes data to aws S3 storage.

    S3FileManager has below major functions:
    1. Find stored files with string search.
    2. Upload files to S3.
    3. Download files from S3.
    4. Read files from S3 into pandas dataframes.
    5. Delete files in S3.
    6. Crete S3 folder path for data upload.

    """

    def __init__(self, bucket: str = 'meiyume-datawarehouse-prod'):
        """__init__ initializes S3FileManager instance with given data bucket.

        Args:
            bucket (str, optional): The S3 bucket from/to which files will be read/downloaded/uploaded.
                                    Defaults to 'meiyume-datawarehouse-prod'.

        """
        self.bucket = bucket

    def get_matching_s3_objects(self, prefix: str = "", suffix: str = ""):
        """get_matching_s3_objects searches S3 with string matching to find relevant keys.

        Args:
            prefix (str, optional): Only fetch objects whose key starts with this prefix. Defaults to "".
            suffix (str, optional): Only fetch objects whose keys end with this suffix. Defaults to "".

        Yields:
            Matching S3 keys.

        """
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")

        kwargs = {'Bucket': self.bucket}

        # We can pass the prefix directly to the S3 API.  If the user has passed
        # a tuple or list of prefixes, we go through them one by one.
        if isinstance(prefix, str):
            prefixes = (prefix, )
        else:
            prefixes = prefix

        for key_prefix in prefixes:
            kwargs["Prefix"] = key_prefix

            for page in paginator.paginate(**kwargs):
                try:
                    contents = page["Contents"]
                except KeyError:
                    break

                for obj in contents:
                    key = obj["Key"]
                    if key.endswith(suffix):
                        yield obj

    def get_matching_s3_keys(self, prefix: str = "", suffix: str = ""):
        """get_matching_s3_keys Generates the matching keys in an S3 bucket.

        Args:
            prefix (str, optional): Only fetch objects whose key starts with this prefix. Defaults to "".
            suffix (str, optional): Only fetch objects whose keys end with this suffix. Defaults to "".

        Yields:
            Any: Matching S3 object key

        """
        for obj in self.get_matching_s3_objects(prefix, suffix):
            yield obj  # obj["Key"]

    def get_last_modified_s3(self, key: str) -> dict:
        """get_last_modified_date_s3 gets the last modified date of a S3 object.

        Args:
            key (str): Object key to find last modified date for.

        Returns:
            dict: Dictionary containing the key and last modified timestamp.

        """
        s3 = boto3.resource('s3')
        k = s3.Bucket(self.bucket).Object(key)  # pylint: disable=no-member
        return {'key_name': k.key, 'key_last_modified': str(k.last_modified)}

    def get_prefix_s3(self, job_name: str) -> str:
        """get_prefix_s3 sets the correct S3 file prefix depending on the upload job.

        [extended_summary]

        Args:
            job_name (str): [description]

        Raises:
            MeiyumeException: [description]

        Returns:
            str: [description]

        """
        upload_jobs = {
            'source_meta': 'Feeds/BeautyTrendEngine/Source_Meta/Staging/',
            'meta_detail': 'Feeds/BeautyTrendEngine/Meta_Detail/Staging/',
            'item': 'Feeds/BeautyTrendEngine/Item/Staging/',
            'ingredient': 'Feeds/BeautyTrendEngine/Ingredient/Staging/',
            'review': 'Feeds/BeautyTrendEngine/Review/Staging/',
            'review_summary': 'Feeds/BeautyTrendEngine/Review_Summary/Staging/',
            'image': 'Feeds/BeautyTrendEngine/Image/Staging/',
            'cleaned_pre_algorithm': 'Feeds/BeautyTrendEngine/CleanedData/PreAlgorithm/',
            'webapp': 'Feeds/BeautyTrendEngine/WebAppData/',
            'webapp_test': 'Feeds/BeautyTrendEngine/WebAppDevelopmentData/Test/'
        }

        try:
            return upload_jobs[job_name]
        except Exception as ex:
            raise MeiyumeException(
                'Unrecognizable job. Please input correct job_name.')
        '''
        if job_name == 'source_meta':
            prefix = 'Feeds/BeautyTrendEngine/Source_Meta/Staging/'
        elif job_name == 'meta_detail':
            prefix = 'Feeds/BeautyTrendEngine/Meta_Detail/Staging/'
        elif job_name == 'item':
            prefix = 'Feeds/BeautyTrendEngine/Item/Staging/'
        elif job_name == 'ingredient':
            prefix = 'Feeds/BeautyTrendEngine/Ingredient/Staging/'
        elif job_name == 'review':
            prefix = 'Feeds/BeautyTrendEngine/Review/Staging/'
        elif job_name == 'review_summary':
            prefix = 'Feeds/BeautyTrendEngine/Review_Summary/Staging/'
        elif job_name == 'image':
            prefix = 'Feeds/BeautyTrendEngine/Image/Staging/'
        elif job_name == 'cleaned_pre_algorithm':
            prefix = 'Feeds/BeautyTrendEngine/CleanedData/PreAlgorithm/'
        elif job_name == 'webapp':
            prefix = 'Feeds/BeautyTrendEngine/WebAppData/'
        else:
            raise MeiyumeException(
                'Unrecognizable job. Please input correct job_name.')
        return prefix
        '''

    def push_file_s3(self, file_path: Union[str, Path], job_name: str) -> None:
        """push_file_s3 upload file to S3 storage with job name specific prefix.

        Args:
            file_path (Union[str, Path]): File path of the file to be uploaded as a string or Path object.
            job_name (str): Type of file to upload: One of [meta_detail, item, ingredient,
                                                        review, review_summary, image,
                                                        cleaned_pre_algorithm, webappdata]

        """
        # cls.make_manager()
        file_name = str(file_path).split("\\")[-1]

        prefix = self.get_prefix_s3(job_name)
        object_name = prefix+file_name
        # try:
        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file(str(file_path), self.bucket, object_name)
            print('file pushed successfully.')
        except Exception:
            print('file pushing task failed.')

    def pull_file_s3(self, key: str, file_path: Path = Path.cwd()) -> None:
        """pull_file_s3 dowload file from S3.

        Args:
            key (str): The file object to download.
            file_path (Path, optional): The path in which the downloaded file will be stored.
                                        Defaults to current working directory (Path.cwd()).

        """
        s3 = boto3.resource('s3')
        file_name = str(key).split('/')[-1]
        s3.Bucket(self.bucket).download_file(  # pylint: disable=no-member
            key, f'{file_path}/{file_name}')

    def read_data_to_dataframe_s3(self, key: str, file_type: str) -> pd.DataFrame:
        """read_data_to_dataframe_s3 reads S3 object into a pandas dataframe.

        Args:
            key (str): S3 object key.
            file_type (str): File format.

        Raises:
            MeiyumeException: Raises exception if incorrect key or file type provied. (Accepted types: csv, feather, pickle)

        Returns:
            pd.DataFrame: File data in pandas dataframe.

        """
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=self.bucket, Key=key)

        try:
            if file_type == 'csv':
                return pd.read_csv(io.BytesIO(obj['Body'].read()), sep='~')
            elif file_type == 'feather':
                return pd.read_feather(io.BytesIO(obj['Body'].read()))
            elif file_type == 'pickle':
                return pd.read_pickle(io.BytesIO(obj['Body'].read()))
        except Exception as ex:
            raise MeiyumeException('Provide correct file key and file type.')

    def read_feather_s3(self, key: str) -> pd.DataFrame:
        """read_feather_s3 will be removed in next version.

        [extended_summary]

        Args:
            key (str): [description]

        Returns:
            pd.DataFrame: [description]

        """
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=self.bucket, Key=key)
        df = pd.read_feather(io.BytesIO(obj['Body'].read()))
        return df

    def read_csv_s3(self, key: str) -> pd.DataFrame:
        """read_csv_s3 will be removed in next version.

        [extended_summary]

        Args:
            key (str): [description]

        Returns:
            pd.DataFrame: [description]

        """
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=self.bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()), sep='~')
        return df

    def delete_file_s3(self, key: str) -> None:
        """delete_file_s3 delete file object from S3.

        Args:
            key (str): The file key to delete.

        """
        s3 = boto3.resource('s3')
        try:
            s3.Object(self.bucket, key).delete()  # pylint: disable=no-member
            print('file deleted.')
        except Exception:
            print('delete operation failed')


class RedShiftReader(object):
    """RedShiftReader connects to Redshift database and performs table querying for trend engine schema."""

    def __init__(self):
        """__init__ initializes RedshiftReader instance with all the database connection properties."""
        self.host = 'lifungprod.cctlwakofj4t.ap-southeast-1.redshift.amazonaws.com'
        self.port = 5439
        self.database = 'lifungdb'
        self.user_name = 'btemymuser'
        self.password = 'Lifung123'
        self.conn = pg8000.connect(
            database=self.database, host=self.host, port=self.port,
            user=self.user_name, password=self.password)

    def query_database(self, query: str) -> pd.DataFrame:
        """query_database takes a sql query in text format and returns table/view query results as pandas dataframe.

        Args:
            query (str): Sql query as a string in double quotes.

        Returns:
            pd.DataFrame: Dataframe containing query results.

        """
        df = pd.read_sql_query(query, self.conn)
        df.columns = [name.decode('utf-8') for name in df.columns]
        return df


def log_exception(logger: Logger, additional_information: Optional[str] = None) -> None:
    """log_exception logs exception when occurred while executing code.

    Args:
        logger (Logger): The logger handler with access to log file.
        additional_information (Optional[str], optional): Any additional text info to add to the exception log.
                                                          Defaults to None.

    """
    exc_type, exc_obj, exc_tb = \
        sys.exc_info(
        )
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    if additional_information:
        logger.info(str.encode(
            f'Exception: {exc_type} occurred at line number {exc_tb.tb_lineno}.\
                (Filename: {file_name}). {additional_information}', 'utf-8', 'ignore'))
    else:
        logger.info(str.encode(
            f'Exception: {exc_type} occurred at line number {exc_tb.tb_lineno}.\
            (Filename: {file_name}).', 'utf-8', 'ignore'))


def close_popups(drv: Union[webdriver.Firefox, webdriver.Chrome]):
    """close_popups closes pop up banners/messages/windows on webpage during scraping.

    Args:
        drv (Union[webdriver.Firefox, webdriver.Chrome]): The selenium driver with opened webpage.

    """
    # close popup windows
    try:
        alert = drv.switch_to.alert
        alert.accept()
    except Exception:
        pass
    try:
        ActionChains(drv).send_keys(Keys.ESCAPE).perform()
        time.sleep(1)
        ActionChains(drv).send_keys(Keys.ESCAPE).perform()
    except Exception:
        pass


def accept_alert(drv: Union[webdriver.Firefox, webdriver.Chrome], wait_time: int) -> None:
    """accept_alert accepts unusual alerts on the webpage when scraping.

    Args:
        drv (Union[webdriver.Firefox, webdriver.Chrome]): The selenium driver with opened webpage.
        wait_time (int): Time to wait for the alert to appear.

    """
    try:
        WebDriverWait(drv, wait_time).until(EC.alert_is_present(),
                                            'Timed out waiting for PA creation ' +
                                            'confirmation popup to appear.')
        alert = drv.switch_to.alert
        alert.accept()
        print("alert accepted")
    except TimeoutException:
        pass


def ranges(N: int, nb: int, start_idx: int = 0) -> list:
    """Ranges partions a sequence of integers into equally spaced ranges.

    Args:
        N (int): end index of the range or length
        nb (int): no. of equally spaced ranges to return
        start_idx (int, optional): Start index of the range list. Defaults to 0.
                                   If start index is given the range partions will start from start_idx instead of 0.

    Returns:
        list: list of equispaced ranges between [(start_idx, N)]

    """
    step = (N-start_idx) / nb
    return [range(start_idx+round(step*i), start_idx+round(step*(i+1))) for i in range(nb)]


def hasNumbers(inputString: str) -> bool:
    """Hasnumbers checks whether string contains any numerical characters.

    Args:
        inputString (str): Input String

    Returns:
        bool: True if contains numbers.

    """
    return bool(re.search(r'\d', inputString))


class DataAggregator(object):
    """DataAggregator future class to handle data merging from multiple sites.

    [extended_summary]

    Args:
        object ([type]): [description]

    """

    # def __init__(self):
    #     self.sph = Sephora(path='.')
    #     pass
