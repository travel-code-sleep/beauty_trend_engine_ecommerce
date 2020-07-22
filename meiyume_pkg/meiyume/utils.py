
"""[summary]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from typing import *
from datetime import datetime, timedelta, date
import sys
import logging
import time
import numpy as np
import os
# import missingno as msno
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.alert import Alert
from selenium.common.exceptions import (ElementClickInterceptedException,
                                        NoSuchElementException,
                                        StaleElementReferenceException,
                                        TimeoutException)
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from retrying import retry
import gc
from pathlib import Path
import boto3
os.environ['WDM_LOG_LEVEL'] = '0'


class MeiyumeException(Exception):
    """class to define custom exceptions in runtime

    Arguments:
        Exception {[type]} -- [description]
    """
    pass


class Browser(object):
    """Browser class serves selenium web-driver in head and headless
       mode. It also provides some additional utilities such as scrolling etc.

    Arguments:
        object {[type]} -- [description]
    """

    # def __init__(self):
    #     pass

    def open_browser(self, open_headless: bool = False, open_for_screenshot: bool = False,
                     open_with_proxy_server: bool = False, path: Path = Path.cwd())-> webdriver.Chrome:
        """open_browser [summary]

        [extended_summary]

        Args:
            open_headless (bool, optional): [description]. Defaults to False.
            open_for_screenshot (bool, optional): True enables image high resolution. If used to take screenshot open_headless must be set to True.
                                                  Defaults to False.
            open_with_proxy_server (bool, optional): [description]. Defaults to False.

        Returns:
            webdriver: [description]
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
                             open_with_proxy_server: bool = False, path: Path = Path.cwd())-> webdriver.Firefox:
        """open_browser_firefox [summary]

        [extended_summary]

        Args:
            open_headless (bool, optional): [description]. Defaults to False.
            open_for_screenshot (bool, optional): True enables image high resolution. If used to take screenshot open_headless must be set to True.
                                                  Defaults to False.
            open_with_proxy_server (bool, optional): [description]. Defaults to False.

        Returns:
            webdriver: [description]
        """
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
                                       desired_capabilities=capabilities, options=firefox_options, firefox_binary=binary)
            driver.set_page_load_timeout(600)
            return driver

        driver = webdriver.Firefox(executable_path=GeckoDriverManager(path=path, log_level=0).install(),
                                   options=firefox_options, firefox_binary=binary)
        driver.set_page_load_timeout(600)
        return driver

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
        current_scroll_position, new_height = h1, h2
        while current_scroll_position <= new_height:
            current_scroll_position += speed
            driver.execute_script(
                "window.scrollTo(0, {});".format(current_scroll_position))
            new_height = driver.execute_script(
                "return document.body.scrollHeight")

    @staticmethod
    def scroll_to_element(driver: webdriver.Firefox, element: WebElement)-> None:
        """scroll_to_element [summary]

        [extended_summary]

        Args:
            driver (webdriver.Firefox): [description]
            element (WebElement): [description]
        """
        driver.execute_script(
            "arguments[0].scrollIntoView();", element)


class Sephora(Browser):
    """ This object is inherited by all crawler and cleaner classes in sph_crawler
        and sph_cleaner modules.

        Sephora class creates and sets directories for respective data definitions.

    Arguments:
        Browser {[type]} -- [Browser class serves selenium webdriver in head and headless
                             mode. It also provides some additional utilities such as scrolling etc.]
    """

    def __init__(self, data_def=None, path=Path.cwd()):
        """__init__ [summary]

        [extended_summary]

        Args:
            data_def ([type], optional): [description]. Defaults to None.
            driver_path ([type], optional): [description]. Defaults to None.
            path ([type], optional): [description]. Defaults to Path.cwd().
            show (bool, optional): [description]. Defaults to True.
        """
        super().__init__()
        self.path = Path(Path(path)/'sephora')

        # set data paths as per calls from data definition classes
        self.metadata_path = self.path/'metadata'
        self.old_metadata_files_path = self.metadata_path/'old_metadata_files'
        self.metadata_clean_path = self.metadata_path/'clean'
        self.old_metadata_clean_files_path = self.metadata_path/'cleaned_old_metadata_files'
        if data_def == 'meta':
            self.metadata_path.mkdir(parents=True, exist_ok=True)
            self.old_metadata_files_path.mkdir(parents=True, exist_ok=True)
            self.metadata_clean_path.mkdir(parents=True, exist_ok=True)
            self.old_metadata_clean_files_path.mkdir(
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

        # set universal log path for sephora
        self.crawl_log_path = self.path/'crawler_logs'
        self.crawl_log_path.mkdir(parents=True, exist_ok=True)
        self.clean_log_path = self.path/'cleaner_logs'
        self.clean_log_path.mkdir(parents=True, exist_ok=True)


class Boots(Browser):
    """ This object is inherited by all crawler classes in sph.crawler module.

        Boots class creates and sets directories for respective data definitions.

    Arguments:
        Browser (class) -- Browser class serves selenium web-drvier in head and headless
                             mode. It also provides some additional utilities such as scrolling etc.
    """

    def __init__(self, data_def=None, path=Path.cwd()):
        """__init__ [summary]

        [extended_summary]

        Args:
            data_def (str, optional): [description]. Defaults to None.
            path (path:str, optional): [description]. Defaults to Path.cwd().
            driver_path (path:str, optional): [description]. Defaults to None.
            show (bool, optional): [description]. Defaults to True.
        """
        super().__init__()
        self.path = Path(Path(path)/'boots')
        # set data paths as per calls from data definition classes
        self.metadata_path = self.path/'metadata'
        self.old_metadata_files_path = self.metadata_path/'old_metadata_files'
        self.metadata_clean_path = self.metadata_path/'clean'
        self.old_metadata_clean_files_path = self.metadata_path/'cleaned_old_metadata_files'
        if data_def == 'meta':
            self.metadata_path.mkdir(parents=True, exist_ok=True)
            self.old_metadata_files_path.mkdir(parents=True, exist_ok=True)
            self.metadata_clean_path.mkdir(parents=True, exist_ok=True)
            self.old_metadata_clean_files_path.mkdir(
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

        # set universal log path for sephora
        self.crawl_log_path = self.path/'crawler_logs'
        self.crawl_log_path.mkdir(parents=True, exist_ok=True)
        self.clean_log_path = self.path/'cleaner_logs'
        self.clean_log_path.mkdir(parents=True, exist_ok=True)


class ModelsAlgorithms(object):
    """ModelsAlgorithms [summary]

    [extended_summary]

    Args:
        object ([type]): [description]
    """

    def __init__(self, path='.'):
        """__init__ [summary]

        [extended_summary]

        Args:
            path (str, optional): [description]. Defaults to '.'.
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
    """[summary]

    Arguments:
        object {[type]} -- [description]

    Returns:
        [type] -- [description]
    """""" pass """

    def __init__(self, task_name, path):
        self.filename = path / \
            f'{task_name}_{time.strftime("%Y-%m-%d-%H%M%S")}.log'

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
        # self.logger.removeHandler(self.file_handler)
        del self.logger, self.file_handler
        gc.collect()


def nan_equal(a, b):
    """[summary]

    Arguments:
        a {[type]} -- [description]
        b {[type]} -- [description]
    """
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


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


def convert_ago_to_date(x):
    """convert_ago_to_date [summary]

    [extended_summary]

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
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
    """S3FileManager [summary]

    [extended_summary]

    Args:
        object ([type]): [description]
    """

    def __init__(self, bucket: str = 'meiyume-datawarehouse-prod'):
        """__init__ [summary]

        [extended_summary]

        Args:
            bucket (str, optional): [description]. Defaults to 'meiyume-datawarehouse-prod'.
        """
        self.bucket = bucket

    def get_matching_s3_objects(self, prefix: str = "", suffix: str = ""):
        """
        Generate objects in an S3 bucket.

        :param bucket: Name of the S3 bucket.
        :param prefix: Only fetch objects whose key starts with
            this prefix (optional).
        :param suffix: Only fetch objects whose keys end with
            this suffix (optional).
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
        """
        Generate the keys in an S3 bucket.

        :param bucket: Name of the S3 bucket.
        :param prefix: Only fetch keys that start with this prefix (optional).
        :param suffix: Only fetch keys that end with this suffix (optional).
        """
        for obj in self.get_matching_s3_objects(prefix, suffix):
            yield obj  # obj["Key"]

    def get_last_modified_s3(self, key):
        """get_last_modified_date_s3 [summary]

        [extended_summary]

        Args:
            key ([type]): [description]
        """
        s3 = boto3.resource('s3')
        k = s3.Bucket(self.bucket).Object(key)  # pylint: disable=no-member
        return {'key_name': k.key, 'key_last_modified': str(k.last_modified)}

    def get_prefix_s3(self, job_name):
        """get_prefix_s3 [summary]

        [extended_summary]

        Args:
            job_name ([type]): [description]

        Raises:
            MeiyumeException: [description]

        Returns:
            [type]: [description]
        """
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
        else:
            raise MeiyumeException(
                'Unrecognizable job. Please input correct job_name.')
        return prefix

    def push_file_s3(self, file_path, job_name):
        """[summary]

        Arguments:
            file_path {[path:str]} -- [File name to store in S3]
            job_name {[str]} -- [Type of job: One of [meta_detail | item | ingredient | review | review_summary]]

        Raises:
            MeiyumeException: [if job name is not in above defined list]
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

    def pull_file_s3(self, key, file_path='.', suffix=None):
        """pull_file_s3 [summary]

        [extended_summary]

        Args:
            job_name ([type]): [description]
            file_path ([type], optional): [description]. Defaults to None.
        """
        # prefix = self.get_prefix_s3(job_name)
        # keys = self.get_matching_s3_keys(prefix)

        s3 = boto3.resource('s3')
        file_name = str(key).split('/')[-1]
        s3.Bucket(self.bucket).download_file(  # pylint: disable=no-member
            key, f'{file_path}/{file_name}')

    def delete_file_s3(self, key: str):
        """delete_file_s3 [summary]

        [extended_summary]

        Args:
            key ([type]): [description]
        """
        s3 = boto3.resource('s3')
        try:
            s3.Object(self.bucket, key).delete()  # pylint: disable=no-member
            print('file deleted.')
        except Exception:
            print('delete operation failed')


def log_exception(logger: Logger, additional_information: Optional[str] = None)->None:
    """log_exception [summary]

    [extended_summary]

    Args:
        logger (Logger): [description]
        additional_information (Optional[str], optional): [description]. Defaults to None.
    """
    exc_type, exc_obj, exc_tb = \
        sys.exc_info(
        )  # type:  Tuple[Optional[Type[BaseException]], Optional[BaseException], Optional[types.TracebackType]]
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    if additional_information:
        logger.info(str.encode(
            f'Exception: {exc_type} occurred at line number {exc_tb.tb_lineno}.\
                (Filename: {file_name}). {additional_information}', 'utf-8', 'ignore'))
    else:
        logger.info(str.encode(
            f'Exception: {exc_type} occurred at line number {exc_tb.tb_lineno}.\
            (Filename: {file_name}).', 'utf-8', 'ignore'))


def close_popups(drv: webdriver.Chrome):
    """close_popups [summary]

    [extended_summary]

    Args:
        drv (webdriver.Chrome): [description]
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


def accept_alert(drv: webdriver.Chrome, wait_time: int):
    """accept_alert [summary]

    [extended_summary]

    Args:
        drv (webdriver.Chrome): [description]
        wait_time (int): [description]
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


def ranges(N: int, nb: int, start_idx: int = 0)->list:
    """ranges [summary]

    [extended_summary]

    Args:
        N (int): end index of the range or length
        nb (int): no. of equally spaced ranges to return
        start_idx (int, optional): start index of the range list. Defaults to 0.

    Returns:
        list: list of equispaced ranges between [(start_idx, N)]
    """
    step = (N-start_idx) / nb
    return [range(start_idx+round(step*i), start_idx+round(step*(i+1))) for i in range(nb)]


class DataAggregator(object):
    """DataAggregator [summary]

    [extended_summary]

    Args:
        object ([type]): [description]
    """

    def __init__(self):
        self.sph = Sephora(path='.')
        pass
