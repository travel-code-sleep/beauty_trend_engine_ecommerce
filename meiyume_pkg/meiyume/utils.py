""" [summary]

[extended_summary]

Returns:
    [type]: [description]
"""
from datetime import datetime, timedelta, date
import logging
import time
import numpy as np
import os
import missingno as msno
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from webdriver_manager.chrome import ChromeDriverManager
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
                'noProxy': ''
            })
            capabilities = dict(DesiredCapabilities.CHROME)
            proxy.add_to_capabilities(capabilities)
            driver = webdriver.Chrome(ChromeDriverManager(path=path, log_level=0).install(),
                                      desired_capabilities=capabilities, options=chrome_options)
            return driver

        return webdriver.Chrome(ChromeDriverManager(path=path,
                                                    log_level=0).install(),
                                options=chrome_options)
    '''
    def open_browser_to_take_screenshot(self):
        """open_browser_to_take_screenshot [summary]

        [extended_summary]

        Returns:
            [type]: [description]
        """
        WINDOW_SIZE = "1920,1080"

        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
        return webdriver.Chrome(executable_path=self.driver_path, options=chrome_options)
    '''
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
        self.path = Path(path)
        # set data paths as per calls from data definition classes
        self.metadata_path = self.path/'sephora/metadata'
        self.metadata_clean_path = self.metadata_path/'clean'
        if data_def == 'meta':
            self.metadata_path.mkdir(parents=True, exist_ok=True)
            self.metadata_clean_path.mkdir(parents=True, exist_ok=True)
        self.detail_path = self.path/'sephora/detail'
        self.detail_clean_path = self.detail_path/'clean'
        if data_def == 'detail':
            self.detail_path.mkdir(parents=True, exist_ok=True)
            self.detail_clean_path.mkdir(parents=True, exist_ok=True)
        self.review_path = self.path/'sephora/review'
        self.review_clean_path = self.review_path/'clean'
        if data_def == 'review':
            self.review_path.mkdir(parents=True, exist_ok=True)
            self.review_clean_path.mkdir(parents=True, exist_ok=True)
        self.image_path = self.path/'sephora/product_images'
        self.image_processed_path = self.image_path/'processed_product_images'
        if data_def == 'image':
            self.image_path.mkdir(parents=True, exist_ok=True)
            self.image_processed_path.mkdir(parents=True, exist_ok=True)
        # set universal log path for sephora
        self.crawl_log_path = self.path/'sephora/crawler_logs'
        self.crawl_log_path.mkdir(parents=True, exist_ok=True)
        self.clean_log_path = self.path/'sephora/cleaner_logs'
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
        self.path = Path(path)
        # set data paths as per calls from data definition classes
        self.metadata_path = self.path/'boots/metadata'
        self.metadata_clean_path = self.metadata_path/'clean'
        if data_def == 'meta':
            self.metadata_path.mkdir(parents=True, exist_ok=True)
            self.metadata_clean_path.mkdir(parents=True, exist_ok=True)
        self.detail_path = self.path/'boots/detail'
        self.detail_clean_path = self.detail_path/'clean'
        if data_def == 'detail':
            self.detail_path.mkdir(parents=True, exist_ok=True)
            self.detail_clean_path.mkdir(parents=True, exist_ok=True)
        self.review_path = self.path/'boots/review'
        self.review_clean_path = self.review_path/'clean'
        if data_def == 'review':
            self.review_path.mkdir(parents=True, exist_ok=True)
            self.review_clean_path.mkdir(parents=True, exist_ok=True)
        self.image_path = self.path/'boots/product_images'
        self.image_processed_path = self.image_path/'processed_product_images'
        if data_def == 'image':
            self.image_path.mkdir(parents=True, exist_ok=True)
            self.image_processed_path.mkdir(parents=True, exist_ok=True)
        # set universal log path for sephora
        self.crawl_log_path = self.path/'boots/crawler_logs'
        self.crawl_log_path.mkdir(parents=True, exist_ok=True)
        self.clean_log_path = self.path/'boots/cleaner_logs'
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

    def __init__(self, bucket='meiyume-datawarehouse-prod'):
        """__init__ [summary]

        [extended_summary]

        Args:
            bucket (str, optional): [description]. Defaults to 'meiyume-datawarehouse-prod'.
        """
        self.bucket = bucket

    def get_matching_s3_objects(self, prefix="", suffix=""):
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

    def get_matching_s3_keys(self, prefix="", suffix=""):
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
        k = s3.Bucket(self.bucket).Object(key)
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
        if job_name == 'meta_detail':
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
        s3.Bucket(self.bucket).download_file(
            key, f'{file_path}/{file_name}')

    def delete_file_s3(self, key):
        """delete_file_s3 [summary]

        [extended_summary]

        Args:
            key ([type]): [description]
        """
        s3 = boto3.resource('s3')
        try:
            s3.Object(self.bucket, key).delete()
            print('file deleted.')
        except Exception:
            print('delete operation failed')


class DataAggregator(object):
    """DataAggregator [summary]

    [extended_summary]

    Args:
        object ([type]): [description]
    """

    def __init__(self):
        self.sph = Sephora(path='.')
        pass
