from datetime import datetime, timedelta, date
import logging
import time
import numpy as np
import os
import missingno as msno
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import gc
from pathlib import Path
import boto3


class MeiyumeException(Exception):
    """class to define custom exceptions in runtime

    Arguments:
        Exception {[type]} -- [description]
    """
    pass


class Browser(object):
    """Browser class serves selenium web-drvier in head and headless
       mode. It also provides some additional utilities such as scrolling etc.

    Arguments:
        object {[type]} -- [description]
    """

    def __init__(self, driver_path, show):
        """ pass """
        self.show = show
        self.driver_path = driver_path

    def open_headless(self, show=False):
        self.show = show

    def open_browser(self):
        """[summary]
        """
        if self.show:
            return webdriver.Chrome(executable_path=self.driver_path)
        else:
            chrome_options = Options()
            # chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--headless')
            return webdriver.Chrome(executable_path=self.driver_path, options=chrome_options)

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
    """ This object is inherited by all crawler and cleaner classes in sph_crwaler
        and sph_cleaner modules.

        Sephora class creates and sets directories for respective data definitions.

    Arguments:
        Browser {[type]} -- [Browser class serves selenium web-drvier in head and headless
                             mode. It also provides some additional utilities such as scrolling etc.]
    """

    def __init__(self, data_def=None, driver_path=None, path=Path.cwd(), show=True):
        super().__init__(driver_path=driver_path, show=show)
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
        # set universal log path for sephora
        self.crawl_log_path = self.path/'sephora/crawler_logs'
        self.crawl_log_path.mkdir(parents=True, exist_ok=True)
        self.clean_log_path = self.path/'sephora/cleaner_logs'
        self.clean_log_path.mkdir(parents=True, exist_ok=True)


class StatAlgorithm(object):
    def __init__(self, path='.'):
        self.path = Path(path)
        self.output_path = self.path/'algorithm_outputs'
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.external_path = self.path/'external_data_sources'
        self.sph = Sephora(path='.')


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
    def __init__(self, bucket='meiyume-datawarehouse-prod'):
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

    # @classmethod
    # def make_manager(cls):
    #     return cls()

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
        except:
            print('file pushing task failed.')

    def pull_file_s3(self, job_name, file_path=None):
        """pull_file_s3 [summary]

        [extended_summary]

        Args:
            job_name ([type]): [description]
            file_path ([type], optional): [description]. Defaults to None.
        """
        prefix = get_prefix_s3(job_name)
        keys = self.get_matching_s3_keys(prefix)

        s3 = boto3.resource('s3')
        for key in keys:
            file_name = str(key).split('/')[-1]
            s3.Bucket(self.bucket).download_file(key, str(file_path/file_name))

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
        except:
            print('delete operation failed')


class DataAggregator(object):

    def __init__(self):
        self.sph = Sephora(path='.')
        pass
