""" [summary]

[extended_summary]

Returns:
    [type]: [description]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from numba import (jit, njit, vectorize, cuda)
import concurrent.futures
from tqdm.notebook import tqdm
from tqdm import tqdm
import swifter
import pandas as pd
import numpy as np
from typing import Union
from functools import reduce
from datetime import datetime, timedelta
from ast import literal_eval
from pathlib import Path
import warnings
import time
import os
import gc
from typing import *
from meiyume.utils import (Logger, Sephora, Boots, nan_equal,
                           show_missing_value,
                           MeiyumeException, S3FileManager)
# text lib imports
import re
import string
import unidecode
import spacy

# nlp = spacy.load('en_core_web_lg')
file_manager = S3FileManager()
tqdm.pandas()
warnings.simplefilter(action='ignore')
np.random.seed(1337)


class Cleaner():
    """Cleaner.

    [extended_summary]
    """

    def __init__(self, path='.'):
        """__init__ [summary]

        [extended_summary]

        Args:
            path (str, optional): [description]. Defaults to '.'.
        """
        self.sph = Sephora()
        self.bts = Boots()
        self.path = Path(path)

    def clean(self, data: Union[str, Path, pd.DataFrame], save: bool = True,
              logs: bool = False, include_category: bool = True) -> pd.DataFrame:
        """clean [summary]

        [extended_summary]

        Args:
            data (Union[str, Path, pd.DataFrame]): [description]
            save (bool, optional): [description]. Defaults to True.
            logs (bool, optional): [description]. Defaults to False.
            include_category (bool, optional): [description]. Defaults to True.

        Raises:
            MeiyumeException: [description]

        Returns:
            pd.DataFrame: [description]
        """

        if isinstance(data, pd.core.frame.DataFrame):
            filename = str(data)
            try:
                data = pd.read_feather(Path(data))
            except Exception:
                data = pd.read_csv(Path(data))
        else:
            filename = ''

        if filename != '':
            if save:
                save = True
            else:
                save = False
            words = filename.split('_')

            self.source, self.definition = words[0], words[2]

            self.clean_file_name = 'cleaned_' + str(filename).split('\\')[-1]

        if self.source not in ['bts', 'sph'] or self.definition not in ['metadata', 'detail',
                                                                        'item', 'review']:
            raise MeiyumeException(
                "Unable to determine data definition. Please provide correct file name.")

        cleaner_utility = self.get_cleaner_utility()

        cleaned_data = cleaner_utility(data, save)

    def get_cleaner_utility(self) -> Callable[[pd.DataFrame, bool], pd.DataFrame]:
        """get_cleaner_utility [summary]

        [extended_summary]

        Raises:
            MeiyumeException: [description]

        Returns:
            Callable[[pd.DataFrame, bool], pd.DataFrame]: [description]
        """
        if self.definition == 'metadata':
            return self.metadata_cleaner
        elif self.definition == 'detail':
            return self.detail_cleaner
        elif self.definition == 'item':
            return self.item_cleaner
        elif self.definition == 'review':
            return self.review_cleaner
        else:
            raise MeiyumeException(
                "Invalid data definition. Please provide correct file")

    @staticmethod
    def make_price(price: str) -> Tuple[str, str, str]:
        """make_price [summary]

        [extended_summary]

        Args:
            price (str): [description]

        Returns:
            Tuple[str, str, str]: [description]
        """
        if '/' not in price and '-' not in price:
            return price, 'no_value', 'no_value'

        elif '/' in price and '-' in price:
            p = re.split('-|/', price)
            return p[0], p[1], p[2]

        elif '/' in price and '-' not in price:
            p = re.split('/', price)
            return p[0], 'no_value', p[1]

        elif price.count('-') > 1 and '/' not in price:
            ts = [m.start() for m in re.finditer(' ', price)]
            p = price[ts[2]:].strip().split('-')
            return p[0], p[1], price[:ts[2]]

        elif '-' in price and price.count('-') < 2 and '/' not in price:
            p = re.split('-', price)
            return p[0], p[1], 'no_value'

        else:
            return 'no_value', 'no_value', 'no_value'

    @staticmethod
    def clean_price(price: str)-> str:
        """clean_price [summary]

        [extended_summary]

        Args:
            price (str): [description]

        Returns:
            str: [description]
        """
        replace_strings = ('$', ''), ('(', '/ '), \
                          (')', ''), ('value', ''), ('£', '')
        return reduce(lambda a, kv: a.replace(*kv), replace_strings, price)

    def metadata_cleaner(self, data: pd.DataFrame, save: bool)->pd.DataFrame:
        """metadata_cleaner [summary]

        [extended_summary]

        Args:
            data (pd.DataFrame): [description]
            save (bool): [description]

        Returns:
            pd.DataFrame: [description]
        """
        self.meta = data
        del data
        gc.collect()

        self.meta.product_name = self.meta.product_name.apply(
            unidecode.unidecode)
        self.meta.brand = self.meta.brand.apply(unidecode.unidecode)

        def fix_multi_low_price(x):
            """[summary]

            Arguments:
                x {[type]} -- [description]
            """
            if len(x) > 7 and ' ' in x:
                p = x.split()
                return p[-1], p[0]
            else:
                return 'no_value', 'no_value'

        # clean price
        if self.source == 'sph':
            self.meta['low_p'], self.meta['high_p'], self.meta['mrp'] = zip(
                *self.meta.price.apply(lambda x:
                                       Cleaner.clean_price(x)).apply(lambda y:
                                                                     Cleaner.make_price(y)))
            self.meta.drop('price', axis=1, inplace=True)

            self.meta.low_p[self.meta.low_p.apply(len) > 7],
            self.meta.mrp[self.meta.low_p.apply(len) > 7] =\
                zip(*self.meta.low_p[self.meta.low_p.apply(len)
                                     > 7].apply(fix_multi_low_price))
        else:
            self.meta.mrp = self.meta.mrp.apply(Cleaner.clean_price)
            self.meta.low_p = self.meta.low_p.apply(Cleaner.clean_price)
            self.meta.high_p = self.meta.high_p.apply(Cleaner.clean_price)

        # create product id
        self.meta['prod_id'] = self.meta.product_page.apply(
            lambda x: 'sph_'+x.split(':')[-1] if self.source == 'sph'
            else 'bts_'+x.split('-')[-1])
        '''
        if self.source == 'sph':
            self.meta['prod_id'] = self.meta.product_page.apply(
                lambda x: 'sph_'+x.split(':')[-1])
        if self.source == 'bts':
            self.meta['prod_id'] = self.meta.product_page.apply(
                lambda x: 'bts_'+x.split('-')[-1])
        '''
        # clean rating
        remove_chars = re.compile('stars|star|No')
        self.meta.rating = self.meta.rating.apply(
            lambda x: remove_chars.sub('', x))
        self.meta.rating[self.meta.rating == ' '] = '0'
        self.meta.rating = self.meta.rating.astype(float)

        # clean ingredient flag
        if self.source = 'sph': clean_prod_type = self.meta.product_type[self.meta.product_type.apply(
                lambda x: True if x.split('-')[0] == 'clean' else False)].unique()
            self.meta['clean_flag'] = self.meta.apply(
                lambda x: 'Yes' if x.product_type in clean_prod_type else 'Undefined', axis=1)

    def detail_cleaner(self, data: pd.DataFrame, save: bool)->pd.DataFrame:
        pass

    def item_cleaner(self, data: pd.DataFrame, save: bool)->pd.DataFrame:
        pass

    def review_cleaner(self, data: pd.DataFrame, save: bool)->pd.DataFrame:
        pass
