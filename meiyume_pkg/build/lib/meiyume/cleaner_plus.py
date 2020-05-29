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
# import swifter
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

        if not isinstance(data, pd.core.frame.DataFrame):
            filename = str(data).split('\\')[-1]
            try:
                data = pd.read_feather(Path(data))
            except Exception:
                data = pd.read_csv(Path(data))
        else:
            filename = ''

        if filename == '':
            save = False
        else:
            words = filename.split('_')

            self.source, self.definition = words[0], words[2]

            self.clean_file_name = 'cleaned_' + str(filename).split('\\')[-1]

        if self.source not in ['bts', 'sph'] or self.definition not in ['metadata', 'detail',
                                                                        'item', 'review']:
            raise MeiyumeException(
                "Unable to determine data definition. Please provide correct file name.")

        cleaner_utility = self.get_cleaner_utility()

        cleaned_data = cleaner_utility(data, save)
        return cleaned_data

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
            price (str): [description]`

        Returns:
            str: [description]
        """
        replace_strings = (('$', ''), ('(', '/ '),
                           (')', ''), ('value', ''),
                           ('£', ''), ('nan', ''))

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
        if self.source == 'sph':
            self.meta.brand = self.meta.brand.apply(unidecode.unidecode)

        self.meta.source = self.meta.source.str.lower()

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

            self.meta.low_p[self.meta.low_p.apply(len) > 7], self.meta.mrp[self.meta.low_p.apply(len) > 7] =\
                zip(*self.meta.low_p[self.meta.low_p.apply(len)
                                     > 7].apply(fix_multi_low_price))
        else:
            for i in ['mrp', 'high_p', 'low_p']:
                self.meta[i] = self.meta[i].map(Cleaner.clean_price)

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
        remove_chars = re.compile('stars|star|No|nan')
        self.meta.rating = self.meta.rating.apply(
            lambda x: remove_chars.sub('', x))
        self.meta.rating[self.meta.rating.isin([' ', ''])] = '0'
        self.meta.rating = self.meta.rating.astype(float)

        # to datetime
        self.meta.meta_date = pd.to_datetime(
            self.meta.meta_date, infer_datetime_format=True)

        # clean ingredient flag
        if self.source == 'sph':
            clean_prod_type = self.meta.product_type[self.meta.product_type.apply(
                lambda x: True if x.split('-')[0] == 'clean' else False)].unique()
            self.meta['clean_flag'] = self.meta.product_type.apply(
                lambda x: 'Yes' if x in clean_prod_type else 'Undefined')
        else:
            self.meta['clean_flag'] = 'Undefined'

        self.meta_no_cat = self.meta.loc[:,
                                         self.meta.columns.difference(['category'])]
        self.meta_no_cat.drop_duplicates(subset='prod_id', inplace=True)

        self.meta_no_cat.reset_index(drop=True, inplace=True)
        self.meta.reset_index(drop=True, inplace=True)

        if save:
            if self.source == 'sph':
                self.meta.to_feather(
                    self.sph.metadata_clean_path/f'cat_{self.clean_file_name}')
                self.meta_no_cat.to_feather(
                    self.sph.metadata_clean_path/f'no_cat_{self.clean_file_name}')
            elif self.source == 'bts':
                self.meta.to_feather(
                    self.bts.metadata_clean_path/f'cat_{self.clean_file_name}')
                self.meta_no_cat.to_feather(
                    self.bts.metadata_clean_path/f'no_cat_{self.clean_file_name}')
        return self.meta

    def detail_cleaner(self, data: pd.DataFrame, save: bool)->pd.DataFrame:
        """detail_cleaner [summary]

        [extended_summary]

        Args:
            data (pd.DataFrame): [description]
            save (bool): [description]

        Returns:
            pd.DataFrame: [description]
        """
        self.detail = data
        del data
        gc.collect()

        self.detail.replace('nan', '', regex=True, inplace=True)
        self.detail.product_name = self.detail.product_name.apply(
            unidecode.unidecode)

        if self.source == 'sph':
            # convert votes to numbers
            self.detail.votes = self.detail.votes.apply(lambda x: float(x.replace('K', ''))*1000
                                                        if x is not np.nan and 'k' in x.lower() else float(x))

            # split sephora rating distribution
            def split_rating_dist(x):
                if x is not np.nan:
                    ratings = literal_eval(x)
                    return ratings[1], ratings[3], ratings[5], ratings[7], ratings[9]
                else:
                    return (0.0 for i in range(5))

            self.detail['five_star'], self.detail['four_star'], self.detail['three_star'],\
                self.detail['two_star'],  self.detail['one_star'] = \
                zip(*self.detail.rating_dist.map(split_rating_dist))
            self.detail.drop('rating_dist', axis=1, inplace=True)

            # clean sephora would recommend
            self.detail.would_recommend = self.detail.would_recommend.str.replace(
                '%', '').astype(float)
            self.detail.would_recommend.fillna(0.0, inplace=True)
            self.detail.rename(
                {'would_recommend': 'would_recommend_percentage'}, inplace=True, axis=1)
            '''
            delete this out of sephora block after adding first review data to boots
            self.detail.first_review_date = pd.to_datetime(
                self.detail.first_review_date, infer_datetime_format=True)
            '''
        else:
            for i in ['five_star', 'four_star', 'three_star', 'two_star', 'one_star']:
                self.detail[i].fillna(0, inplace=True)
                self.detail[i][self.detail[i] == ''] = 0
            # create would recommend percentage for boots
            self.detail['would_recommend_percentage'] = 0.0
            # delete it after adding first review data to boots detail
            # self.detail['first_review_date'] = ''
        self.detail.reviews.fillna(0.0, inplace=True)
        self.detail.reviews[self.detail.reviews == ''] = 0.0
        self.detail.reviews = self.detail.reviews.astype(float)

        '''
        uncomment this block after adding first review data to boots detail
        self.detail.first_review_date = pd.to_datetime(
            self.detail.first_review_date, infer_datetime_format=True)
        '''
        self.detail.meta_date = pd.to_datetime(
            self.detail.meta_date, infer_datetime_format=True)

        self.detail.reset_index(drop=True, inplace=True)

        if save:
            if self.source == 'sph':
                self.detail.to_csv(
                    self.sph.detail_clean_path/f'{self.clean_file_name}', index=None)
            elif self.source == 'bts':
                self.detail.to_csv(
                    self.bts.detail_clean_path/f'cat_{self.clean_file_name}.csv', index=None)

        return self.detail

    def item_cleaner(self, data: pd.DataFrame, save: bool)->pd.DataFrame:
        """item_cleaner [summary]

        [extended_summary]

        Args:
            data (pd.DataFrame): [description]
            save (bool): [description]

        Returns:
            pd.DataFrame: [description]
        """
        self.item = data
        del data
        gc.collect()

    def review_cleaner(self, data: pd.DataFrame, save: bool)->pd.DataFrame:
        """review_cleaner [summary]

        [extended_summary]

        Args:
            data (pd.DataFrame): [description]
            save (bool): [description]

        Returns:
            pd.DataFrame: [description]
        """
        self.review = data
        del data
        gc.collect()

        self.review = self.review[~self.review.review_text.isna()]
        self.review.dropna(
            subset=['prod_id', 'review_text'], axis=0, inplace=True)
        self.review.reset_index(drop=True, inplace=True)

        if self.source == 'sph':
            '''
            it is a hassle to split helpful not helpful at later stage. Best is to get the data separately
            at crawler level or just split the string at the crawler level so that later processing is not
            required.
            '''
            # separate helpful and not helpful
            self.review['helpful_n'], self.review['helpful_y'] = zip(*self.review.helpful.apply(
                lambda x: literal_eval(x)[0] if not isinstance(x, int) else '0 \n 0').str.split('\n', expand=True).values)

            hlp_regex = re.compile('[a-zA-Z()]')
            self.review.helpful_y = self.review.helpful_y.apply(
                lambda x: hlp_regex.sub('', str(x)))
            self.review.helpful_n = self.review.helpful_n.apply(
                lambda x: hlp_regex.sub('', str(x)))

            self.review.drop('helpful', inplace=True, axis=1)

            # separate and create user attribute column
            def make_dict(x):
                return {k: v for d in literal_eval(x) for k, v in d.items() if k not in
                        ['hair_condition_chemically_treated_(colored,_relaxed,_or']}

            def get_attributes(x):
                if x.get('age') is not None:
                    age = x.get('age')
                elif x.get('age_over') is not None:
                    age = x.get('age_over')
                else:
                    age = np.nan

                if x.get('eye_color') is not None:
                    eye_c = x.get('eye_color')
                else:
                    eye_c = np.nan
                if x.get('hair_color') is not None:
                    hair_c = x.get('hair_color')
                else:
                    hair_c = np.nan

                if x.get('skin_tone') is not None:
                    skintn = x.get('skin_tone')
                else:
                    skintn = np.nan

                if x.get('skin_type') is not None:
                    skinty = x.get('skin_type')
                else:
                    skinty = np.nan

                return age, eye_c, hair_c, skintn, skinty

            self.review.user_attribute = self.review.user_attribute.map(
                make_dict)

            self.review['age'], self.review['eye_color'], self.review['hair_color'],\
                self.review['skin_tone'], self.review['skin_type'] = \
                zip(*self.review.user_attribute.apply(get_attributes))

            self.review.drop('user_attribute', inplace=True, axis=1)

        # convert ratings to numbers
        rating_regex = re.compile('stars|star|No')
        self.review.review_rating = self.review.review_rating.astype(str).apply(
            lambda x: rating_regex.sub('', x)).astype(int)
        # self.review.review_rating = self.review.review_rating.astype(int)
        # convert to pd datetime
        self.review.review_date = pd.to_datetime(
            self.review.review_date, infer_datetime_format=True)
        # clean and convert recommendation
        # if rating is 5 then it is assumed that the person recommends
        # id rating is 1 or 2 then it is assumed that the person does not recommend
        # for all the other cases data is not available
        self.review.recommend[(self.review.recommend.isin(['Recommends this product'])) | (
            self.review.review_rating == 5)] = 'Yes'
        self.review.recommend[(self.review.recommend != 'Yes') & (
            self.review.review_rating.isin([1, 2]))] = 'No'
        self.review.recommend[(self.review.recommend != 'Yes') & (
            self.review.review_rating.isin([3, 4]))] = 'not_avlbl'

        self.review.review_text = self.review.review_text.str.replace(
            '...read more', '')
        self.review.review_text = self.review.review_text.str.replace(
            '…read more', '')
        self.review = self.review.replace('\n', ' ', regex=True)
        self.review.reset_index(drop=True, inplace=True)

        if save:
            if self.source == 'sph':
                self.review.to_feather(
                    self.sph.review_clean_path/f'{self.clean_file_name}')
            elif self.source == 'bts':
                self.review.to_feather(
                    self.bts.review_clean_path/f'{self.clean_file_name}')

        return self.review
