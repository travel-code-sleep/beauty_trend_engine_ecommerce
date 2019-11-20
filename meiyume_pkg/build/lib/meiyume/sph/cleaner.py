from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import re
import string
import time
from ast import literal_eval
from datetime import datetime, timedelta
from functools import reduce
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import swifter
from tqdm import tqdm

from meiyume.utils import Logger, Sephora, nan_equal, show_missing_value

plt.style.use('fivethirtyeight')
np.random.seed(42)

class Cleaner(Sephora):
    """[summary]

    Arguments:
        Sephora {[type]} -- [description]
    """
    def __init__(self, path='.'):
        super().__init__(path=path)

    def clean_data(self, data, filename=None, rtn_typ='no_cat', save=True, logs=False):
        """[summary]

        Arguments:
            data {[file_path, pandas.DataFrame]} -- [files can be of csv or feather format]

        Keyword Arguments:
            filename {[type]} -- [description] (default: {None})
            rtn_typ {str} -- [description] (default: {'no_cat'})
            save {bool} -- [description] (default: {True})
            logs {bool} -- [description] (default: {False})

        Returns:
            [type] -- [description]
        """

        if type(data) == 'pandas.core.frame.DataFrame':
            self.data = data
            filename = filename
        else:
            filename = data
            try: self.data = pd.read_feather(data)
            except: self.data = pd.read_csv(data)

        data_def = self.find_data_def(str(filename))

        if logs:
            self.cleaner_log = Logger(f"sph_prod_{data_def}_cleaning", path=self.clean_log_path)
            self.logger, _ = self.cleaner_log.start_log()

        clean_file_name = 'cleaned_'+str(filename).split('\\')[-1]

        if data_def == 'meta':
            cleaned_metadata = self.meta_cleaner(rtn_typ)
            if save:
                cleaned_metadata.to_feather(self.metadata_clean_path/f'{rtn_typ}_{clean_file_name}')
            return cleaned_metadata
        if data_def == 'detail':
            cleaned_detail = self.detail_cleaner()
            cleaned_detail.drop_duplicates(inplace=True)
            cleaned_detail.reset_index(inplace=True, drop=True)
            if save:
                cleaned_detail.to_feather(self.detail_clean_path/f'{clean_file_name}')
            return cleaned_detail
        if data_def == 'item':
            cleaned_item = self.item_cleaner(return_type='None')
            cleaned_item.drop_duplicates(inplace=True)
            cleaned_item.reset_index(inplace=True, drop=True)
            if save:
                cleaned_item.to_feather(self.detail_clean_path/f'{clean_file_name}')
            return cleaned_item
        if data_def == 'review':
            cleaned_review = self.review_cleaner()
            cleaned_review.drop_duplicates(inplace=True)
            cleaned_review.reset_index(inplace=True, drop=True)
            if save:
                cleaned_review.to_feather(self.review_clean_path/f'{clean_file_name}')
            return cleaned_review

    def find_data_def(self, filename):
        if 'meta' in filename.lower(): return 'meta'
        elif 'detail' in filename.lower(): return 'detail'
        elif 'item' in filename.lower(): return 'item'
        elif 'review' in filename.lower(): return 'review'
        elif 'item' in filename.lower(): return 'item'
        else: raise MeiyumeException("Unable to determine data definition. Please provide correct file names.")

    def make_price(self, x):
        """[summary]

        Arguments:
            x {[type]} -- [description]
        """
        if '/' not in x and '-' not in x:
            return x, np.nan, np.nan
        elif '/' in x and '-' in x:
            p = re.split('-|/', x)
            return p[0], p[1], p[2]
        elif '/' in x and '-' not in x:
            p = re.split('/', x)
            return p[0], np.nan, p[1]
        elif x.count('-')>1 and '/' not in x:
            ts = [m.start() for m in re.finditer(' ', x)]
            p = x[ts[2]:].strip().split('-')
            return p[0], p[1], x[:ts[2]]
        elif '-' in x and x.count('-')<2 and '/' not in x:
            p = re.split('-', x)
            return p[0], p[1], np.nan
        else:
            return np.nan, np.nan, np.nan

    def clean_price(self, x):
        """[summary]

        Arguments:
            x {[type]} -- [description]
        """
        repls = ('$', ''), ('(', '/ '), (')', ''), ('value','')
        return reduce(lambda a, kv: a.replace(*kv), repls, x)

    def meta_cleaner(self, rtn_typ):
        """[summary]

        Arguments:
            rtn_typ {[type]} -- [description]
        """
        self.meta = self.data

        def fix_multi_low_price(x):
            """[summary]

            Arguments:
                x {[type]} -- [description]
            """
            if len(x)>7 and ' ' in x:
                p = x.split()
                return p[-1], p[0]
            else:
                return np.nan, np.nan

        #price cleaning
        self.meta['low_p'], self.meta['high_p'], self.meta['mrp'] = zip(*self.meta.price.swifter.apply(lambda x: self.clean_price(x)).swifter.apply(lambda y: self.make_price(y)))
        self.meta.drop('price', axis=1, inplace=True)
        self.meta.low_p[self.meta.low_p.swifter.apply(len)>7], self.meta.mrp[self.meta.low_p.swifter.apply(len)>7] =\
             zip(*self.meta.low_p[self.meta.low_p.swifter.apply(len)>7].swifter.apply(fix_multi_low_price))
        #create product id
        sph_prod_ids = self.meta.product_page.str.split(':', expand=True)
        sph_prod_ids.columns = ['a', 'b', 'id']
        self.meta['prod_id'] = 'sph_' + sph_prod_ids.id
        #clean rating
        clean_rating = re.compile('(\s*)stars|star|No(\s*)')
        self.meta.rating = self.meta.rating.swifter.apply(lambda x: clean_rating.sub('',x))
        self.meta.rating[self.meta.rating==''] = np.nan

        self.meta_no_cat = self.meta.loc[:, self.meta.columns.difference(['category'])]
        self.meta_no_cat.drop_duplicates(subset='prod_id', inplace=True)
        self.meta_no_cat.reset_index(drop=True, inplace=True)

        self.meta.reset_index(drop=True, inplace=True)

        if rtn_typ == 'no_cat':
            return self.meta_no_cat
        else:
            return self.meta

    def detail_cleaner(self):
        """[summary]

        Returns:
            [type] -- [description]
        """
        self.detail = self.data

        def convert_votes_to_number(x):
            """[summary]

            Arguments:
                x {[type]} -- [description]

            Returns:
                [type] -- [description]
            """
            if not nan_equal(np.nan, x):
                if 'K' in x: return int(x.replace('K',''))*1000
                else: return int(x)
            else: return np.nan

        self.detail.votes = self.detail.votes.swifter.apply(convert_votes_to_number)

        def split_rating_dist(x):
            if x is not np.nan:
                ratings = literal_eval(x)
                return ratings[1], ratings[3], ratings[5], ratings[7], ratings[9]
            else: return (np.nan for i in range(5))

        self.detail['five_star'], self.detail['four_star'], self.detail['three_star'], self.detail['two_star'], self.detail['one_star'] = \
            zip(*self.detail.rating_dist.swifter.apply(split_rating_dist))
        self.detail.drop('rating_dist', axis=1, inplace=True)

        self.detail.would_recommend = self.detail.would_recommend.str.replace('%','').astype(float)
        self.detail.rename({'would_recommend':'would_recommend_percentage'}, inplace=True, axis=1)
        return self.detail

    def calculate_ratings(self, x):
        """pass"""
        if x is np.nan:
            return (x['five_star']*5 + x['four_star']*4 + x['three_star']*3 + x['two_star']*2 + x['one_star'])\
                    /(x['five_star'] + x['four_star'] + x['three_star'] + x['two_star'] + x['one_star'])
        else: return x

    def item_cleaner(self):
        self.item = self.data
        self.item.item_price = self.item.item_price.swifter.apply(lambda x: self.clean_price(x))