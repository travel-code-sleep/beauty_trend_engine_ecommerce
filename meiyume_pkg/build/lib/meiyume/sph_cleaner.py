from __future__ import print_function, absolute_import
from pathlib import Path
import re
from functools import reduce
from datetime import datetime, timedelta
import time
import pandas as pd
import swifter
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from ast import literal_eval
import re
import string
import seaborn as sns
plt.style.use('fivethirtyeight')
np.random.seed(42)
from .utils import Logger, Sephora, nan_equal, show_missing_value

class Cleaner(Sephora):
    """[summary]

    Arguments:
        Sephora {[type]} -- [description]
    """
    def __init__(self, path=Path.cwd()):
        super().__init__(path=path)
        self.nan_equal = nan_equal
        self.show_missing_values = show_missing_value

    def clean_data(self, data, file_name=None, rtn_typ='no_cat', save=True, logs=False):
        """[summary]

        Arguments:
            data {[feather, pandas.DataFrame]} -- [description]

        Keyword Arguments:
            file_name {[type]} -- [description] (default: {None})
            rtn_typ {str} -- [description] (default: {'no_cat'})
            save {bool} -- [description] (default: {True})
            logs {bool} -- [description] (default: {False})

        Returns:
            [type] -- [description]
        """

        if type(data) == 'pandas.core.frame.DataFrame':
            self.data = data
            file_name = file_name
        else:
            file_name = data
            self.data = pd.read_feather(data)

        data_def = self.find_data_def(str(file_name))

        if logs:
            self.cleaner_log = Logger(f"sph_prod_{data_def}_extraction", path=self.clean_log_path)
            self.logger, _ = self.cleaner_log.start_log()

        clean_file_name = 'cleaned_'+str(file_name).split('\\')[-1]

        if data_def == 'meta':
            cleaned_metadata = self.meta_cleaner(rtn_typ)
            if save:
                cleaned_metadata.to_feather(self.metadata_clean_path/f'{rtn_typ}_{clean_file_name}')
            return cleaned_metadata
        if data_def == 'detail':
            cleaned_detail = self.detail_cleaner(return_type='None')
            if save:
                cleaned_detail.to_feather(self.detail_clean_path/'clean_file_name')
            return cleaned_detail
        if data_def == 'review':
            cleaned_review = self.review_cleaner()
            if save:
                cleaned_review.to_feather(self.review_clean_path/'clean_file_name')
            return cleaned_review

    def find_data_def(self, file_name):
        if 'meta' in file_name.lower():
            return 'meta'
        elif 'detail' in file_name.lower():
            return 'detail'
        elif 'item' in file_name.lower():
            return 'item'
        elif 'review' in file_name.lower():
            return 'review'
        else:
            raise MeiyumeException("Unable to determine data definition. Please provide correct file names.")

    def meta_cleaner(self, rtn_typ):
        """[summary]

        Arguments:
            rtn_typ {[type]} -- [description]
        """
        self.meta = self.data

        def make_price(x):
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

        def clean_price(x):
            """[summary]

            Arguments:
                x {[type]} -- [description]
            """
            repls = ('$', ''), ('(', '/ '), (')', ''), ('value','')
            return reduce(lambda a, kv: a.replace(*kv), repls, x)

        def fix_multi_lowp(x):
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
        self.meta.price = self.meta.price.swifter.apply(clean_price)
        self.meta['low_p'], self.meta['high_p'], self.meta['mrp'] = \
             zip(*self.meta['price'].swifter.apply(make_price))
        self.meta.drop('price', axis=1, inplace=True)
        self.meta.low_p[self.meta.low_p.swifter.apply(len)>7], self.meta.mrp[self.meta.low_p.swifter.apply(len)>7] =\
             zip(*self.meta.low_p[self.meta.low_p.swifter.apply(len)>7].swifter.apply(fix_multi_lowp))
        #create product id
        sph_prod_ids = self.meta.product_page.str.split(':', expand=True)
        sph_prod_ids.columns = ['a', 'b', 'id']
        self.meta['prod_id'] = sph_prod_ids.id
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
