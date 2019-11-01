from __future__ import print_function, absolute_import
from pathlib import Path
import re
from functools import reduce
from datetime import datetime, timedelta
import time
import pandas as pd
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
    
    def clean_data(self, file_name=file_name, rtn_typ='no_cat', save=True, logs=False):
        """[summary]
        
        Keyword Arguments:
            file_name {[type]} -- [description] (default: {file_name})
            rtn_typ {str} -- [description] (default: {'no_cat'})
            save {bool} -- [description] (default: {True})
            logs {bool} -- [description] (default: {False})
        """
        data_def = self.find_data_def(file_name)
        if logs:
            self.cleaner_log = Logger(f"sph_prod_{data_def}_extraction", path=self.clean_log_path)
            self.logger, _ = self.cleaner_log.start_log()
        if data_def == 'meta':
            cleaned_metadata = self.meta_cleaner(rtn_typ)
            if save:
                cleaned_metadata.to_feather(f'{file_name}_cleaned_{rtn_typ}')
            return cleaned_metadata
        if data_def == 'detail':
            cleaned_detail = self.detail_cleaner(return_type='None')
            if save:
                cleaned_detail.to_feather(f'{file_name}_cleaned_{rtn_typ}')
            return cleaned_detail
        if data_def == 'review':
            cleaned_review = self.review_cleaner()
            if save:
                cleaned_review.to_feather(f'{file_name}_cleaned')
            return cleaned_review
        
    def find_data_def(self, file_name):
        if 'meta' in file_name.lower():
            return data_def = 'meta'
        elif 'detail' in file_name.lower():
            return data_def = 'detail'
        elif 'item' in file_name.lower():
            return data_def = 'item'
        elif 'review' in file_name.lower():
            return data_def = 'review'
        else: 
            raise MeiyumeException("Unable to determine data definition. Please provide correct file names.")

    def meta_cleaner(self, rtn_typ):
        """pass"""
        self.meta = pd.read_feather(file_name)
        
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
        #price cleaning
        self.meta.price = self.meta.price.map(clean_price)
        self.meta['low_p'], self.meta['high_p'], self.meta['mrp'] = \
             zip(*self.meta['price'].map(make_price))
        self.meta.drop('price', axis=1, inplace=True)
        #create product id
        sph_prod_ids = self.meta.product_page.str.split(':', expand=True)
        sph_prod_ids.columns = ['a', 'b', 'id']
        self.meta['prod_id'] = sph_prod_ids.id
        #clean rating 
        clean_rating = re.compile('(\s*)stars|star|No(\s*)')
        self.meta.rating = self.meta.rating.apply(lambda x: clean_rating.sub('',x))
        self.meta.rating[self.meta.rating==''] = np.nan
        
        self.meta_no_cat = self.meta.loc[:, self.meta.columns.difference(['category'])]
        self.meta_no_cat.drop_duplicates(subset='prod_id', inplace=True)
        self.meta_no_cat.reset_index(drop=True, inplace=True)
        
        self.meta.reset_index(drop=True, inplace=True)
        
        if rtn_typ == 'no_cats':
            return self.sph_meta_no_cats
        else:
            return self.sph_meta
