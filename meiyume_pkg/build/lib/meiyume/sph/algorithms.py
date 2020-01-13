from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import re
import string
import time
import warnings
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
from meiyume.utils import Logger, Sephora, nan_equal, show_missing_value, MeiyumeException
from tqdm import tqdm

class StatAlgorithm(object):
    def __inti__(self, path='.'):
        self.path = Path(path)
        self.output_path = self.path/'algorithm_output'
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.sph = Sephora(path='.')

class Ranker(StatAlgorithm):
    """[summary]

    Arguments:
        StatAlgorithm {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    def __init__(self, path='.'):
        super().__(self,path)

    def rank(self):
        meta_files = self.sph.metadata_clean_path.glob(
            'cat_cleaned_sph_product_metadata_all*')
        meta = pd.read_feather(max(meta_files, key=os.path.getctime))
        meta.drop_duplicates(subset='prod_id', inplace=True)

        detail_files = self.sph.detail_clean_path.glob('cleaned_sph_product_detail*')
        detail = pd.read_feather(max(detail_files, key=os.path.getctime))
        detail.drop_duplicates(subset='prod_id', inplace=True)

        meta.set_index('prod_id', inplace=True)
        detail.set_index('prod_id', inplace=True)
        meta_detail = meta.join(detail, how='inner', rsuffix='detail')
        meta_detail[['rating', 'reviews', 'votes', 'would_recommend_percentage', 'five_star', 'four_star', 'three_star',
                                          'two_star', 'one_star']] = meta_detail[['rating', 'reviews', 'votes', 'would_recommend_percentage', 'five_star', 'four_star', 'three_star',
                                                                                  'two_star', 'one_star']].apply(pd.to_numeric)
        meta_detail.reset_index(inplace=True)

        review_conf = meta_detail.groupby(by=['category','product_type'])['reviews'].mean().reset_index()
        prior_rating = meta_detail.groupby(by=['category','product_type'])['rating'].mean().reset_index()

        def total_stars (x): return x.reviews * x.rating

        def bayesian_estimate (x):
            c = int(round(review_conf['reviews'][(review_conf.category==x.category) & (review_conf.product_type==x.product_type)].values[0]))
            prior = int(round(prior_rating['rating'][(prior_rating.category==x.category) & (prior_rating.product_type==x.product_type)].values[0]))
            return (c * prior + x.rating * x.reviews	) / (c + x.reviews)

        meta_detail['total_stars'] = meta_detail.swifter.apply(lambda x: total_stars(x), axis=1).reset_index(drop=True)
        meta_detail['bayesian_estimate'] = meta_detail.swifter.apply(bayesian_estimate, axis=1)
        meta_detail.reset_index(drop=True,inplace=True)

        def ratio(x, which='+ve-ve'):
            """pass"""
            if which == '+ve-ve':
                return ((x.five_star + x.four_star)+1) / ((x.two_star+1 + x.one_star+1)+1)
            elif which == '+ve_total':
                return (x.five_star + x.four_star) / (x.reviews)

        meta_detail['pstv_to_ngtv_stars'] = meta_detail.apply(lambda x: ratio(x), axis=1)
        meta_detail['pstv_to_total_stars'] = meta_detail.apply(lambda x: ratio(x, which='+ve_total'), axis=1)

        rank_file_name = 'ranked_'+str(meta).split('\\')[-1]
        meta_detail.to_feather(self.output_path/f'ranked_cleaned_sph_product_meta_detail_all_{meta.meta_date.max()}')
        return meta_detail


class SexyIngredient(StatAlgorithm):
    def __init__(self, path='.'):
        super().__(self,path)

    def prepare_data(self):
        meta_files = self.sph.metadata_clean_path.glob(
            'cat_cleaned_sph_product_metadata_all*')
        meta = pd.read_feather(max(meta_files, key=os.path.getctime))
        ingredient_files = self.sph.detail_clean_path.glob(
            'cleaned_sph_product_ingredient_all*')
        self.ingredient = pd.read_feather(
            max(ingredient_files, key=os.path.getctime))




