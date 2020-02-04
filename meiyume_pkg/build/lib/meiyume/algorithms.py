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
    def __init__(self, path='.'):
        self.path = Path(path)
        self.output_path = self.path/'algorithm_outputs'
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.external_path = self.path/'external_data_sources'
        self.sph = Sephora(path='.')


class Ranker(StatAlgorithm):
    """[summary]

    Arguments:
        StatAlgorithm {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    def __init__(self, path='.'):
        super().__init__(path=path)

    def rank(self, meta_file=None, detail_file=None):
        """[summary]

        Keyword Arguments:
            meta_file {[file_path:str]} -- [description] (default: {None})
            detail_file {[file_path:str]} -- [description] (default: {None})

        Returns:
            [type] -- [description]
        """
        if meta_file:
            meta = pd.read_feather(meta_file)
        else:
            meta_files = self.sph.metadata_clean_path.glob(
                'cat_cleaned_sph_product_metadata_all*')
            meta = pd.read_feather(max(meta_files, key=os.path.getctime))
        meta.drop_duplicates(subset='prod_id', inplace=True)

        if detail_file:
            detail = pd.read_feather(detail_file)
        else:
            detail_files = self.sph.detail_clean_path.glob(
                'cleaned_sph_product_detail*')
            detail = pd.read_feather(max(detail_files, key=os.path.getctime))
        detail.drop_duplicates(subset='prod_id', inplace=True)

        meta.set_index('prod_id', inplace=True)
        detail.set_index('prod_id', inplace=True)
        meta_detail = meta.join(detail, how='inner', rsuffix='detail')
        meta_detail[['rating', 'reviews', 'votes', 'would_recommend_percentage', 'five_star', 'four_star', 'three_star',
                                          'two_star', 'one_star']] = meta_detail[['rating', 'reviews', 'votes', 'would_recommend_percentage', 'five_star', 'four_star', 'three_star',
                                                                                  'two_star', 'one_star']].apply(pd.to_numeric)
        meta_detail.reset_index(inplace=True)

        review_conf = meta_detail.groupby(by=['category', 'product_type'])[
            'reviews'].mean().reset_index()
        prior_rating = meta_detail.groupby(by=['category', 'product_type'])[
            'rating'].mean().reset_index()

        def total_stars(x): return x.reviews * x.rating

        def bayesian_estimate(x):
            c = int(round(review_conf['reviews'][(review_conf.category == x.category) & (
                review_conf.product_type == x.product_type)].values[0]))
            prior = int(round(prior_rating['rating'][(prior_rating.category == x.category) & (
                prior_rating.product_type == x.product_type)].values[0]))
            return (c * prior + x.rating * x.reviews	) / (c + x.reviews)

        meta_detail['total_stars'] = meta_detail.swifter.apply(
            lambda x: total_stars(x), axis=1).reset_index(drop=True)
        meta_detail['bayesian_estimate'] = meta_detail.swifter.apply(
            bayesian_estimate, axis=1)
        meta_detail.reset_index(drop=True, inplace=True)

        def ratio(x):
            """pass"""
            pstv_to_ngtv_stars = ((x.five_star + x.four_star)+1) / \
                ((x.two_star+1 + x.one_star+1)+1)
            pstv_to_total_stars = (x.five_star + x.four_star) / (x.reviews)
            return pstv_to_ngtv_stars, pstv_to_total_stars

        meta_detail['pstv_to_ngtv_stars'], meta_detail['pstv_to_total_stars'] = zip(*meta_detail.swifter.apply(
            lambda x: ratio(x), axis=1))

        meta_detail.meta_date = pd.to_datetime(
            meta_detail.meta_date, format='%Y-%m-%d')

        rank_file_name = 'ranked_'+str(meta).split('\\')[-1]
        meta_detail.drop(columns=['meta_datedetail',
                                  'product_namedetail'], axis=1, inplace=True)

        meta_detail.to_feather(
            self.output_path/f'ranked_cleaned_sph_product_meta_detail_all_{meta.meta_date.max()}')
        return meta_detail


class SexyIngredient(StatAlgorithm):
    def __init__(self, path='.'):
        """[summary]

        Arguments:
            StatAlgorithm {[class]} -- [description]

        Keyword Arguments:
            path {str} -- [description] (default: {'.'})
        """
        super().__init__(path=path)

    def make(self, meta_detail_file=None, ingredient_file=None):
        """[summary]

        Arguments:
            detail_file {[type]} -- [description]

        Keyword Arguments:
            meta_file {[type]} -- [description] (default: {None})
        """
        if meta_detail_file is None:
            meta_detail_files = self.output_path.glob(
                'ranked_cleaned_sph_product_meta_detail_all*')
            filename = max(meta_detail_files, key=os.path.getctime)
        else:
            filename = meta_detail_file
        meta_rank = pd.read_feather(filename)

        if ingredient_file is None:
            ingredient_files = self.sph.detail_clean_path.glob(
                'cleaned_sph_product_ingredient_all*')
            filename = max(ingredient_files, key=os.path.getctime)
        else:
            filename = ingredient_file
        self.ingredient = pd.read_feather(filename)

        old_ing_list = self.ingredient.ingredient[self.ingredient.new_flag.str.lower() == 'old'].str.strip(
        ).tolist()

        # find new ingredients based on new products
        def find_new_ingredient(x):
            if x.new_flag.lower() == 'new':
                if x.ingredient in old_ing_list:
                    return 'New_Product'
                else:
                    return 'New_Ingredient'
            else:
                return x.new_flag
        self.ingredient.new_flag = self.ingredient.swifter.apply(
            find_new_ingredient, axis=1)

        # rule based category assignment of ingredients
        # replace withe ingredient classification model prediction/inference

        food = pd.read_excel(self.sph.detail_path /
                             'ingredient_type_db.xlsx', sheet_name='food').name.dropna().str.strip().tolist()
        chemical = pd.read_excel(self.sph.detail_path /
                                 'ingredient_type_db.xlsx', sheet_name='chemical').name.dropna().str.strip().tolist()
        organic = pd.read_excel(
            self.sph.detail_path/'ingredient_type_db.xlsx', sheet_name='organic').name.dropna().str.strip().tolist()

        def assign_food_type(x):
            if any(w in food for w in x.split()):
                return 'food'
            else:
                return np.nan

        self.ingredient['ingredient_type'] = self.ingredient.ingredient.swifter.apply(
            assign_food_type)

        def assign_organic_chemical_type(x):
            if x.ingredient_type != 'food':
                if any(w in x.ingredient for w in organic):
                    return 'natural/organic'
                elif any(wc in x.ingredient for wc in chemical):
                    return 'chemical_compound'
                else:
                    return np.nan
            else:
                return x.ingredient_type

        self.ingredient['ingredient_type'] = self.ingredient.swifter.apply(
            assign_organic_chemical_type, axis=1)

        # assign vegan type
        self.ingredient.ingredient_type[self.ingredient.ingredient.str.contains(
            'vegan')] = 'vegan'

        # add columns for ranking
        meta_rank = meta_rank[['prod_id', 'brand', 'category', 'product_name',
                               'product_type', 'rating', 'bayesian_estimate', 'low_p', 'source']]
        meta_rank.set_index('prod_id', inplace=True)
        self.ingredient.set_index('prod_id', inplace=True)

        self.ingredient = self.ingredient.join(
            meta_rank, how='left', rsuffix='_meta')
        self.ingredient.reset_index(inplace=True)

        banned_ingredients = pd.read_csv(
            self.external_path/'banned_substances.csv')
        banned_ingredients.dropna(inplace=True)
        # banned_ingredients = banned_ingredients[banned_ingredients.substances.astype(
        #     str).apply(len) > 2]
        # banned_ingredients.substances = banned_ingredients.substances.str.strip(
        # ).str.lower().astype(str)
        # banned_ingredients = banned_ingredients[banned_ingredients.substances != '-']
        # s1 = pd.Series(['paraben', 'parabens'])
        # banned_ingredients.substances = banned_ingredients.substances.append(
        #     s1).reset_index(drop=True)

        self.ingredient['ban_flag'] = self.ingredient.ingredient.swifter.apply(
            lambda x: 'Yes' if x in banned_ingredients.substances.tolist() else 'No')
        self.ingredient.clean_flag[self.ingredient.ban_flag ==
                                   'Yes'] = 'Unclean'

        filename = str(filename).split("\\")[-1]
        self.ingredient.to_feather(
            self.output_path/f'ranked_{filename}')
        return self.ingredient
