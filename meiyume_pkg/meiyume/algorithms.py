"""The module to run NLP Algorithms to extract insights from data.

There are several algorithms that carry out tasks such as summarization, classification, tagging, keyphrase extraction,
ranking, searching and relation extraction.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import gc
import os
# text lib imports
import re
import string
import warnings
from ast import literal_eval
from collections import Counter
from datetime import datetime, timedelta
from functools import reduce
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import pke
import seaborn as sns
# spaCy based imports
import spacy
import textacy
import textacy.ke as ke
# transformers imports
import torch
# import unidecode
# fast ai imports
from fastai import *
from fastai.text import *
from nltk.corpus import stopwords
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.matcher import Matcher
from textacy import preprocessing
from tqdm import tqdm
from tqdm.notebook import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from meiyume.utils import (Boots, Logger, MeiyumeException, ModelsAlgorithms,
                           RedShiftReader, S3FileManager, Sephora, chunks)

# # multiprocessing
# import multiprocessing as mp
# from multiprocessing import Pool
# from concurrent.futures import process

file_manager = S3FileManager()
db = RedShiftReader()
# process_manager = mp.Pool(mp.cpu_count())
warnings.simplefilter(action='ignore')
np.random.seed(1337)
tqdm.pandas()


class SexyMetaDetail(ModelsAlgorithms):
    """SexyMetaDetail uses Baayesian Ranker for product ranking and prepares data for Redshift ingestion.

    This module is a subclass of ModelsAlgorithms which initializes folder paths for data reading and
    model output storage.

    Args:
        ModelsAlgorithms (object): Parent class that defines folder paths and data locations.

    """

    def __init__(self, path: Union[str, Path] = Path.cwd()) -> None:
        """__init__ SexyMetaDetail class instace initializer.

        Args:
            path (Union[str, Path], optional): Folder path where the output folder structure will be saved
                                               and data will be read. Defaults to current directory(Path.cwd()).

        """
        super().__init__(path=path)

    def make(self, source: str, metadata: Optional[Union[Path, str, pd.DataFrame]] = None,
             detail_data: Optional[Union[Path, str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Make defines and runs bayesian ranker and data transformations to feed data correctly into redshift database.

        Args:
            source (str): source code of the metadata and detail files. (Current accepted values: [sph, bts])
            metadata (Optional[Union[Path, str, pd.DataFrame]], optional): Cleaned metadata file. Defaults to None.
            detail_data (Optional[Union[Path, str, pd.DataFrame]], optional): Cleaned detail file. Defaults to None.

        Raises:
            MeiyumeException: Raises exception if source or data files are incorrect.

        Returns:
            pd.DataFrame: Datafrane containing outputs from algorithms and other transformations.

        """
        if source not in ['bts', 'sph']:  # replace the list with sql source metadata table read
            raise MeiyumeException(
                "Unable to determine data source. Please provide correct source code.")

        if metadata:
            if not isinstance(metadata, pd.core.frame.DataFrame):
                try:
                    meta = pd.read_feather(metadata)
                except Exception as ex:
                    meta = pd.read_csv(metadata)
            else:
                meta = metadata
        else:
            if source == 'sph':
                meta_file_key = [i['Key'] for i in
                                 file_manager.get_matching_s3_keys(
                                     prefix='Feeds/BeautyTrendEngine/CleanedData/PreAlgorithm/cat_cleaned_sph_product_metadata_all')][-1]
                meta = file_manager.read_feather_s3(meta_file_key)
                # meta_files = self.sph.metadata_clean_path.glob(
                #     'cat_cleaned_sph_product_metadata_all*')
                # meta = pd.read_feather(max(meta_files, key=os.path.getctime))
            elif source == 'bts':
                # meta_files = self.bts.metadata_clean_path.glob(
                #     'cat_cleaned_bts_product_metadata_all*')
                # meta = pd.read_feather(max(meta_files, key=os.path.getctime))
                meta_file_key = [i['Key'] for i in
                                 file_manager.get_matching_s3_keys(
                                     prefix='Feeds/BeautyTrendEngine/CleanedData/PreAlgorithm/cat_cleaned_bts_product_metadata_all')][-1]
                meta = file_manager.read_feather_s3(meta_file_key)

        if source == 'sph':
            dft = meta.groupby('prod_id').product_type.apply(
                ' '.join).reset_index()

            exclude_pdt = ["bath-set-gifts",
                           "beauty-tool-gifts",
                           "clean-fragrance",
                           "clean-hair-care",
                           "clean-makeup",
                           "clean-skin-care",
                           "bath-set-gifts",
                           "cologne-gift-sets",
                           "fragrance-gift-sets",
                           "fragrance-gifts-gift-value-sets-men",
                           "gifts-for-her",
                           "gifts-for-men",
                           "gifts-for-teenage-girls",
                           "gifts-for-them",
                           "gifts-under-10",
                           "gifts-under-100",
                           "gifts-under-15",
                           "gifts-under-25",
                           "gifts-under-50",
                           "gifts-under-75",
                           "hair-gift-sets",
                           "home-fragrance-candle-gift-sets",
                           "makeup-bags-accessories-by-category-gifts",
                           "makeup-gift-sets",
                           "mens-gifts",
                           "perfume-gift-sets",
                           "skin-care-gift-sets",
                           "hair-brushes-combs-hair-tools-accessories-tools-accessories",
                           "eyelash-curlers-eyes-makeup",
                           "fragrance-gifts-gift-value-sets-men",
                           "cleansing-brushes-men",
                           "makeup-gift-sets",
                           "sunblock",
                           "professional-spa-tools-men",
                           "mens-body-spray-deodorant-products",
                           "blotting-paper-oil-control-skincare",
                           "teeth-whitening-tools"
                           "hair-straightener-curling-iron-flat-iron",
                           "makeup-brush-cleaner",
                           "hair-removal-shaving-bath-body",
                           "curling-wands-curling-irons",
                           "thinning-hair-loss",
                           "hair-brushes-combs-hair-tools-accessories-tools-accessories",
                           "lip-plumper",
                           "face-tanner-self-tanner-face",
                           "at-home-laser-hair-removal",
                           "anti-aging-tools-bath-body",
                           "bb-cc-cream-face-makeup",
                           "preciscion-tweezers",
                           ]

            def choose_type(x: str) -> str:
                """choose_type correct product_type of a product.

                Args:
                    x (str): Product types.

                Returns:
                    str: Correct product type.

                """
                x = x.split()
                t = list(set(x) - set(exclude_pdt))
                if len(t) > 0:
                    return t[0]
                else:
                    return x[0]
            dft.product_type = dft.product_type.apply(choose_type)

            meta.drop_duplicates(subset='prod_id', inplace=True)

            dft.set_index('prod_id', inplace=True)
            meta.set_index('prod_id', inplace=True)
            meta.drop('product_type', inplace=True, axis=1)

            meta = meta.join(dft, how='left')
            meta.reset_index(inplace=True)

        if source == 'bts':
            meta.drop_duplicates(subset='prod_id', inplace=True)

        if detail_data:
            if not isinstance(detail_data, pd.core.frame.DataFrame):
                try:
                    detail = pd.read_feather(detail_data)
                except Exception as ex:
                    detail = pd.read_csv(detail_data)
            else:
                detail = detail_data
        else:
            if source == 'sph':
                detail_file_key = [i['Key'] for i in
                                   file_manager.get_matching_s3_keys(
                                       prefix='Feeds/BeautyTrendEngine/CleanedData/PreAlgorithm/cleaned_sph_product_detail_all')][-1]
                detail = file_manager.read_feather_s3(detail_file_key)
                # detail_files = self.sph.detail_clean_path.glob(
                #     'cleaned_sph_product_detail*')
                # detail = pd.read_feather(
                #     max(detail_files, key=os.path.getctime))
            elif source == 'bts':
                detail_file_key = [i['Key'] for i in
                                   file_manager.get_matching_s3_keys(
                                       prefix='Feeds/BeautyTrendEngine/CleanedData/PreAlgorithm/cleaned_bts_product_detail_all')][-1]
                detail = file_manager.read_feather_s3(detail_file_key)
                # detail_files = self.bts.detail_clean_path.glob(
                #     'cleaned_bts_product_detail*')
                # detail = pd.read_feather(
                #     max(detail_files, key=os.path.getctime))
        detail.drop_duplicates(subset='prod_id', inplace=True)

        meta.set_index('prod_id', inplace=True)
        detail.set_index('prod_id', inplace=True)
        meta_detail = meta.join(detail, how='inner', rsuffix='detail')
        meta_detail[['rating', 'reviews', 'would_recommend_percentage', 'five_star', 'four_star', 'three_star',
                                          'two_star', 'one_star']] = meta_detail[['rating', 'reviews',
                                                                                  'would_recommend_percentage',
                                                                                  'five_star', 'four_star', 'three_star',
                                                                                  'two_star', 'one_star']].apply(pd.to_numeric)
        meta_detail.reset_index(inplace=True)

        review_conf = meta_detail.groupby(by=['category', 'product_type'])[
            'reviews'].mean().reset_index()
        prior_rating = meta_detail.groupby(by=['category', 'product_type'])[
            'rating'].mean().reset_index()

        meta_detail.sort_index(inplace=True)

        def total_stars(x): return x.reviews * x.rating

        def bayesian_estimate(x) -> float:
            """bayesian_estimate computes bayesian ranking score.

            Args:
                x : pandas iterrow.

            Returns:
                float: ranking score.

            """
            c = round(review_conf['reviews'][(review_conf.category == x.category) & (
                review_conf.product_type == x.product_type)].values[0])
            prior = round(prior_rating['rating'][(prior_rating.category == x.category) & (
                prior_rating.product_type == x.product_type)].values[0])
            return (c * prior + x.rating * x.reviews) / (c + x.reviews)

        meta_detail['total_stars'] = meta_detail.apply(
            lambda x: total_stars(x), axis=1).reset_index(drop=True)
        meta_detail['bayesian_estimate'] = meta_detail.apply(
            bayesian_estimate, axis=1)
        meta_detail.reset_index(drop=True, inplace=True)

        def ratio(x) -> Tuple[float, float]:
            """Ratio computes ration of positive to negative and total stars.

            Args:
                x : pandas iterrow.

            Returns:
                Tuple[float, float]: ratios.

            """
            pstv_to_ngtv_stars = ((x.five_star + x.four_star)+1) / \
                ((x.two_star+1 + x.one_star+1)+1)
            pstv_to_total_stars = (
                x.five_star + x.four_star + 1) / (x.reviews + 1)
            return pstv_to_ngtv_stars, pstv_to_total_stars

        meta_detail['pstv_to_ngtv_stars'], meta_detail['pstv_to_total_stars'] = zip(*meta_detail.apply(
            lambda x: ratio(x), axis=1))

        meta_detail.first_review_date.fillna('', inplace=True)

        meta_detail.meta_date = meta_detail.meta_date.apply(
            lambda x: str(pd.to_datetime(x).date()) if x is not np.nan else '')
        meta_detail.first_review_date = meta_detail.first_review_date.apply(
            lambda x: str(pd.to_datetime(x).date()) if x != '' else '')
        meta_detail.complete_scrape_flag = 'y'

        meta_detail.drop(columns=['meta_datedetail',
                                  'product_namedetail'], axis=1, inplace=True)
        meta_detail['complete_scrape_flag'] = 'y'

        columns = ["prod_id",
                   "product_name",
                   "product_page",
                   "brand",
                   "rating",
                   "category",
                   "product_type",
                   "new_flag",
                   "complete_scrape_flag",
                   "meta_date",
                   "source",
                   "low_p",
                   "high_p",
                   "mrp",
                   "clean_flag",
                   "abt_product",
                   "how_to_use",
                   "abt_brand",
                   "reviews",
                   "votes",
                   "would_recommend_percentage",
                   "five_star",
                   "four_star",
                   "three_star",
                   "two_star",
                   "one_star",
                   "total_stars",
                   "bayesian_estimate",
                   "pstv_to_ngtv_stars",
                   "pstv_to_total_stars",
                   "first_review_date"
                   ]
        meta_detail = meta_detail[columns]
        # meta_detail.product_name = meta_detail.product_name.apply(
        #     unidecode.unidecode)
        # meta_detail.brand = meta_detail.brand.apply(
        #     unidecode.unidecode)
        filename = f'ranked_cleaned_{source}_product_meta_detail_all_{pd.to_datetime(meta.meta_date.max()).date()}'
        meta_detail.to_feather(self.output_path/filename)

        meta_detail.fillna('', inplace=True)
        meta_detail = meta_detail.replace('\n', ' ', regex=True)
        meta_detail = meta_detail.replace('~', ' ', regex=True)

        filename = filename + '.csv'
        meta_detail.to_csv(self.output_path/filename, index=None, sep='~')

        file_manager.push_file_s3(file_path=self.output_path /
                                  filename, job_name='meta_detail')
        Path(self.output_path/filename).unlink()
        return meta_detail


'''class Ranker(ModelsAlgorithms):
    """Ranker [summary]

    [extended_summary]

    Args:
        ModelsAlgorithms ([type]): [description]

    Returns:
        [type]: [description]
    """

    def __init__(self, path: Path = '.')->None:
        """__init__ [summary]

        [extended_summary]

        Args:
            path (Path, optional): [description]. Defaults to '.'.
        """
        super().__init__(path=path)

    def rank(self, data: Union[Path, str, pd.DataFrame],
             group_by_columns: lst = ['category', 'product_type'],
             confidence_column: str = 'reviews',
             prior_column: str = 'rating'):

        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.read_feather(data)
            except Exception as ex:
                data = pd.read_csv(data)

        conf = data.groupby(by=group_by_columns)[
            confidence_column].mean().reset_index()
        prior = data.groupby(by=group_by_columns)[
            prior_column].mean().reset_index()

        data.sort_index(inplace=True)

        def total_stars(x): return x[confidence_column] * x[prior_column]

        def bayesian_estimate(x):
            c = round(conf[confidence_column][(conf[group_by_columns[0]] == x[group_by_columns[0]]) & (
                conf[group_by_columns[1]] == x[group_by_columns[1]])].values[0])
            p = round(prior[prior_column][(prior.category == x.category) & (
                prior[group_by_columns[1]] == x[group_by_columns[1])].values[0])
            return (c * p + x.rating * x.reviews	) / (c + x.reviews)

        meta_detail['total_stars'] = meta_detail.apply(
            lambda x: total_stars(x), axis=1).reset_index(drop=True)
        meta_detail['bayesian_estimate'] = meta_detail.apply(
            bayesian_estimate, axis=1)
        meta_detail.reset_index(drop=True, inplace=True)

        def ratio(x):
            """pass"""
            pstv_to_ngtv_stars = ((x.five_star + x.four_star)+1) / \
                ((x.two_star+1 + x.one_star+1)+1)
            pstv_to_total_stars = (x.five_star + x.four_star) / (x.reviews)
            return pstv_to_ngtv_stars, pstv_to_total_stars

        meta_detail['pstv_to_ngtv_stars'], meta_detail['pstv_to_total_stars'] = zip(*meta_detail.apply(
            lambda x: ratio(x), axis=1))'''


class SexyIngredient(ModelsAlgorithms):
    """SexyIngredient uses nlp algorithms to transform data for Redshift ingestion.

    This module is a subclass of ModelsAlgorithms which initializes folder paths for data reading and
    model output storage.

    Major functions of SexyIngredient are:
    * 1. Ingredient classification.
    * 2. Ingredient tagging.
    * 3. Banned ingredient identification.
    * 4. Data transformations to feed into Redshift database.

    Args:
        ModelsAlgorithms (object): Parent class that defines folder paths and data locations.

    """

    def __init__(self, path: Union[str, Path] = Path.cwd()):
        """__init__ SexyIngredient class instace initializer.

        Args:
            path (Union[str, Path], optional): Folder path where the output folder structure will be saved
                                               and data will be read. Defaults to current directory(Path.cwd()).

        """
        super().__init__(path=path)

    def make(self, source: str, meta_detail_data: Optional[Union[Path, str, pd.DataFrame]] = None,
             ingredient_data: Optional[Union[Path, str, pd.DataFrame]] = None) -> pd.DataFrame:
        """make defines and runs ingredient classifier, tagger, identifier and data transformations.

        Args:
            source (str): source code of the metadata and detail files. (Current accepted values: [sph, bts])
            meta_detail_data (Optional[Union[Path, str, pd.DataFrame]], optional): Meta_Detail data generated by
                                                                                   SexyMetaDetail algorithm. Defaults to None.
            ingredient_data (Optional[Union[Path, str, pd.DataFrame]], optional): Cleaned ingredient data. Defaults to None.

        Raises:
            MeiyumeException: Raises exception if source or data files are incorrect.

        Returns:
            pd.DataFrame: Datafrane containing outputs from algorithms and other transformations.

        """
        if source not in ['bts', 'sph']:  # replace the list with sql source metadata table read
            raise MeiyumeException(
                "Unable to determine data source. Please provide correct source code.")

        if meta_detail_data:
            if not isinstance(meta_detail_data, pd.core.frame.DataFrame):
                try:
                    meta_rank = pd.read_feather(meta_detail_data)
                except Exception as ex:
                    meta_rank = pd.read_csv(meta_detail_data)
            else:
                meta_rank = meta_detail_data
        else:
            if source == 'sph':
                meta_detail_files = self.output_path.glob(
                    'ranked_cleaned_sph_product_meta_detail_all*')
                meta_rank = pd.read_feather(
                    max(meta_detail_files, key=os.path.getctime))
            elif source == 'bts':
                meta_detail_files = self.output_path.glob(
                    'ranked_cleaned_bts_product_meta_detail_all*')
                meta_rank = pd.read_feather(
                    max(meta_detail_files, key=os.path.getctime))

        if ingredient_data:
            if not isinstance(ingredient_data, pd.core.frame.DataFrame):
                try:
                    self.ingredient = pd.read_feather(ingredient_data)
                except Exception as ex:
                    self.ingredient = pd.read_csv(ingredient_data)
            else:
                self.ingredient = ingredient_data
        else:
            if source == 'sph':
                # ingredient_file_key = [i['Key'] for i in
                #                        file_manager.get_matching_s3_keys(
                #                            prefix='Feeds/BeautyTrendEngine/CleanedData/PreAlgorithm/cleaned_sph_product_ingredient_all')][-1]
                # self.ingredient = file_manager.read_feather_s3(
                #     ingredient_file_key)
                ingredient_files = self.sph.detail_clean_path.glob(
                    'cleaned_sph_product_ingredient_all*')
                self.ingredient = pd.read_feather(
                    max(ingredient_files, key=os.path.getctime))
            elif source == 'bts':
                # ingredient_file_key = [i['Key'] for i in
                #                        file_manager.get_matching_s3_keys(
                #                            prefix='Feeds/BeautyTrendEngine/CleanedData/PreAlgorithm/cleaned_bts_product_ingredient_all')][-1]
                # self.ingredient = file_manager.read_feather_s3(
                #     ingredient_file_key)
                ingredient_files = self.bts.detail_clean_path.glob(
                    'cleaned_bts_product_ingredient_all*')
                self.ingredient = pd.read_feather(
                    max(ingredient_files, key=os.path.getctime))

        old_ing_list = self.ingredient.ingredient[self.ingredient.new_flag == ''].str.strip(
        ).tolist()

        # find new ingredients based on new products
        def find_new_ingredient(x):
            if x.new_flag == 'new':
                if x.ingredient in old_ing_list:
                    return 'new_product'
                else:
                    return 'new_ingredient'
            else:
                return x.new_flag
        self.ingredient.new_flag = self.ingredient.apply(
            find_new_ingredient, axis=1)

        # rule based category assignment of ingredients
        # replace withe ingredient classification model prediction/inference

        food = pd.read_excel(self.external_path /
                             'ingredient_type_db.xlsx', sheet_name='food').name.dropna().str.strip().tolist()
        chemical = pd.read_excel(self.external_path /
                                 'ingredient_type_db.xlsx', sheet_name='chemical').name.dropna().str.strip().tolist()
        organic = pd.read_excel(
            self.external_path/'ingredient_type_db.xlsx', sheet_name='organic').name.dropna().str.strip().tolist()

        def assign_food_type(x):
            if any(w in food for w in x.split()):
                return 'food'
            else:
                return ''

        self.ingredient['ingredient_type'] = self.ingredient.ingredient.apply(
            assign_food_type)

        def assign_organic_chemical_type(x):
            if x.ingredient_type != 'food':
                if any(w in x.ingredient for w in organic):
                    return 'natural/organic'
                elif any(wc in x.ingredient for wc in chemical):
                    return 'chemical_compound'
                else:
                    return ''
            else:
                return x.ingredient_type

        self.ingredient['ingredient_type'] = self.ingredient.apply(
            assign_organic_chemical_type, axis=1)

        # assign vegan type
        self.ingredient.ingredient_type[self.ingredient.vegan_flag ==
                                        'vegan'] = 'vegan'

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

        self.ingredient['ban_flag'] = self.ingredient.ingredient.apply(
            lambda x: 'yes' if x in banned_ingredients.substances.tolist() else 'no')
        self.ingredient.clean_flag[self.ingredient.ban_flag ==
                                   'yes'] = 'unclean'
        columns = ["prod_id",
                   "clean_flag",
                   "ingredient",
                   "new_flag",
                   "meta_date",
                   "ingredient_type",
                   "brand",
                   "category",
                   "product_name",
                   "product_type",
                   "rating",
                   "bayesian_estimate",
                   "low_p",
                   "source",
                   "ban_flag",
                   "vegan_flag"
                   ]
        self.ingredient = self.ingredient[columns]

        filename = f'ranked_cleaned_{source}_product_ingredient_all_{pd.to_datetime(self.ingredient.meta_date.max()).date()}'
        self.ingredient.to_feather(self.output_path/filename)

        self.ingredient.fillna('', inplace=True)
        self.ingredient = self.ingredient[self.ingredient.ingredient != '']
        self.ingredient = self.ingredient[self.ingredient.ingredient.str.len(
        ) < 200]
        self.ingredient.reset_index(inplace=True, drop=True)
        self.ingredient = self.ingredient.replace('\n', ' ', regex=True)
        self.ingredient = self.ingredient.replace('~', ' ', regex=True)

        filename = filename + '.csv'
        self.ingredient.to_csv(self.output_path/filename, index=None, sep='~')

        file_manager.push_file_s3(file_path=self.output_path /
                                  filename, job_name='ingredient')
        Path(self.output_path/filename).unlink()
        return self.ingredient


class KeyWords(ModelsAlgorithms):
    """KeyWords uses nlp algorithms to extract keywords and phrases from text.

    This module is a subclass of ModelsAlgorithms which initializes folder paths for data reading and
    model output storage.

    Major functions of KeyWords Class are:
    * 1. Keyword/Phrase extraction.
    * 2. Keyword summarization over large bodies of text.
    * 3. Keyphrase sentiment classification and segregation into sentiment phrases.

    Args:
        ModelsAlgorithms (object): Parent class that defines folder paths and data locations.

    """

    def __init__(self, path: Union[str, Path] = '.'):
        """__init__ KeyWords class instace initializer.

        Args:
            path (Union[str, Path], optional): Folder path where the output folder structure will be saved
                                               and data will be read. Defaults to current directory(Path.cwd()).

        """
        super().__init__(path=path)
        self.en = textacy.load_spacy_lang(
            'en_core_web_lg', disable=("parser",))
        self.analyser = SentimentIntensityAnalyzer()
        self.nlp = spacy.load("en_core_web_lg")

    def get_no_of_words(self, length: int) -> int:
        """get_no_of_words determines the no. of keywords to extract depending word count of input text.

        Args:
            length (int): word count of input text.

        Returns:
            int: no. of keywords to extract.

        """
        p = 0.2
        if length < 50:
            k = 2
        elif length >= 50 and length <= 100:
            k = 6
        elif length > 100 and length <= 300:
            k = 15
        elif length > 300 and length <= 1000:
            p = 0.18
            k = 35
        elif length > 1000 and length <= 5000:
            p = 0.16
            k = 60
        elif length > 5000:
            p = 0.14
            k = 100
        return int(round(k)), p

    def extract_keywords(self, text: Union[str, spacy.tokens.doc.Doc],
                         include_pke: bool = False, is_doc: bool = False) -> str:
        """extract_keywords uses nlp algorithms to identify and extract keywords.

        Args:
            text (Union[str, spacy.tokens.doc.Doc]): Input text or document.
            include_pke (bool, optional): Whether to include PKE lib keywords . Defaults to False.
            is_doc (bool, optional): Whether the input is a spacey document. Defaults to False.

        Returns:
            str: Keywords separated by comma.

        """
        try:
            if is_doc:
                doc = text
                length = len(doc.text.split())
            else:
                length = len(text.split())
                doc = None
            if length > 7:
                k, p = self.get_no_of_words(length)

                if doc is None:
                    doc = textacy.make_spacy_doc(text, lang=self.en)

                if include_pke:
                    self.extractor_por = pke.unsupervised.PositionRank()
                    self.extractor_por.load_document(input=text, language='en')
                    self.extractor_por.candidate_selection()
                    self.extractor_por.candidate_weighting()

                    self.extractor_yke = pke.unsupervised.YAKE()
                    self.extractor_yke.load_document(
                        input=text, language='en')
                    stoplist = stopwords.words('english')
                    self.extractor_yke.candidate_selection(
                        n=3, stoplist=stoplist)
                    self.extractor_yke.candidate_weighting(
                        window=4, stoplist=stoplist, use_stems=False)
                    pke_keywords = [i[0] for i in self.extractor_por.get_n_best(n=k) if i[1] > 0.02] +\
                        [i[0] for i in self.extractor_yke.get_n_best(
                            n=k, threshold=0.8, redundancy_removal=True) if i[1] > 0.02]
                else:
                    pke_keywords = []

                keywords = [i[0] for i in ke.textrank(doc, window_size=4, normalize='lower', topn=k) if i[1] > 0.02] +\
                    [i[0] for i in ke.sgrank(doc, ngrams=(
                        1, 2, 3, 4, 5), normalize="lower", topn=p) if i[1] > 0.02] + pke_keywords

                self.keywords = sorted(list(set(keywords)), reverse=False,
                                       key=lambda x: len(x))

                if len(self.keywords) > 0:
                    self.bad_words = []
                    for i in range(len(self.keywords)):
                        for j in range(i + 1, len(self.keywords)):
                            if self.keywords[i] in self.keywords[j]:
                                self.bad_words.append(self.keywords[i])

                clean_keywords = [
                    word for word in self.keywords if word not in self.bad_words]

                return ', '.join(clean_keywords)  # list(set(keywords))
            else:
                return None
        except IndexError:
            return 'failed'

    def summarize_keywords(self, keywords: Union[str, list], sep: str = ',', exclude_keys: list = [],
                           max_keys: int = -1) -> dict:
        """summarize_keywords summarizes generated keywords of a document.

        Args:
            keywords (Union[str, list]): Input keywords delimited by sep to summarize.
            sep (str, optional): Keyword delimiter. Defaults to ','.
            exclude_keys (list, optional): List of keys to exclude from generated summary dictionary. Defaults to [].
            max_keys (int, optional): Top N keywords to summarize and retrieve. Defaults to -1.

        Returns:
            dict[str, int]: Dictionary containing keywords and their frequencies.

        """
        self.exclude_keys = ['', 'product', 'easy', 'glad', 'minutes', 'fingers', 'job', 'year',
                             'negative reviews', 'negative review', 'stuff', 'store', 'lot',
                             ]
        self.exclude_keys.extend(exclude_keys)

        if type(keywords) == list:
            keywords = Counter(keywords).items()
        else:
            keywords = Counter(keywords.split(f'{sep}')).items()

        skw = sorted([(k.strip(), v) for k, v in keywords if v >= 3 and k.strip() != ''],
                     reverse=False, key=lambda x: len(x[0]))

        self.irrelevant_keys = []
        for i in skw:
            tags = []
            tokens = self.nlp(i[0])
            for token in tokens:
                tags.append(token.pos_)
            if all(t not in ['NOUN', 'PROPN', 'PRON'] for t in tags):
                self.irrelevant_keys.append(i[0])
        skw = [item for item in skw if item[0] not in self.irrelevant_keys]

        for i in range(len(skw)):
            for j in range(i + 1, len(skw)):
                if skw[i][0] in skw[j][0]:
                    self.exclude_keys.append(skw[i][0])

        keyword_summary = {item[0]: item[1]
                           for item in skw if item[0] not in self.exclude_keys}

        if len(keyword_summary.keys()) > max_keys and max_keys != -1:
            self.keyword_summary = dict(sorted(keyword_summary.items(), key=lambda x: len(
                x[0].split()), reverse=True)[:max_keys])
        else:
            self.keyword_summary = keyword_summary

        return self.keyword_summary

    def generate_sentiment_ngrams(self, text: str, n: list = [2, 3, 4, 5, 6], min_freq: int = 2,
                                  max_terms: int = -1, exclude_ngrams: list = [],
                                  sentiment: str = 'negative', sentiment_threshold: float = 0.0,
                                  increase_threshold_by: float = 0.2) -> dict:
        """generate_sentiment_ngrams extracts sentiment wise ngrams from input text.

        Depending on the input text's sentiment this method generates the most crucial n-gram phrases of a document
        that are above the sentiment score threshold in either positive or negative direction.

        Args:
            text (str): Input text
            n (list, optional): ngrams word length. Defaults to [2, 3, 4, 5, 6].
            min_freq (int, optional): Minimum number of times a n-gram occurs to qualify for extraction. Defaults to 2.
            max_terms (int, optional): Top N ngrams/keyphrases to extract. Defaults to -1.
            exclude_ngrams (list, optional): List of ngrams to exclude. Defaults to [].
            sentiment (str, optional): Input text sentiment. Defaults to 'negative'.
            sentiment_threshold (float, optional): Positive/Negative sentiment threshold only above which a phrase will be
                                                   accepted. Defaults to 0.0.
            increase_threshold_by (float, optional): Sentiment threshold increment step. Defaults to 0.2.

        Returns:
            dict[Any, int]: Dictionary containing sentiment phrases and their frequencies.

        """
        self.exclude_ngram_list = ['good product', 'great product', 'works best', 'love love', 'great job', 'great tool',
                                   'works good', 'tool is great', 'recommend this product', 'tool works great', 'like this tool',
                                   'best way to use', 'ready to come', 'better to use', 'works pretty', 'useful tool',
                                   'skin care', 'love this product', 'like this product', 'stuff is amazing',
                                   'love this stuff', 'love love love', 'far so good', ]
        self.exclude_ngram_list.extend(exclude_ngrams)
        self.sentiment_threshold_ = sentiment_threshold

        doc = textacy.make_spacy_doc(text, lang=self.en)

        ngrams = []
        while len(ngrams) == 0:
            if min_freq == 0:
                break
            for gram in n:
                ngrams.extend(list(textacy.extract.ngrams(doc, gram, filter_stops=True, filter_punct=True,
                                                          filter_nums=True, min_freq=min_freq)))

            ngrams = [str(n) for n in ngrams]
            ngrams = [n.replace('nt ', 'dont ') for n in ngrams]

            if min_freq == 1:
                self.sentiment_threshold_ += increase_threshold_by
            if sentiment == 'negative':
                self.sentiment_threshold_ = -self.sentiment_threshold_

            if sentiment == 'positive':
                ngrams = [n for n in ngrams if self.analyser.polarity_scores(
                    n)['compound'] > self.sentiment_threshold_]
            elif sentiment == 'negative':
                ngrams = [n for n in ngrams if self.analyser.polarity_scores(
                    n)['compound'] < self.sentiment_threshold_]

            min_freq -= 1

        if len(ngrams) > 1:
            ngrams_u = list(set(ngrams))
            ngrams_u = sorted(ngrams_u, reverse=False, key=lambda x: len(x))
            for i in range(len(ngrams_u)):
                for j in range(i + 1, len(ngrams_u)):
                    if ngrams_u[i] in ngrams_u[j]:
                        self.exclude_ngram_list.append(ngrams_u[i])
            ngrams = [n for n in ngrams if n not in self.exclude_ngram_list]

            self.irrelevant_ngrams = []
            ngrams_u = list(set(ngrams))
            for i in ngrams_u:
                tags = []
                tokens = self.nlp(i)
                for token in tokens:
                    tags.append(token.pos_)
                if all(t not in ['NOUN', 'PROPN', 'PRON', 'ADJ', ] for t in tags):
                    self.irrelevant_ngrams.append(i)
            selected_grams = [
                n for n in ngrams if n not in self.irrelevant_ngrams]
            selected_grams = dict(Counter(selected_grams))

            if len(selected_grams.keys()) > max_terms and max_terms != -1:
                self.ngrams = dict(sorted(selected_grams.items(), key=lambda x: len(
                    x[0].split()), reverse=True)[:max_terms])
            else:
                self.ngrams = selected_grams
        else:
            self.ngrams = {}
        return self.ngrams


class PredictSentiment(ModelsAlgorithms):
    """PredictSentiment carries out sentiment classification using ULMfit language model.

    This module is a subclass of ModelsAlgorithms which initializes folder paths for data reading and
    model output storage.

    Major functions of PredictSentiment Class are:
    * 1. Single Input Sentiment Prediction
    * 2. Batch Sentiment Prediction

    Args:
        ModelsAlgorithms (object): Parent class that defines folder paths and data locations.

    """

    def __init__(self, model_file: str = 'sentiment_model_two_class',
                 data_vocab_file: str = 'sentiment_class_databunch_two_class.pkl',
                 model_path: Path = None, path: Union[Path, str] = Path.cwd()) -> None:
        """__init__ PredictSentiment class instace initializer.

        Args:
            model_file (str, optional): Trained model weights file. Defaults to 'sentiment_model_two_class'.
            data_vocab_file (str, optional): Trained model vocabulary file. Defaults to 'sentiment_class_databunch_two_class.pkl'.
            model_path (Path, optional): Folder path for trained model files. Defaults to None.
            path (Union[Path,str], optional): Folder path where the output folder structure will be saved
                                              and data will be read. Defaults to current directory(Path.cwd()).

        """
        super().__init__(path=path)
        if model_path:
            data_class = load_data(path=model_path,
                                   file=data_vocab_file)
        else:
            data_class = load_data(path=self.model_path,
                                   file=Path(f'data/{data_vocab_file}'))

        self.learner = text_classifier_learner(
            data_class, AWD_LSTM, drop_mult=0.5)
        self.learner.load(model_file)

    def predict_instance(self, text: str) -> Tuple[str, float, float]:
        """predict_instance classifies the sentiment of one input document/text.

        Args:
            text (str): Input text

        Returns:
            Tuple[str, float, float]: Predicted Label, Positive Probability, Negative Probability

        """
        pred = self.learner.predict(text)
        return pred[0], pred[1].numpy(), pred[2].numpy()

    def predict_batch(self, text_column_name: str, data: Union[Path, str, pd.DataFrame],
                      save: bool = False) -> pd.DataFrame:
        """predict_batch classifies sentiment of an array of documents.

        Takes a pandas dataframe or filepath and the column name of which sentiment will be classified.

        Args:
            text_column_name (str): Column name that contains instances for classification.
            data (Union[Path, str, pd.DataFrame]): Data file or path. (csv or feather file format accepted)
            save (bool, optional): Whether to save the sentiment classification result to disk. Defaults to False.

        Returns:
            pd.DataFrame: Output of the sentiment classification algorithm contaiing predicted labels and probabilities.

        """
        if type(data) != pd.core.frame.DataFrame:
            filename = data
            try:
                data = pd.read_feather(Path(data))
            except Exception as ex:
                data = pd.read_csv(Path(data))
        else:
            filename = None

        data.reset_index(inplace=True, drop=True)

        self.learner.data.add_test(data[text_column_name])
        prob_preds = self.learner.get_preds(
            ds_type=DatasetType.Test, ordered=True)

        data = pd.concat([data, pd.DataFrame(
            prob_preds[0].numpy(), columns=['neg_prob', 'pos_prob'])], axis=1)

        data.neg_prob = data.neg_prob.apply(lambda x: round(x, 3))
        data.pos_prob = data.pos_prob.apply(lambda x: round(x, 3))

        data['sentiment'] = data.apply(
            lambda x: 'positive' if x.pos_prob > x.neg_prob else 'negative', axis=1)
        data.reset_index(inplace=True, drop=True)

        if filename:
            filename = str(filename).split('\\')[-1]
            if save:
                data.to_feather(
                    self.output_path/f'with_sentiment_{filename}')

        return data


class PredictInfluence(ModelsAlgorithms):
    """PredictInfluence determines whether a review is influenced by giveaways or marketing events.

    This module is a subclass of ModelsAlgorithms which initializes folder paths for data reading and
    model output storage.

    Major functions of PredictInfluence Class are:
    * 1. Tokenize input texts/documents.
    * 2. Predict influence on a single input text/document.
    * 3. Predict influence on a batch of texts/documents.

    Args:
        ModelsAlgorithms (object): Parent class that defines folder paths and data locations.

    """

    def __init__(self) -> None:
        """__init__ PredictInfluence class instace initializer.

        Initializes the influence keywords, punctuations, stopwords and english language parser.

        """
        super().__init__()
        self.lookup_ = ['free sample', 'free test', 'complimentary test', 'complimentary review', 'complimentary review',
                        'complimentary test', 'receive product free', 'receive product complimentary', 'product complimentary',
                        'free test', 'product free', 'test purpose', 'got this as a sample']
        self.punctuations = string.punctuation
        self.stopwords = list(STOP_WORDS)
        self.parser = English()

    def spacy_tokenizer(self, text: str) -> str:
        """spacy_tokenizer tokenizes input text/document.

        Args:
            text (str): Input text.

        Returns:
            str: Tokenized text.

        """
        tokens = self.parser(text)
        tokens = [word.lemma_.lower().strip() if word.lemma_ !=
                  "-PRON-" else word.lower_ for word in tokens]
        tokens = [
            word for word in tokens if word not in self.stopwords and word not in self.punctuations]
        tokens = " ".join([i for i in tokens])
        return tokens

    def predict_instance(self, text: str) -> str:
        """predict_instance predicts influence on one single input.

        Args:
            text (str): Input text.

        Returns:
            str: Predicted label.

        """
        tokenized_text = self.spacy_tokenizer(text)
        if any(i in tokenized_text.lower() for i in self.lookup_):
            return 'influenced'
        else:
            return 'not_influenced'

    def predict_batch(self, text_column_name: str, data: Union[Path, str, pd.DataFrame], save: bool = False):
        """predict_batch predicts influce on a batch of documents.

        Takes a pandas dataframe or filepath and the column name of which influence will be classified.

        Args:
            text_column_name (str): Column name that contains instances for classification.
            data (Union[Path, str, pd.DataFrame]): Data file or path. (csv or feather file format accepted)
            save (bool, optional): Whether to save the influence classification result to disk. Defaults to False.

        Returns:
            data(pd.DataFrame): Output of the influence classification algorithm contaiing predicted labels.

        """
        if not isinstance(data, pd.core.frame.DataFrame):
            filename = data
            try:
                data = pd.read_feather(Path(data))
            except Exception as ex:
                data = pd.read_csv(Path(data))
        else:
            filename = None

        data['tokenized_text'] = data[text_column_name].progress_apply(
            self.spacy_tokenizer).str.lower()
        data['is_influenced'] = data.tokenized_text.progress_apply(lambda x: "yes" if
                                                                   any(y in x for y in self.lookup_) else "no")

        data.reset_index(inplace=True, drop=True)
        if filename:
            filename = str(filename).split('\\')[-1]
            if save:
                data.to_feather(
                    self.output_path/f'{filename}')
        return data


class SelectCandidate(ModelsAlgorithms):
    """SelectCandidate selects best document/text candidates for summarization of reviews, keywords etc.

    Args:
        ModelsAlgorithms (object): Parent class that defines folder paths and data locations.

    """

    def __init__(self) -> None:
        """__init__ SelectCandidate class instace initializer."""
        super().__init__()

    def select(self, data: Union[str, Path, pd.DataFrame], weight_column: str,
               groupby_columns: Union[str, list], fraction: float = 0.3,
               select_column=Optional[str], sep: str = ' ', drop_weights: bool = True,
               inverse_weights: bool = True, keep_all: bool = True, **kwargs) -> pd.DataFrame:
        """Select method determines which documents are the best candidates based on weighted threshold.

        Args:
            data (Union[str, Path, pd.DataFrame]): dataset. prefarably dataframe, csv or feather file.
            weight_column (str): numerical column on which weights will be calculated
            groupby_columns (Union[str, list]): columns over which values the sampling candidate groups will be generated
            fraction (float, optional): fraction of data to keep. Defaults to 0.3.
            select_column (str, optional): the column of which rows will be combined over groups and
                                              be returned as group columns + combined data column. Defaults to None.
            sep (str, optional): separator by which the select column rows will be joined over groups. Defaults to ' '.
            drop_weights (bool, optional): whether to drop the weight column values. Defaults to True.
            keep_all (bool, optional): keep all the original groups. Defaults to True.

        Returns:
            data_sample(pd.DataFrame): Sampled data after probabilistic weighted candidate selection.

        """
        if type(groupby_columns) != list:
            groupby_columns = [groupby_columns]

        if type(data) != pd.core.frame.DataFrame:
            filename = data
            try:
                data = pd.read_feather(Path(filename))
            except Exception as ex:
                data = pd.read_csv(Path(filename))
        else:
            filename = None

        data[weight_column][data[weight_column] == ''] = 0
        data[weight_column] = data[weight_column].astype(int) + 1

        data['weight_avg'] = data.groupby(by=groupby_columns)[
            weight_column].transform('mean')
        data['candidate_weight'] = data[weight_column].astype(
            int)/data.weight_avg.astype(float)

        data_sample = data.groupby(by=groupby_columns, group_keys=False).apply(
            pd.DataFrame.sample, frac=fraction, weights='candidate_weight')

        if keep_all:
            missing_groups = set(set(data[groupby_columns[0]].tolist(
            ))-set(data_sample[groupby_columns[0]].tolist()))

            data_sample = pd.concat([data_sample, data[data[groupby_columns[0]].isin(
                missing_groups)]], axis=0)

        if drop_weights:
            data_sample.drop(
                ['weight_avg', 'candidate_weight'], inplace=True, axis=1)

        data_sample[weight_column] = data_sample[weight_column].astype(int) - 1

        if select_column:
            data_select = data_sample.groupby(by=groupby_columns)[
                select_column].progress_apply(f'{sep}'.join).reset_index()
            return data_select

        return data_sample


class Summarizer(ModelsAlgorithms):
    """Summarizer summarizes thousands of text documents into a few sentences.

    This module is a subclass of ModelsAlgorithms which initializes folder paths for data reading and
    model output storage.

    Major functions of Summarizer Class are:
    * 1. Generate summary of a single large text document.
    * 2. Generate summary of a batch of many small to medium text documents.

    Args:
        ModelsAlgorithms (object): Parent class that defines folder paths and data locations.

    """

    def __init__(self, current_device: int = -1, initialize_model: bool = False):
        """__init__ Summarizer class instace initializer.

        Initializer has two major functions:
        * 1. Determine whether to run model inference on GPU or CPU.
        * 2. Instantiate the pre-trained Bart language model for summarization task with pretrained weights.
        Args:
            current_device (int, optional): GPU or CPU to run inference. Defaults to -1(CPU).
            initialize_model (bool, optional): Set to True if using method summarize_instance or summarize_batch.
                                               Set to False if using method summarize_batch_plus. Defaults to False.

        """
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_device = 0 if torch.cuda.is_available() else -1

        if initialize_model:
            self.bart_summarizer = pipeline(
                task='summarization', model='bart-large-cnn', device=self.current_device)
        else:
            self.bart_summarizer = None

    def generate_summary(self, text: str, min_length: int = 150) -> str:
        """generate_summary generates summary on a single text document.

        Args:
            text (str): Input text document.
            min_length (int, optional): Minimum length of chracters in generated summary. Defaults to 150.

        Returns:
            str: Generated text summary.

        """
        assert self.bart_summarizer is not None, "Set initialize model parameter to True when using summarize_instance or \
                                                  summarize_batch methods. "

        length = len(text.split())
        if length > 1024:
            max_length = 1024
        else:
            max_length = length

        if length < min_length+30:
            return text
        else:
            summary = self.bart_summarizer(text, min_length=min_length,
                                           max_length=max_length)
            return summary[0]['summary_text']

    def generate_summary_batch(self, examples: list, model_name: str = "bart-large-cnn", min_length: int = 150,
                               max_length: int = 1024, batch_size: int = 12) -> List:
        """generate_summary_batch genereates summary of a batch of text documents at a time.

        Args:
            examples (list): List of text documents to summarize.
            model_name (str, optional): Name of the pre-trained model to use to generate text summaries. Defaults to "bart-large-cnn".
            min_length (int, optional): Minimum length of chracters in generated summary. Defaults to 150.
            max_length (int, optional): Maximum length of chracters in generated summary. Defaults to 1024.
            batch_size (int, optional): Size of the batch to run inference parrallely. Defaults to 12.

        Returns:
            List: List of generated document summaries.

        """
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        generated_summaries = []
        model = BartForConditionalGeneration.from_pretrained(
            model_name).to(self.device)
        tokenizer = BartTokenizer.from_pretrained("bart-large")

        for batch in tqdm(list(chunks(examples, batch_size))):
            dct = tokenizer.batch_encode_plus(
                batch, max_length=1024, return_tensors="pt", pad_to_max_length=True)

            summaries = model.generate(input_ids=dct["input_ids"].to(self.device),
                                       attention_mask=dct["attention_mask"].to(
                self.device),
                num_beams=4,
                length_penalty=2.0,
                # +2 from original because we start at step=1 and stop before max_length
                max_length=max_length + 2,
                min_length=min_length + 1,  # +1 from original because we start at step=1
                no_repeat_ngram_size=3,
                early_stopping=True,
                decoder_start_token_id=model.config.eos_token_id,
            )

            dec = [tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summaries]

            generated_summaries.extend(dec)

        return generated_summaries

    def summarize_instance(self, text: str, min_length: int = 150) -> str:
        """summarize_instance uses generate_summary to summarize one text document.

        text (str): Input text document.
        min_length (int, optional): Minimum length of chracters in generated summary. Defaults to 150.

        Returns:
            str: Generated text summary.

        """
        return self.generate_summary(text, min_length=min_length)

    def summarize_batch(self, data: Union[str, Path, pd.DataFrame], text_column_name: str,
                        min_length: int = 150, save: bool = False) -> pd.DataFrame:
        """summarize_batch uses generate_summary_batch to summarize a huge amount of text documents in batches.

        Args:
            data (Union[str, Path, pd.DataFrame]): Dataframe or file path containing text documents to summarize.
            text_column_name (str): Name of the dataframe column containing text documents.
            min_length (int, optional): Minimum length of chracters in generated summary. Defaults to 150.
            save (bool, optional): Whether to save the summarization results to disk. Defaults to False.. Defaults to False.

        Returns:
            pd.DataFrame: Dataframe containing generated summaries of the text documents.

        """
        if type(data) != pd.core.frame.DataFrame:
            filename = data
            try:
                data = pd.read_feather(Path(data))
            except Exception as ex:
                data = pd.read_csv(Path(data))
        else:
            filename = None

        data.reset_index(inplace=True, drop=True)

        data['summary'] = data[text_column_name].progress_apply(
            lambda x: self.generate_summary(x, min_length=min_length))

        if filename:
            filename = str(filename).split('\\')[-1]
            if save:
                data.to_feather(
                    self.output_path/f'with_summary_{filename}')

        return data

    def summarize_batch_plus(self, data: Union[str, Path, pd.DataFrame], id_column_name: str = 'prod_id', text_column_name: str = 'text',
                             min_length: int = 150, max_length: int = 1024, batch_size: int = 12,
                             summary_column_name: str = 'summary', save=False) -> pd.DataFrame:
        """summarize_batch_plus uses generate_summary_batch to summarize a huge amount of text documents in batches.

        Args:
            data (Union[str, Path, DataFrame]): Dataframe or file path containing text documents to summarize.
            id_column_name (str, optional): Name of column that uniquely identifies a text document in the dataframe.
                                            Defaults to 'prod_id'.
            text_column_name (str, optional): Name of the dataframe column containing text documents. Defaults to 'text'.
            min_length (int, optional): Minimum length of chracters in generated summary. Defaults to 150.
            max_length (int, optional): Maximum length of chracters in generated summary. Defaults to 1024.
            batch_size (int, optional): Size of the batch to run inference parrallely. Defaults to 12.
            summary_column_name (str, optional): Name of the genereated summary column in the pandas dataframe.
                                                 Defaults to 'summary'.
            save (bool, optional): Whether to save the summarization results to disk. Defaults to False.

        Returns:
            pd.DataFrame: Dataframe containing generated summaries of the text documents.

        """
        if type(data) != pd.core.frame.DataFrame:
            filename = data
            try:
                data = pd.read_feather(Path(data))
            except Exception as ex:
                data = pd.read_csv(Path(data))
        else:
            filename = None

        data.reset_index(inplace=True, drop=True)

        data['word_count'] = data[text_column_name].str.split().apply(len)

        data_to_summarize = data[data.word_count >= min_length+100]

        data_not_to_summarize = data[~data[id_column_name].isin(
            data_to_summarize[id_column_name].tolist())]

        data_to_summarize.reset_index(drop=True, inplace=True)
        data_not_to_summarize.reset_index(drop=True, inplace=True)

        summaries = self.generate_summary_batch(
            data_to_summarize[text_column_name].tolist(), min_length=min_length, max_length=max_length, batch_size=batch_size)

        data_to_summarize[summary_column_name] = pd.Series(summaries)

        data_not_to_summarize.rename(
            columns={text_column_name: summary_column_name}, inplace=True)

        self.data_summary = pd.concat([data_to_summarize[[id_column_name, summary_column_name]],
                                       data_not_to_summarize[[id_column_name, summary_column_name]]], axis=0)

        self.data_summary.reset_index(drop=True, inplace=True)

        return self.data_summary


class SexyReview(ModelsAlgorithms):
    """SexyReview utilizes PredictSentiment, PredictInfluence, Keywords, CandidateSelection and Summarizer to extract insights from review data.

    This object is a subclass of ModelsAlgorithms which initializes folder paths for data reading and
    model output storage. It uses all the nlp algorithms in the algorithm module to generate business insights and
    transforms review data to feed into Redshift along with algorithm generated outputs.

    Major functions of SexyReview Class are:
    * 1. Predict sentiment of reviews.
    * 2. Extract keywords from reviews.
    * 3. Predict influence on reviews.
    * 4. Review document candidate selection for summarization tasks.
    * 5. Summarize reviews.
    * 6. Summarize keywords.
    * 7. Find positive and negative talking points.

    Args:
        ModelsAlgorithms (object): Parent class that defines folder paths and data locations.

    """

    def __init__(self, path: Union[Path, str] = Path.cwd(), initialize_sentiment_model: bool = True,
                 initialize_summarizer_model: bool = False) -> None:
        """__init__ SexyReview class instace initializer.

        Depending on the use case the instace constructor either initializes sentiment model or summarizer model.

        Args:
            path (Union[Path, str], optional): path (Union[Path,str], optional): Folder path where the output folder structure will be saved
                                               and data will be read. Defaults to current directory(Path.cwd()).
            initialize_sentiment_model (bool, optional): Initialize sentiment model for classification. Defaults to True.
            initialize_summarizer_model (bool, optional): Initialize pre-trained language model for summarization. Defaults to False.

        """
        super().__init__(path=path)
        if initialize_sentiment_model:
            self.sentiment_model = PredictSentiment(model_file='sentiment_model_two_class',
                                                    data_vocab_file='sentiment_class_databunch_two_class.pkl')
        self.influence_model = PredictInfluence()
        self.keys = KeyWords()
        self.select_ = SelectCandidate()
        if initialize_summarizer_model:
            self.summarizer = Summarizer()

    def make(self, source: str, review_data: Optional[Union[str, Path, pd.DataFrame]] = None,
             text_column_name: str = 'review_text',
             predict_sentiment: bool = True,
             predict_influence: bool = True,
             extract_keywords: bool = True) -> pd.DataFrame:
        """Make performs sentiment and influence classification, keyword extraction and data transformation for Redshift ingestion.

        Args:
            source (str): source code of the metadata and detail files. (Accepted values: [sph, bts])
            review_data (Optional[Union[str, Path, pd.DataFrame]], optional): Review dataframe or data file path. Defaults to None.
            text_column_name (str, optional): Name of the column containing reviews. Defaults to 'review_text'.
            predict_sentiment (bool, optional): Whether to classify sentiment. Defaults to True.
            predict_influence (bool, optional): Whether to classify influence. Defaults to True.
            extract_keywords (bool, optional): Whether to extract keywords. Defaults to True.

        Raises:
            MeiyumeException: Raises exception if source or data files are incorrect.

        Returns:
            pd.DataFrame: Review dataframe with keywords, sentiment labels and influence flag.

        """
        if source not in ['bts', 'sph']:  # replace the list with sql source metadata table read
            raise MeiyumeException(
                "Unable to determine data source. Please provide correct source code.")

        if review_data:
            if not isinstance(review_data, pd.core.frame.DataFrame):
                try:
                    self.review = pd.read_feather(review_data)
                except Exception as ex:
                    self.review = pd.read_csv(review_data)
            else:
                self.review = review_data
        else:
            if source == 'sph':
                review_file_key = [i['Key'] for i in
                                   file_manager.get_matching_s3_keys(
                    prefix='Feeds/BeautyTrendEngine/CleanedData/PreAlgorithm/cleaned_sph_product_review_all')][-1]
                print(review_file_key)
                self.review = file_manager.read_feather_s3(review_file_key)

            elif source == 'bts':
                review_file_key = [i['Key'] for i in
                                   file_manager.get_matching_s3_keys(
                    prefix='Feeds/BeautyTrendEngine/CleanedData/PreAlgorithm/cleaned_bts_product_review_all')][-1]
                self.review = file_manager.read_feather_s3(review_file_key)
            # if source == 'sph':
            #     review_files = self.sph.review_clean_path.glob(
            #         'cleaned_sph_product_review_all*')
            #     self.review = pd.read_feather(
            #         max(review_files, key=os.path.getctime))
            # elif source == 'bts':
            #     meta_files = self.bts.review_clean_path.glob(
            #         'cleaned_bts_product_review_all*')
            #     self.review = pd.read_feather(
            #         max(review_files, key=os.path.getctime))

        self.review.review_text.fillna('', inplace=True)
        self.review = self.review[self.review.review_text != '']
        self.review.reset_index(inplace=True, drop=True)

        if predict_sentiment:
            self.review = self.sentiment_model.predict_batch(
                data=self.review, text_column_name=text_column_name, save=False)

        if predict_influence:
            self.review = self.influence_model.predict_batch(
                data=self.review, text_column_name=text_column_name, save=False)

        if extract_keywords:
            self.review['text'] = self.review.progress_apply(
                lambda x: x.review_title + ". " + x.review_text if x.review_title is not None else x.review_text, axis=1)
            self.review.text = self.review.text.str.lower().str.strip()
            self.review['keywords'] = self.review.text.progress_apply(
                self.keys.extract_keywords)
            # self.review['keywords'] = process_manager.map(
            #     self.keys.extract_keywords, self.review.text)

        filename = f'with_keywords_sentiment_cleaned_{source}_product_review_all_{pd.to_datetime(self.review.meta_date.max()).date()}'

        columns = ["prod_id",
                   "product_name",
                   "recommend",
                   "review_date",
                   "review_rating",
                   "review_text",
                   "review_title",
                   "meta_date",
                   "helpful_n",
                   "helpful_y",
                   "age",
                   "eye_color",
                   "hair_color",
                   "skin_tone",
                   "skin_type",
                   "neg_prob",
                   "pos_prob",
                   "sentiment",
                   "is_influenced",
                   "keywords",
                   "product_variant"
                   ]
        self.review = self.review[columns]

        self.review.to_feather(
            self.output_path/filename)

        self.review.fillna('', inplace=True)
        self.review = self.review.replace('\n', ' ', regex=True)
        self.review = self.review.replace('~', ' ', regex=True)

        filename = filename + '.csv'
        self.review.to_csv(
            self.output_path/filename, index=None, sep='~')
        file_manager.push_file_s3(file_path=self.output_path /
                                  filename, job_name='review')
        Path(self.output_path/filename).unlink()

        return self.review

    def make_summary(self, source: str, review_data: Optional[Union[str, Path, pd.DataFrame]] = None,
                     # candidate_criterion=[],
                     summarize_review: bool = True, summarize_keywords: bool = True,
                     extract_ngrams: bool = True, extract_topic: bool = True) -> pd.DataFrame:
        """make_summary performs summarization of all reviews and keywords of a product.

        Also performs positive/negative keyphrase generation.

        Args:
            source (str): source code of the metadata and detail files. (Accepted values: [sph, bts])
            review_data (Optional[Union[str, Path, pd.DataFrame]], optional): Review dataframe or data file path. Defaults to None.
            summarize_review (bool, optional): Whether to summarize reviews. Defaults to True.
            summarize_keywords (bool, optional): Whether to summarize keywords. Defaults to True.
            extract_ngrams (bool, optional): Whether to extract sentiment ngrams. Defaults to True.
            extract_topic (bool, optional): Whether to extract dominant topics. Defaults to True.

        Raises:
            MeiyumeException: Raises exception if source or data files are incorrect.

        Returns:
            pd.DataFrame: Algorithm output dataframe containing positive/negative review summary,
                          positive/negative keyword summary and positive/negative talking points.

        """

        # replace the list with sql source metadata table read
        if source not in ['bts', 'sph']:
            raise MeiyumeException(
                "Unable to determine data source. Please provide correct source code.")

        if review_data:
            if not isinstance(review_data, pd.core.frame.DataFrame):
                try:
                    self.review = pd.read_feather(review_data)
                except Exception as ex:
                    self.review = pd.read_csv(review_data)
            else:
                self.review = review_data
        else:
            if source == 'sph':
                self.review = db.query_database(
                    "select prod_id, product_name, sentiment, is_influenced, \
                        review_text, review_title, helpful_n, helpful_y, keywords\
                    from r_bte_product_review_f \
                    where prod_id like 'sph%'")
                # review_files = self.output_path.glob(
                #     'with_keywords_sentiment_cleaned_sph_product_review_all_*')
                # rev_li = [pd.read_feather(file) for file in review_files]
                # self.review = pd.concat(rev_li, axis=0, ignore_index=True)
                self.review.drop_duplicates(inplace=True)
                self.review = self.review.drop_duplicates(
                    subset=['prod_id', 'review_text', 'review_date'])
                self.review.reset_index(inplace=True, drop=True)
            elif source == 'bts':
                self.review = db.query_database(
                    "select prod_id, product_name, sentiment, is_influenced, \
                        review_text, review_title, helpful_n, helpful_y, keywords\
                    from r_bte_product_review_f \
                    where prod_id like 'bts%'")
                # review_files = self.output_path.glob(
                #     'with_keywords_sentiment_cleaned_bts_product_review_all_*')
                # rev_li = [pd.read_feather(file) for file in review_files]
                # print()
                # self.review = pd.concat(rev_li, axis=0, ignore_index=True)
                self.review.drop_duplicates(inplace=True)
                self.review = self.review.drop_duplicates(
                    subset=['prod_id', 'review_text', 'review_date'])
                self.review.reset_index(inplace=True, drop=True)

        self.review.meta_date = self.review.meta_date.astype(str)
        meta_date = self.review.meta_date.max()
        print(meta_date)
        self.review = self.review[['prod_id', 'product_name', 'review_text', 'review_title', 'helpful_n',
                                   'helpful_y', 'sentiment', 'is_influenced', 'keywords']]
        self.review = self.review.replace('\n', ' ', regex=True)
        self.review.fillna('', inplace=True)
        self.review = self.review[self.review.review_text != '']
        self.review.reset_index(inplace=True, drop=True)

        self.review['text'] = self.review.progress_apply(
            lambda x: x.review_title + ". " + x.review_text if x.review_title is not None and x.review_title != '' else x.review_text, axis=1)

        self.review.text = self.review.text.str.lower().apply(
            preprocessing.normalize_whitespace)
        # process_manager.map(
        #     preprocessing.normalize_whitespace, self.review.text.str.lower())
        self.review.keywords = self.review.keywords.str.lower()

        pos_review = self.review[self.review.sentiment == 'positive']
        neg_review = self.review[self.review.sentiment == 'negative']

        generated_dataframes = []

        if summarize_keywords:
            pos_kw_selected = self.select_.select(data=pos_review, weight_column='helpful_y', groupby_columns=['prod_id'],
                                                  fraction=0.7, select_column='keywords', sep=', ')
            neg_kw_selected = self.select_.select(data=neg_review, weight_column='helpful_y', groupby_columns=[
                'prod_id'], fraction=0.8, select_column='keywords', sep=', ')

            pos_kw_selected['pos_keyword_summary'] = pos_kw_selected.keywords.progress_apply(lambda x:
                                                                                             self.keys.summarize_keywords(x, max_keys=15))
            neg_kw_selected['neg_keyword_summary'] = neg_kw_selected.keywords.progress_apply(lambda x:
                                                                                             self.keys.summarize_keywords(x, max_keys=15))

            pos_kw_selected.drop(columns='keywords', inplace=True)
            neg_kw_selected.drop(columns='keywords', inplace=True)

            pos_kw_selected.set_index('prod_id', inplace=True)
            neg_kw_selected.set_index('prod_id', inplace=True)

            self.keyword_summary = pos_kw_selected.join(
                neg_kw_selected, how='outer')
            self.keyword_summary.reset_index(inplace=True)

            del pos_kw_selected, neg_kw_selected
            gc.collect()

            generated_dataframes.append(self.keyword_summary)

        if summarize_review or extract_topic:
            pos_review_selected = self.select_.select(data=pos_review, weight_column='helpful_y', groupby_columns=[
                'prod_id'], fraction=0.7, select_column='text', sep=' ')

            neg_review_selected = self.select_.select(data=neg_review, weight_column='helpful_y', groupby_columns=[
                'prod_id'], fraction=0.8, select_column='text', sep=' ')

        if summarize_review:
            pos_review_summary = self.summarizer.summarize_batch_plus(data=pos_review_selected, id_column_name='prod_id',
                                                                      text_column_name='text',
                                                                      min_length=150, max_length=1024, batch_size=10,
                                                                      summary_column_name='pos_review_summary')

            neg_review_summary = self.summarizer.summarize_batch_plus(data=neg_review_selected, id_column_name='prod_id',
                                                                      text_column_name='text',
                                                                      min_length=80, max_length=1024, batch_size=10,
                                                                      summary_column_name='neg_review_summary')
            pos_review_summary.set_index('prod_id', inplace=True)
            neg_review_summary.set_index('prod_id', inplace=True)

            self.review_summary = pos_review_summary.join(
                neg_review_summary, how='outer')
            self.review_summary.reset_index(inplace=True)

            del pos_review_summary, neg_review_summary
            gc.collect()

            generated_dataframes.append(self.review_summary)

        if extract_topic:

            del pos_review_selected, neg_review_selected
            gc.collect()
            pass

        if extract_ngrams:
            pos_ngram_selected = self.select_.select(data=pos_review, weight_column='helpful_y', groupby_columns=[
                'prod_id'], fraction=0.7, select_column='text')

            neg_ngram_selected = self.select_.select(data=neg_review, weight_column='helpful_y', groupby_columns=[
                'prod_id'], fraction=0.8, select_column='text', sep=' ')

            pos_ngram_selected['pos_talking_points'] =\
                pos_ngram_selected.text.progress_apply(lambda x:
                                                       self.keys.generate_sentiment_ngrams(x, min_freq=2, max_terms=15,
                                                                                           sentiment='positive',
                                                                                           sentiment_threshold=0.3,
                                                                                           increase_threshold_by=0.2))

            neg_ngram_selected['neg_talking_points'] =\
                neg_ngram_selected.text.progress_apply(lambda x:
                                                       self.keys.generate_sentiment_ngrams(x, min_freq=2, max_terms=15,
                                                                                           sentiment='negative',
                                                                                           sentiment_threshold=0.0,
                                                                                           increase_threshold_by=0.2))
            pos_ngram_selected.drop(columns='text', inplace=True)
            neg_ngram_selected.drop(columns='text', inplace=True)

            pos_ngram_selected.set_index('prod_id', inplace=True)
            neg_ngram_selected.set_index('prod_id', inplace=True)

            self.sentiment_ngram = pos_ngram_selected.join(
                neg_ngram_selected, how='outer')
            self.sentiment_ngram.reset_index(inplace=True)

            del pos_ngram_selected, neg_ngram_selected
            gc.collect()

            generated_dataframes.append(self.sentiment_ngram)

        self.review_summary_all = reduce(lambda x, y: pd.merge(
            x, y, on='prod_id'), generated_dataframes)

        del generated_dataframes
        gc.collect()

        filename = f'{source}_product_review_summary_all_{meta_date}'

        columns = ['prod_id',
                   'pos_review_summary',
                   'neg_review_summary',
                   'pos_talking_points',
                   'neg_talking_points',
                   'pos_keyword_summary',
                   'neg_keyword_summary'
                   ]

        self.review_summary_all = self.review_summary_all[columns]

        # self.review_summary_all.to_feather(
        #     self.output_path/f'{filename}')
        self.review_summary_all.to_feather(self.output_path/filename)

        self.review_summary_all.fillna('', inplace=True)
        self.review_summary_all.replace(
            '\n', ' ', regex=True, inplace=True)
        self.review_summary_all.replace(
            '~', ' ', regex=True, inplace=True)
        self.review_summary_all.replace(
            '~', '', regex=True, inplace=True)

        filename = filename + '.csv'
        self.review_summary_all.to_csv(
            self.output_path/filename, index=None, sep='~')
        file_manager.push_file_s3(file_path=self.output_path /
                                  filename, job_name='review_summary')
        Path(self.output_path/filename).unlink()
        return self.review_summary_all
