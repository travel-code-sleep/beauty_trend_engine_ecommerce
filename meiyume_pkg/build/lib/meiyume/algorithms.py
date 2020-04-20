from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import warnings
import os
from pathlib import Path
import time
from datetime import datetime, timedelta
from functools import reduce
from collections import Counter
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import swifter
from meiyume.utils import Logger, Sephora, nan_equal, show_missing_value, MeiyumeException, ModelsAlgorithms, S3FileManager

# fast ai imports
from tqdm.notebook import tqdm
from fastai import *
from fastai.text import *

# text lib imports
import re
import string
import unidecode
from ast import literal_eval
import textacy
import textacy.ke as ke
import pke
from nltk.corpus import stopwords
# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy.matcher import Matcher

# spacy.load('en_core_web_lg')
warnings.simplefilter(action='ignore')

file_manager = S3FileManager()


class Ranker(ModelsAlgorithms):
    """Ranker [summary]

    [extended_summary]

    Args:
        ModelsAlgorithms ([type]): [description]

    Returns:
        [type]: [description]
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
                       "skin-care-gift-sets"]

        def choose_type(x):
            x = x.split()
            t = list(set(x) - set(exclude_pdt))
            if len(t) > 0:
                return t[0]
            else:
                return x[0]
        dft.product_type = dft.product_type.swifter.apply(choose_type)

        meta.drop_duplicates(subset='prod_id', inplace=True)

        dft.set_index('prod_id', inplace=True)
        meta.set_index('prod_id', inplace=True)
        meta.drop('product_type', inplace=True, axis=1)

        meta = meta.join(dft, how='left')
        meta.reset_index(inplace=True)

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

        meta_detail.sort_index(inplace=True)

        def total_stars(x): return x.reviews * x.rating

        def bayesian_estimate(x):
            c = round(review_conf['reviews'][(review_conf.category == x.category) & (
                review_conf.product_type == x.product_type)].values[0])
            prior = round(prior_rating['rating'][(prior_rating.category == x.category) & (
                prior_rating.product_type == x.product_type)].values[0])
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

        meta_detail.drop(columns=['meta_datedetail',
                                  'product_namedetail'], axis=1, inplace=True)

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
        meta_detail.product_name = meta_detail.product_name.swifter.apply(
            unidecode.unidecode)
        meta_detail.brand = meta_detail.brand.swifter.apply(
            unidecode.unidecode)

        filename = f'ranked_cleaned_sph_product_meta_detail_all_{meta.meta_date.max()}'
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


class SexyIngredient(ModelsAlgorithms):
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
                   "ban_flag"
                   ]
        self.ingredient = self.ingredient[columns]

        filename = str(filename).split("\\")[-1]
        self.ingredient.to_feather(
            self.output_path/f'ranked_{filename}')

        self.ingredient.fillna('', inplace=True)
        self.ingredient = self.ingredient[self.ingredient.ingredient.str.len(
        ) < 200]
        self.ingredient = self.ingredient.replace('\n', ' ', regex=True)
        self.ingredient = self.ingredient.replace('~', ' ', regex=True)

        filename = 'ranked_' + filename + '.csv'
        self.ingredient.to_csv(self.output_path/filename, index=None, sep='~')

        file_manager.push_file_s3(file_path=self.output_path /
                                  filename, job_name='ingredient')
        Path(self.output_path/filename).unlink()
        return self.ingredient


class KeyWords(ModelsAlgorithms):
    def __init__(self, path='.'):
        super().__init__(path=path)
        self.en = textacy.load_spacy_lang(
            'en', disable=("parser",))

    def get_no_of_words(self, l):
        p = 0.2
        if l < 50:
            k = 2
        elif l >= 50 and l <= 100:
            k = 6
        elif l > 100 and l <= 300:
            k = 15
        elif l > 300 and l <= 1000:
            p = 0.18
            k = 35
        elif l > 1000 and l <= 5000:
            p = 0.16
            k = 60
        elif l > 5000:
            p = 0.14
            k = 100
        return int(round(k)), p

    def extract_keywords(self, text, include_pke=False, is_doc=False):
        try:
            if is_doc:
                doc = text
                l = len(doc.text.split())
            else:
                l = len(text.split())
                doc = None
            if l > 7:
                k, p = self.get_no_of_words(l)

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


class PredictSentiment(ModelsAlgorithms):
    """PredictSentiment [summary]

    [extended_summary]

    Args:
        ModelsAlgorithms ([type]): [description]

    Returns:
        [type]: [description]
    """

    def __init__(self, model_file, data_vocab_file, model_path=None, path='.'):
        """__init__ [summary]

        [extended_summary]

        Args:
            model_file ([type]): [description]
            data_vocab_file ([type]): [description]
            model_path ([type], optional): [description]. Defaults to None.
            path (str, optional): [description]. Defaults to '.'.
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

    def predict_instance(self, text):
        """predict_instance [summary]

        [extended_summary]

        Args:
            text ([type]): [description]

        Returns:
            [type]: [description]
        """
        pred = self.learner.predict(text)
        return pred[0], pred[1].numpy(), pred[2].numpy()

    def predict_batch(self, text_column_name, data, save=False):
        """predict_batch [summary]

        [extended_summary]

        Args:
            text_column_name ([type]): Name of the text field for which sentiment will be predicted.
            data (str:dataframe): Filename of the data file with or without complete path variable.
            save (bool, optional): True if you want to save the resulting dataframe with sentiment and probabilities. Defaults to True.

        Returns:
            [type]: [description]
        """

        if type(data) != pd.core.frame.DataFrame:
            filename = data
            try:
                data = pd.read_feather(Path(data))
            except:
                data = pd.read_csv(Path(data))
        else:
            filename = None

        data.reset_index(inplace=True, drop=True)

        self.learner.data.add_test(data[text_column_name])
        prob_preds = self.learner.get_preds(
            ds_type=DatasetType.Test, ordered=True)

        data = pd.concat([data, pd.DataFrame(
            prob_preds[0].numpy(), columns=['neg_prob', 'pos_prob'])], axis=1)

        data.neg_prob = data.neg_prob.swifter.apply(lambda x: round(x, 3))
        data.pos_prob = data.pos_prob.swifter.apply(lambda x: round(x, 3))

        data['sentiment'] = data.swifter.apply(
            lambda x: 'positive' if x.pos_prob > x.neg_prob else 'negative', axis=1)
        data.reset_index(inplace=True, drop=True)
        if filename:
            filename = str(filename).split('\\')[-1]
            if save:
                data.to_feather(
                    self.output_path/f'with_sentiment_{filename}')
        return data


class PredictInfluence(ModelsAlgorithms):

    def __init__(self):
        super().__init__()
        self.lookup_ = ['free sample', 'free test', 'complimentary test', 'complimentary review', 'complimentary review',
                        'complimentary test', 'receive product free', 'receive product complimentary', 'product complimentary',
                        'free test', 'product free', 'test purpose']
        self.punctuations = string.punctuation
        self.stopwords = list(STOP_WORDS)
        self.parser = English()

    def spacy_tokenizer(self, text):
        """spacy_tokenizer [summary]

        [extended_summary]

        Args:
            text ([type]): [description]

        Returns:
            [type]: [description]
        """
        tokens = self.parser(text)
        tokens = [word.lemma_.lower().strip() if word.lemma_ !=
                  "-PRON-" else word.lower_ for word in tokens]
        tokens = [
            word for word in tokens if word not in self.stopwords and word not in self.punctuations]
        tokens = " ".join([i for i in tokens])
        return tokens

    def predict_instance(self, text):
        tokenized_text = self.spacy_tokenizer(text)
        if any(i in tokenized_text for i in self.lookup_):
            return 'Influenced'
        else:
            return 'Not Influenced'

    def predict_batch(self, text_column_name, data, save=False):
        """predict_batch [summary]

        [extended_summary]

        Args:
            text_column_name ([type]): [description]
            data ([type]): [description]
            save (bool, optional): [description]. Defaults to False.
        """
        if type(data) != pd.core.frame.DataFrame:
            filename = data
            try:
                data = pd.read_feather(Path(data))
            except:
                data = pd.read_csv(Path(data))
        else:
            filename = None

        data['tokenized_text'] = data[text_column_name].swifter.apply(
            self.spacy_tokenizer)
        data['is_influenced'] = data.tokenized_text.swifter.apply(lambda x: "Yes" if
                                                                  any(y in x for y in self.lookup_) else "No")

        data.reset_index(inplace=True, drop=True)
        if filename:
            filename = str(filename).split('\\')[-1]
            if save:
                data.to_feather(
                    self.output_path/f'{filename}')
        return data


class CandidateSelection(ModelsAlgorithms):
    """Review_Summarizer [summary]

    [extended_summary]

    Args:
        ModelsAlgorithms ([type]): [description]
    """

    def __init__(self):
        super().__init__()

    def select(self, data, weight_column, groupby_columns, fraction=0.3, select_column=None,
               drop_weights=True, **kwargs):

        if type(data) != pd.core.frame.DataFrame:
            filename = data
            try:
                data = pd.read_feather(Path(filename))
            except:
                data = pd.read_csv(Path(filename))
        else:
            filename = None

        data[weight_column][data[weight_column] == ''] = 0
        data[weight_column] = data[weight_column].astype(int) + 1

        data['weight_avg'] = data.groupby(by=groupby_columns)[
            weight_column].transform('mean')
        data['candidate_weight'] = data[weight_column].astype(
            int)/data.weight_avg.astype(float)

        data = data.groupby(by=groupby_columns, group_keys=False).apply(
            pd.DataFrame.sample, frac=fraction, weights='candidate_weight')

        data[weight_column] = data[weight_column].astype(int) - 1

        if drop_weights:
            data.drop(['weight_avg', 'candidate_weight'], inplace=True, axis=1)

        if select_column:
            data_select = data.groupby(by=groupby_columns)[
                select_column].progress_apply(' '.join).reset_index()
            return data_select

        return data


class SexyReview(ModelsAlgorithms):
    def __init__(self, path='.'):
        super().__init__(path=path)
        self.sentiment_model = PredictSentiment(model_file='sentiment_model_two_class',
                                                data_vocab_file='sentiment_class_databunch_two_class.pkl')
        self.influence_model = PredictInfluence()
        self.keys = KeyWords()
        self.candidate_selector = CandidateSelection()

    def make(self, review_file, text_column_name='review_text', predict_sentiment=True,
             predict_influence=True, extract_keywords=True):
        """make [summary]

        [extended_summary]

        Args:
            review_file ([type]): [description]
            predict_sentiment (bool, optional): [description]. Defaults to True.
            predict_influence (bool, optional): [description]. Defaults to True.
            extract_keywords (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        if type(review_file) != pd.core.frame.DataFrame:
            filename = review_file
            try:
                self.review = pd.read_feather(Path(review_file))
            except:
                self.review = pd.read_csv(Path(review_file))
        else:
            filename = None
            self.review = review_file

        self.review.review_text = self.review.review_text.str.replace(
            '…read more', '')
        self.review = self.review.replace('\n', ' ', regex=True)
        self.review.reset_index(inplace=True, drop=True)

        if predict_sentiment:
            self.review = self.sentiment_model.predict_batch(
                data=self.review, text_column_name=text_column_name, save=False)

        if predict_influence:
            self.review = self.influence_model.predict_batch(
                data=self.review, text_column_name=text_column_name, save=False)

        if extract_keywords:
            self.review['text'] = self.review.swifter.apply(
                lambda x: x.review_title + ". " + x.review_text if x.review_title is not None else x.review_text, axis=1)
            self.review.text = self.review.text.str.lower().str.strip()
            self.review['keywords'] = self.review.text.swifter.apply(
                self.keys.extract_keywords)

        if filename:
            filename = str(review_file).split('\\')[-1]

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
                       'neg_prob',
                       'pos_prob',
                       'sentiment',
                       'is_influenced',
                       'keywords'
                       ]
            self.review = self.review[columns]

            self.review.to_feather(
                self.output_path/f'with_keywords_sentiment_{filename}')

            self.review.fillna('', inplace=True)
            self.review = self.review.replace('\n', ' ', regex=True)
            self.review = self.review.replace('~', ' ', regex=True)

            filename = 'with_keywords_sentiment_' + filename + '.csv'
            self.review.to_csv(
                self.output_path/filename, index=None, sep='~')
            file_manager.push_file_s3(file_path=self.output_path /
                                      filename, job_name='review')
            Path(self.output_path/filename).unlink()

        return self.review

    def make_summary(self, review_file, text_column_names='text', summarize_review=True,  # candidate_criterion=[],
                     summarize_keywords=True, extract_ngrams=True, extract_topic=True):

        if type(review_file) != pd.core.frame.DataFrame:
            filename = review_file
            try:
                self.review = pd.read_feather(Path(review_file))
            except:
                self.review = pd.read_csv(Path(review_file))
        else:
            filename = None
            self.review = review_file

        self.review = self.review[['prod_id', 'product_name', 'review_text', 'review_title', 'helpful_n',
                                   'helpful_y', 'sentiment', 'is_influenced', 'keywords']]
        self.review = self.review.replace('\n', ' ', regex=True)
        self.review.reset_index(inplace=True, drop=True)

        self.review['text'] = self.review.swifter.apply(
            lambda x: x.review_title + ". " + x.review_text if x.review_title is not None else x.review_text, axis=1)

        pos_review = self.review[self.review.sentiment == 'positive']
        neg_review = self.review[self.review.sentiment == 'negative']

        if summarize_review or extract_topic:
            pos_review_selected = self.candidate_selector.select(data=pos_review, weight_column='helpful_y', groupby_columns=[
                'prod_id'], fraction=0.35, select_column='text')

            neg_review_selected = self.candidate_selector.select(data=neg_review, weight_column='helpful_y', groupby_columns=[
                'prod_id'], fraction=0.55, select_column='text')
