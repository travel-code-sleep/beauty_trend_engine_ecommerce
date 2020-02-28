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
import unidecode


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import swifter
from meiyume.utils import Logger, Sephora, nan_equal, show_missing_value, MeiyumeException, S3FileManager
from tqdm import tqdm

nlp = spacy.load('en_core_web_lg')
file_manager = S3FileManager()

# , category=[FutureWarning, SettingWithCopyWarning])
warnings.simplefilter(action='ignore')
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
        if filename:
            self.data = data
            filename = filename
        else:
            filename = Path(data)
            try:
                self.data = pd.read_feather(data)
            except:
                self.data = pd.read_csv(data)

        data_def = self.find_data_def(str(filename))

        if logs:
            self.cleaner_log = Logger(
                f"sph_prod_{data_def}_cleaning", path=self.clean_log_path)
            self.logger, _ = self.cleaner_log.start_log()

        clean_file_name = 'cleaned_'+str(filename).split('\\')[-1]

        if data_def == 'meta':
            cleaned_metadata = self.meta_cleaner(rtn_typ)
            if save:
                cleaned_metadata.to_feather(
                    self.metadata_clean_path/f'{rtn_typ}_{clean_file_name}')
            return cleaned_metadata
        if data_def == 'detail':
            cleaned_detail = self.detail_cleaner()
            cleaned_detail.drop_duplicates(inplace=True)
            cleaned_detail.reset_index(inplace=True, drop=True)
            if save:
                cleaned_detail.to_feather(
                    self.detail_clean_path/f'{clean_file_name.replace(".csv","")}')
            return cleaned_detail
        if data_def == 'item':
            cleaned_item, cleaned_ingredient = self.item_cleaner()
            cleaned_item.drop_duplicates(inplace=True)
            cleaned_item.reset_index(inplace=True, drop=True)
            if save:
                cleaned_item.to_feather(
                    self.detail_clean_path/f'{clean_file_name.replace(".csv", "")}')
                cleaned_ingredient.to_feather(
                    self.detail_clean_path/f'{clean_file_name.replace("item", "ingredient").replace(".csv","")}')

            cleaned_item.fillna('', inplace=True)
            cleaned_item = cleaned_item.replace('\n', ' ', regex=True)
            cleaned_item = cleaned_item.replace('~', ' ', regex=True)

            cleaned_item.to_csv(
                self.detail_clean_path/clean_file_name, index=None, sep='~')
            file_manager.push_file_s3(file_path=self.detail_clean_path /
                                      clean_file_name, job_name='item')
            Path(self.detail_clean_path/clean_file_name).unlink()
            return cleaned_item, cleaned_ingredient
        if data_def == 'review':
            cleaned_review = self.review_cleaner()
            cleaned_review.drop_duplicates(inplace=True)
            cleaned_review.reset_index(inplace=True, drop=True)
            if save:
                cleaned_review.to_feather(
                    self.review_clean_path/f'{clean_file_name}')

            cleaned_review.fillna('', inplace=True)
            cleaned_review = cleaned_review.replace('\n', ' ', regex=True)
            cleaned_review = cleaned_review.replace('~', ' ', regex=True)

            clean_file_name = clean_file_name+'.csv'
            cleaned_review.to_csv(
                self.review_clean_path/clean_file_name, index=None, sep='~')
            file_manager.push_file_s3(file_path=self.review_clean_path /
                                      clean_file_name, job_name='review')
            Path(self.review_clean_path/clean_file_name).unlink()
            return cleaned_review

    def find_data_def(self, filename):
        filename = str(filename).split('\\')[-1]
        if 'meta' in filename.lower():
            return 'meta'
        elif 'detail' in filename.lower():
            return 'detail'
        elif 'item' in filename.lower():
            return 'item'
        elif 'review' in filename.lower():
            return 'review'
        else:
            raise MeiyumeException(
                "Unable to determine data definition. Please provide correct file names.")

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
        elif x.count('-') > 1 and '/' not in x:
            ts = [m.start() for m in re.finditer(' ', x)]
            p = x[ts[2]:].strip().split('-')
            return p[0], p[1], x[:ts[2]]
        elif '-' in x and x.count('-') < 2 and '/' not in x:
            p = re.split('-', x)
            return p[0], p[1], np.nan
        else:
            return np.nan, np.nan, np.nan

    def clean_price(self, x):
        """[summary]

        Arguments:
            x {[type]} -- [description]
        """
        repls = ('$', ''), ('(', '/ '), (')',
                                         ''), ('value', '')  # , ('\n', '/')
        return reduce(lambda a, kv: a.replace(*kv), repls, x)

    def meta_cleaner(self, rtn_typ):
        """[summary]

        Arguments:
            rtn_typ {[type]} -- [description]
        """
        self.meta = self.data
        self.meta.product_name = self.meta.product_name.swifter.apply(
            unidecode.unidecode)
        self.meta.brand = self.meta.brand.swifter.apply(unidecode.unidecode)

        def fix_multi_low_price(x):
            """[summary]

            Arguments:
                x {[type]} -- [description]
            """
            if len(x) > 7 and ' ' in x:
                p = x.split()
                return p[-1], p[0]
            else:
                return np.nan, np.nan

        # price cleaning
        self.meta['low_p'], self.meta['high_p'], self.meta['mrp'] = zip(
            *self.meta.price.swifter.apply(lambda x: self.clean_price(x)).swifter.apply(lambda y: self.make_price(y)))
        self.meta.drop('price', axis=1, inplace=True)
        self.meta.low_p[self.meta.low_p.swifter.apply(len) > 7], self.meta.mrp[self.meta.low_p.swifter.apply(len) > 7] =\
            zip(*self.meta.low_p[self.meta.low_p.swifter.apply(len)
                                 > 7].swifter.apply(fix_multi_low_price))
        # create product id
        sph_prod_ids = self.meta.product_page.str.split(':', expand=True)
        sph_prod_ids.columns = ['a', 'b', 'id']
        self.meta['prod_id'] = 'sph_' + sph_prod_ids.id
        # clean rating
        clean_rating = re.compile('(\s*)stars|star|No(\s*)')
        self.meta.rating = self.meta.rating.swifter.apply(
            lambda x: clean_rating.sub('', x))
        self.meta.rating[self.meta.rating == ''] = np.nan

        # clean ingredient flag
        clean_prod_type = self.meta.product_type[self.meta.product_type.swifter.apply(
            lambda x: True if x.split('-')[0] == 'clean' else False)].unique()
        self.meta['clean_flag'] = self.meta.swifter.apply(
            lambda x: 'Yes' if x.product_type in clean_prod_type else 'Undefined', axis=1)

        self.meta_no_cat = self.meta.loc[:,
                                         self.meta.columns.difference(['category'])]
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
                if 'K' in x:
                    return int(x.replace('K', ''))*1000
                else:
                    return int(x)
            else:
                return np.nan

        self.detail.votes = self.detail.votes.swifter.apply(
            convert_votes_to_number)

        def split_rating_dist(x):
            if x is not np.nan:
                ratings = literal_eval(x)
                return ratings[1], ratings[3], ratings[5], ratings[7], ratings[9]
            else:
                return (np.nan for i in range(5))

        self.detail['five_star'], self.detail['four_star'], self.detail['three_star'], self.detail['two_star'], self.detail['one_star'] = \
            zip(*self.detail.rating_dist.swifter.apply(split_rating_dist))
        self.detail.drop('rating_dist', axis=1, inplace=True)

        self.detail.would_recommend = self.detail.would_recommend.str.replace(
            '%', '').astype(float)
        self.detail.rename(
            {'would_recommend': 'would_recommend_percentage'}, inplace=True, axis=1)

        self.detail.first_review_date = pd.to_datetime(
            self.detail.first_review_date, infer_datetime_format=True)
        self.detail.meta_date = pd.to_datetime(
            self.detail.meta_date, infer_datetime_format=True)

        return self.detail

    def calculate_ratings(self, x):
        """pass"""
        if x is np.nan:
            return (x['five_star']*5 + x['four_star']*4 + x['three_star']*3 + x['two_star']*2 + x['one_star'])\
                / (x['five_star'] + x['four_star'] + x['three_star'] + x['two_star'] + x['one_star'])
        else:
            return x

    def review_cleaner(self):
        """[summary]

        """
        self.review = self.data
        self.review = self.review[~self.review.review_text.isna()]
        self.review.dropna(subset=['review_text'], axis=0, inplace=True)
        # self.review = self.review[self.review.helpful.apply(type)!= 'int']
        self.review.reset_index(inplace=True, drop=True)

        # separate helpful/not_helpful
        # self.review['helpful_n'], self.review['helpful_y'] = zip(*self.review.helpful.swifter.apply(
        #     lambda x: literal_eval(x)[0] if type(x) != 'int' else '0 \n 0').str.split('\n', expand=True).values)
        self.review['helpful_n'], self.review['helpful_y'] = zip(*self.review.helpful.swifter.apply(
            lambda x: x if type(x) != 'int' else '0 \n 0').str.split('\n', expand=True).values)
        hlp_regex = re.compile('[a-zA-Z()]')
        self.review.helpful_y = self.review.helpful_y.swifter.apply(
            lambda x: hlp_regex.sub('', str(x)))
        self.review.helpful_n = self.review.helpful_n.swifter.apply(
            lambda x: hlp_regex.sub('', str(x)))
        self.review.drop('helpful', inplace=True, axis=1)

        # convert ratings to numbers
        rat_regex = re.compile('(\s*)stars|star|No(\s*)')
        self.review.review_rating = self.review.review_rating.swifter.apply(
            lambda x: rat_regex.sub('', x))
        self.review.review_rating = self.review.review_rating.astype(int)

        # convert data format
        self.review.review_date = pd.to_datetime(
            self.review.review_date, infer_datetime_format=True)

        # clean and convert recommendation
        # if rating is 5 then it is assumed that the person recommends
        # id rating is 1 or 2 then it is assumed that the person does not recommend
        # for all the other cases data is not available
        self.review.recommend[(self.review.recommend == 'Recommends this product') | (
            self.review.review_rating == 5)] = 'Yes'
        self.review.recommend[(self.review.recommend != 'Yes') & (
            self.review.review_rating.isin([1, 2]))] = 'No'
        self.review.recommend[(self.review.recommend != 'Yes') & (
            self.review.review_rating.isin([3, 4]))] = 'not_avlbl'

        # separate and create user attribute column
        def make_dict(x):
            return {k: v for d in literal_eval(x) for k, v in d.items() if k not in ['hair_condition_chemically_treated_(colored,_relaxed,_or']}
        # def create_attributes(attr, x):
        #     if attr == 'age':
        #         if x.get(attr) is not None: return x.get(attr)
        #         elif x.get('age_over') is not None: return x.get('age_over')
        #         else: return np.nan
        #     else:
        #         if x.get(attr) is not None: return x.get(attr)
        #         else: return np.nan

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

        self.review.user_attribute = self.review.user_attribute.swifter.apply(
            make_dict)
        self.review['age'], self.review['eye_color'], self.review['hair_color'], self.review['skin_tone'], self.review['skin_type'] = \
            zip(*self.review.user_attribute.swifter.apply(get_attributes))
        self.review.drop('user_attribute', inplace=True, axis=1)
        # self.review['age'] = self.review.user_attribute.swifter.apply(lambda x: create_attributes('age', x))
        # self.review['eye_color'] = self.review.user_attribute.swifter.apply(lambda x: create_attributes('eye_color', x))
        # self.review['hair_color'] = self.review.user_attribute.swifter.apply(lambda x: create_attributes('hair_color', x))
        # self.review['skin_tone'] = self.review.user_attribute.swifter.apply(lambda x: create_attributes('skin_tone', x))
        # self.review['skin_type'] = self.review.user_attribute.swifter.apply(lambda x: create_attributes('skin_type', x))
        self.review.review_text = self.review.review_text.str.replace(
            '...read more', '')
        self.review.reset_index(drop=True, inplace=True)
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
                   "skin_type"
                   ]
        self.review = self.review[columns]
        return self.review

    def item_cleaner(self):
        """[summary]

        Returns:
            [type] -- [description]
        """
        meta_files = self.metadata_clean_path.glob(
            'cat_cleaned_sph_product_metadata_all*')
        meta = pd.read_feather(max(meta_files, key=os.path.getctime))
        clean_prod_type = meta.product_type[meta.product_type.apply(
            lambda x: True if x.split('-')[0] == 'clean' else False)].unique()
        clean_product_list = meta.prod_id[meta.product_type.isin(
            clean_prod_type)].unique()
        new_product_list = meta.prod_id[meta.new_flag == 'NEW'].unique()

        self.item = self.data

        def get_item_price(x):
            if len(x) == 1:
                return x[0]
            else:
                return min(x)

        self.item.item_price = self.item.item_price.swifter.apply(
            lambda x: self.clean_price(x)).str.replace('/', ' ').str.split()

        self.item.item_price = self.item.item_price.swifter.apply(
            get_item_price)

        self.item.item_size = self.item.item_size.fillna('not_available')
        self.item.item_name = self.item.item_name.str.lower().str.replace(
            'selected', '').str.replace('-', ' ').str.strip()

        def get_item_size_from_item_name(x):
            if x.item_size == 'not_available' and x.item_name is not np.nan:
                if ' oz' in x.item_name or x.item_name.count(' ml') >= 1:
                    return x.item_name
                else:
                    return np.nan
            else:
                return x.item_size
        self.item.item_size = self.item.swifter.apply(
            get_item_size_from_item_name, axis=1)
        # self.item.item_size = self.item.item_size.str.replace('SIZE', '').str.encode(
        #     'ascii', errors='ignore').astype(str).str.decode('utf8', errors='ignore')

        def get_item_size(x):
            if x != 'not_available' and x is not np.nan:
                l = str(x).split('/')
                if len(l) == 1:
                    size_oz, size_ml_gm = l[0], 'not_available'
                else:
                    size_oz, size_ml_gm = l[0], l[1]
                return size_oz, size_ml_gm
            else:
                return 'not_available', 'not_available'

        self.item.item_size = self.item.item_size.str.lower().str.replace(
            'size', '').str.replace('•', '')
        self.item['size_oz'], self.item['size_ml_gm'] = zip(
            *self.item.item_size.swifter.apply(get_item_size))
        self.item.drop('item_size', inplace=True, axis=1)

        self.item.meta_date = pd.to_datetime(
            self.item.meta_date, infer_datetime_format=True)

        #self.item.item_size = self.item.item_size.astype(str).str.decode('utf8', errors='ignore')

        self.item['clean_flag'] = self.item.prod_id.swifter.apply(
            lambda x: 'Clean' if x in clean_product_list else 'No')
        self.item['new_flag'] = self.item.prod_id.swifter.apply(
            lambda x: 'New' if x in new_product_list else 'Old')

        def clean_ing_sep(x):
            if x.clean_flag == 'Clean' and x.item_ingredients is not np.nan:
                return x.item_ingredients.lower().split('clean at sephora')[0]+'\n'
            else:
                return x.item_ingredients

        self.item.item_ingredients = self.item.swifter.apply(lambda x: clean_ing_sep(x), axis=1).str.replace('(and)', ', ').str.replace(';', ', ').str.lower()\
            .replace('may contain', 'xxcont').str.replace('(', '/').str.replace(')', ' ')

        def removeBannedWords(text):
            pattern = re.compile("\\b(off|for|without|defined|which|must|supports|please|protect|prevents|provides|exfoliate|exfoliates|calms|calm|irritating|effects|of|pollution|\
                            on|skins|skin|breakout|prone|helps|help|reduces|reduce|shine|top|yourself|will|namely|between|name|why|amount|comforts|comfort|contour|sunscreens|\
                            so|regarding|from|next|seeming|had|among|seemed|per|beyond|thereafter|because|only|hundred|throughout|never|might|our|sensitized|volumizing|effect|plump|\
                            Strengthen|strengthen|relieve|stresses|stress|Removed|removed|remove|redness|Scaling|derived|Aids|body|renewal|spot|size|clear|prematurely|age|encapsulating|\
                            really|these|whereby|none|last|above|not|always|until|something|prone|became|less|whether|of|everywhere|ca|supports|support|fights|fight|wrinkles|appears|\
                            forty|all|becoming|hereupon|pollution|could|afterwards|twenty|are|keep|though|side|hereafter|unless|may|hereby|gently|effectively|unclog|purify|pores|\
                            nevertheless|whatever|no|since|should|is|again|but|calms|after|re|further|neither|now|some|via|first|else|travel|cortex|nourish|repairs|repair|inside|feel|\
                            anyway|amongst|shine|Provides|out|say|behind|put|much|ours|under|my_new_stopword|who|those|nor|please|few|and|refines|refine|absorbs|absorb|deeper|elasticity|\
                            when|besides|reduce|each|into|do|beforehand|he|also|where|mostly|make|wherein|perhaps|myself|would|breakout|meanwhile|minimize|symptoms|fatigue|insomnia|while|\
                            while|elsewhere|except|together|whereas|how|around|just|three|yourselves|she|your|about|done|therein|an|whereupon|am|providing|protection|chronic|diseases|associated|\
                            somewhere|they|both|bottom|skin|whence|onto|seems|whenever|or|such|almost|see|everyone|my|five|either|go|latterly|Soothe|soften|aging|excessive|sun|exposure|\
                            along|below|another|thence|enough|nt|at|must|by|during|within|hers|whither|upon|has|than|effects|Calms|ve|it|barrier|guards|guard|external|irritants|\
                            anywhere|you|being|us|we|without|himself|therefore|other|towards|thus|up|own|defined|ll|even|any|m|although|designed|battles|battle|environmental|aggressors|\
                            eleven|nobody|anything|whom|breakout-prone|every|moreover|themselves|anyhow|serious|does|whole|Exfoliate|once|others|the|aggressor|skin.|updated|periodically|\
                            third|seem|was|hence|made|sometimes|were|ourselves|then|on|six|across|alone|against|move|back|Helps|in|can|and|signs|aging|minimizes|looks|look|damages|damage|induced|\
                            irritating|what|their|sometime|here|provides|noone|still|indeed|prevents|doing|before|been|somehow|this|several|aid|Increases|circulation|temperature|induces|induce|\
                            for|whoever|former|have|which|become|thereupon|sixty|formerly|quite|using|me|and|get|already|very|well|cannot|work|effectively|utilizing|called|entourage|effect|\
                            many|off|i|nine|otherwise|nowhere|fifty|over|everything|empty|used|with|his|him|most|wherever|more|toward|Please|purifying|sweat|eliminate|toxins|while|soften|tone|\
                            front|itself|same|however|least|often|eight|latter|through|its|anyone|full|if|supports|her|beside|someone|read|list|packaging|sure|resilience|boosts|boost|firmness|\
                            four|be|ever|ten|too|twelve|exfoliate|whose|them|various|rather|thereby|did|down|that|protect|whereafter|yet|lasting|required|fight|age|related|dryness|makes|make|lines|\
                            as|take|nothing|yours|two|herself|there|give|part|becomes|mine|herein|call|due|one|show|fifteen|to|thru|Undisclosed|appropriate|personal|use|\
                            synthetic|Products|formulated|disclosed|meet|following|criteria|include|ingredients|listed|numbers|total|product|ingredient|listings|brightening|\
                            Brand|brand|conducts|testing|ensure|provided|applied|applies|apply|comply|thresholds|threshold|as|follows|follows|meant|rinsed|off|wiped|removed|\
                            for|mean|remain|giving|feeling|a|smooth|pain|inflammation|inflammation|antibacterial|provides|provide|antiseptic|calming|properties|diminish|\
                            appearance|blemishes|prevents|prevent|new|ones|occurring|Contains|contain|benefits|benefit|tone|Protects|protect|hair|refers|refer|shown|data|information|\
                            damage|delivers|deliver|essentials|essential|tame|frizz|balances|balance|hydration|hairs|hair|locks|lock|moisture|delivers|deliver|silky|feel|\
                            hair|moisture|retention|restores|balance|improves|improve|elasticity|hair|Forms|form|protective|film|)\\W", re.I)
            return pattern.sub(" ", text)

        self.item.item_ingredients = self.item.item_ingredients.str.replace('\n', ',').str.replace(
            '%', ' percent ').str.replace('.', ' dottt ').str.replace('/', ' slash ')
        self.item['clean_ing_list'] = self.item.item_ingredients.swifter.apply(lambda x: [" ".join(re.sub(r"[^a-zA-Z0-9%\s,-.]+", '', removeBannedWords(text)).replace('-', ' ').strip().split())
                                                                                          for text in nlp(x.replace('\n', ',')).text.split(',') if removeBannedWords(text).strip() not in ['', ' ']]
                                                                               if x is not np.nan else np.nan, axis=1)

        self.ing = pd.DataFrame(
            columns=['prod_id', 'ingredient', 'clean_flag', 'new_flag'])
        for i in self.item.index:
            clean_list = self.item.loc[i, 'clean_ing_list']
            if clean_list is np.nan:
                continue
            prod_id = self.item.loc[i, 'prod_id']
            clean_flag = self.item.loc[i, 'clean_flag']
            new_flag = self.item.loc[i, 'new_flag']
            df = pd.DataFrame(clean_list, columns=['ingredient'])
            df['prod_id'] = prod_id
            df['clean_flag'] = clean_flag
            df['new_flag'] = new_flag
            self.ing = pd.concat([self.ing, df], axis=0)

        self.ing.drop_duplicates(inplace=True)
        self.ing.ingredient = self.ing.ingredient.str.lower()
        self.ing = self.ing[~self.ing.ingredient.isin(
            ['synthetic fragrances synthetic fragrances 1 synthetic fragrances 1 12 2 synthetic fragrances concentration 1 formula type acrylates ethyl acrylate', '1'])]

        self.ing.ingredient = self.ing.ingredient.str.replace('percent', '% ').str.replace('dottt', '.').str.replace('xxcont', ':may contain ').str.rstrip('.')\
            .str.replace('slash', ' / ').str.lstrip('.')
        self.ing.ingredient = self.ing.ingredient.str.replace(
            'er fruit oil', 'lavender fruit oil')

        bannedwords = pd.read_excel(self.detail_path/'banned_words.xlsx',
                                    sheet_name='banned_words')['words'].str.strip().str.lower().tolist()
        banned_phrases = pd.read_excel(self.detail_path/'banned_phrases.xlsx',
                                       sheet_name='banned_phrases')['phrases'].str.strip().str.lower().tolist()
        i = 0
        while i < 6:
            self.ing.ingredient = self.ing.ingredient.str.lstrip(
                '/').str.rstrip('/').str.lstrip('.').str.rstrip('.').str.strip()
            self.ing.ingredient = self.ing.ingredient.swifter.apply(lambda x: (' ').join(
                [w if w not in bannedwords else ' ' for w in x.split()]).strip())
            self.ing.ingredient = self.ing.ingredient.str.lstrip(
                '/').str.rstrip('/').str.lstrip('.').str.rstrip('.').str.strip()
            self.ing = self.ing[~self.ing.ingredient.isin(banned_phrases)]
            self.ing = self.ing[self.ing.ingredient != '']
            self.ing.ingredient = self.ing.ingredient.str.lstrip(
                '.').str.rstrip('.').str.rstrip('/').str.lstrip('/').str.strip()
            self.ing = self.ing[~self.ing.ingredient.str.isnumeric()]
            self.ing = self.ing[self.ing.ingredient != '']
            i += 1

        self.ing.reset_index(inplace=True, drop=True)

        self.item.drop(columns=['item_ingredients', 'clean_ing_list',
                                'new_flag', 'clean_flag'], inplace=True, axis=1)
        self.item.reset_index(inplace=True, drop=True)

        self.ing['meta_date'] = self.item.meta_date.max()
        columns = ['prod_id',
                   'product_name',
                   'item_name',
                   'item_price',
                   'meta_date',
                   'size_oz',
                   'size_ml_gm']
        self.item = self.item[columns]
        return self.item, self.ing
