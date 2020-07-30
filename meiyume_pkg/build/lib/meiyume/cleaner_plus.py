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
                           ModelsAlgorithms, MeiyumeException, S3FileManager)
# text lib imports
from unidecode import unidecode
import re
import string
import spacy

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
        self.path = Path(path)
        self.sph = Sephora(path=self.path)
        self.bts = Boots(path=self.path)
        self.out = ModelsAlgorithms(path=self.path)

    def clean(self, data: Union[str, Path, pd.DataFrame], save: bool = True,
              logs: bool = False, source: Optional[str] = None, definition: Optional[str] = None) -> pd.DataFrame:
        """clean [summary]

        [extended_summary]

        Args:
            data (Union[str, Path, pd.DataFrame]): [description]
            save (bool, optional): [description]. Defaults to True.
            logs (bool, optional): [description]. Defaults to False.
            source (Optional[str], optional): [description]. Defaults to None.
            definition (Optional[str], optional): [description]. Defaults to None.

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
            self.source, self.definition = source, definition
        else:
            words = filename.split('_')

            self.source, self.definition = words[0], words[2]

            self.clean_file_name = 'cleaned_' + str(filename).split('\\')[-1]
            self.clean_file_name = '_'.join(
                self.clean_file_name.split('_')[:-1]) + f'_{pd.to_datetime(data.meta_date.max()).date()}'
            print(self.clean_file_name)

        if self.source not in ['bts', 'sph'] or self.definition not in ['metadata', 'detail',
                                                                        'item', 'review']:
            raise MeiyumeException(
                "Unable to determine data definition. Please provide correct file name.")

        cleaner_utility = self.get_cleaner_utility()

        cleaned_data = cleaner_utility(data, save)
        return cleaned_data

    def get_cleaner_utility(self) -> Callable[[pd.DataFrame, bool], pd.DataFrame]:
        """get_cleaner_utility chooses the correct cleaning function based on the data definition.

        [extended_summary]

        Raises:
            MeiyumeException: raises exception if incorrect data definition is passed to the utility function.

        Returns:
            Callable[[pd.DataFrame, bool], pd.DataFrame]: the cleaning utility function to clean the data.

        """
        clean_utility_dict = {'metadata': self.metadata_cleaner,
                              'detail': self.detail_cleaner,
                              'item': self.item_cleaner,
                              'review': self.review_cleaner}
        try:
            return clean_utility_dict[str(self.definition)]
        except KeyError:
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
            return price, '', ''

        elif '/' in price and '-' in price:
            p = re.split('-|/', price)
            return p[0], p[1], p[2]

        elif '/' in price and '-' not in price:
            p = re.split('/', price)
            return p[0], '', p[1]

        elif price.count('-') > 1 and '/' not in price:
            ts = [m.start() for m in re.finditer(' ', price)]
            p = price[ts[2]:].strip().split('-')
            return p[0], p[1], price[:ts[2]]

        elif '-' in price and price.count('-') < 2 and '/' not in price:
            p = re.split('-', price)
            return p[0], p[1], ''

        else:
            return '', '', ''

    @staticmethod
    def clean_price(price: str) -> str:
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

    def metadata_cleaner(self, data: pd.DataFrame, save: bool) -> pd.DataFrame:
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

        self.meta[self.meta.columns.difference(['product_name', 'product_page', 'brand'])] \
            = self.meta[self.meta.columns.difference(['product_name', 'product_page', 'brand'])]\
            .apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
        # self.meta.source = self.meta.source.str.lower()
        if self.source == 'bts':
            cat_dict = {"men": ["aftershave", "male grooming tools", "men's toiletries"],
                        "skincare":	["skincare", "suncare", "premium beauty & skincare"],
                        "makeup-tools":	["beauty tools"],
                        "makeup-cosmetics": ["make-up"],
                        "fragrance": ["fragrance gift sets", "perfume", "fragrance offers"],
                        "gifts": ["gifts for her", "gifts for him"],
                        "hair-products":	["hair", "hair styling tools"],
                        "bath-body":	["bathroom essentials", "luxury bath & body"]
                        }
            df = pd.DataFrame.from_dict(cat_dict, orient='index').reset_index()
            df = df.melt(id_vars=["index"]).drop(columns='variable')
            df.columns = ['to_cat', 'from_cat']

            self.meta = self.meta[self.meta.category.isin(['aftershave',
                                                           'bathroom essentials',
                                                           'beauty tools',
                                                           'fragrance gift sets',
                                                           'fragrance offers',
                                                           'gifts for her',
                                                           'gifts for him',
                                                           'hair',
                                                           'hair styling tools',
                                                           'luxury bath & body',
                                                           'make-up',
                                                           'male grooming tools',
                                                           "men's toiletries",
                                                           #    'new in beauty & skincare',
                                                           #    'new in fragrance',
                                                           'perfume',
                                                           'premium beauty & skincare',
                                                           'skincare',
                                                           'suncare'])]

            self.meta.reset_index(inplace=True, drop=True)

            self.meta.category = self.meta.category.apply(
                lambda x: df.to_cat[df.from_cat == x].values[0])

            brand_names = pd.read_csv(
                self.out.external_path/'brand_db.csv').brand_name.tolist()

            self.meta.brand = self.meta.product_name.apply(
                lambda x: [i for i in brand_names if unidecode(i.lower()) in unidecode(x.lower())])
            self.meta.brand = self.meta.brand.apply(
                lambda x: x[0] if len(x) > 0 else '')

        def fix_multi_low_price(x):
            """[summary]

            Arguments:
                x {[type]} -- [description]
            """
            if len(x) > 7 and ' ' in x:
                p = x.split()
                return p[-1], p[0]
            else:
                return '', ''

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
        remove_chars = re.compile('stars|star|no|nan')
        self.meta.rating = self.meta.rating.apply(
            lambda x: remove_chars.sub('', x)).str.strip()
        self.meta.rating[self.meta.rating.isin([' ', ''])] = '0'
        self.meta.rating = self.meta.rating.astype(float)

        # to datetime
        self.meta.meta_date = pd.to_datetime(
            self.meta.meta_date, infer_datetime_format=True)

        # clean ingredient flag
        if self.source == 'sph':
            clean_product_list = self.meta.prod_id[self.meta.product_type.apply(
                lambda x: True if x.split('-')[0] == 'clean' else False)].unique()
            self.meta['clean_flag'] = self.meta.prod_id.apply(
                lambda x: 'clean' if x in clean_product_list else '')
        else:
            self.meta['clean_flag'] = ''

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
                self.meta_no_cat.to_feather(
                    self.sph.detail_crawler_trigger_path/f'no_cat_{self.clean_file_name}')
                self.meta_no_cat.to_feather(
                    self.sph.review_crawler_trigger_path/f'no_cat_{self.clean_file_name}')

            elif self.source == 'bts':
                self.meta.to_feather(
                    self.bts.metadata_clean_path/f'cat_{self.clean_file_name}')
                self.meta_no_cat.to_feather(
                    self.bts.metadata_clean_path/f'no_cat_{self.clean_file_name}')
                self.meta_no_cat.to_feather(
                    self.bts.detail_crawler_trigger_path/f'no_cat_{self.clean_file_name}')
                self.meta_no_cat.to_feather(
                    self.bts.review_crawler_trigger_path/f'no_cat_{self.clean_file_name}')
        return self.meta

    def detail_cleaner(self, data: pd.DataFrame, save: bool) -> pd.DataFrame:
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
        self.detail = self.detail.apply(lambda x: x.str.lower()
                                        if(x.dtype == 'object') else x)

        if self.source == 'sph':
            # convert votes to numbers
            self.detail.votes.fillna('0.0', inplace=True)
            self.detail.votes = self.detail.votes.apply(lambda x: float(x.replace('k', ''))*1000
                                                        if 'k' in x else float(x.replace('m', ''))*1000000)

            # split sephora rating distribution
            def split_rating_dist(x):
                if x is not np.nan:
                    ratings = literal_eval(x)
                    return int(ratings[1]), int(ratings[3]), int(ratings[5]),\
                        int(ratings[7]), int(ratings[9])
                else:
                    return (0 for i in range(5))

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
                self.detail[i][self.detail[i] == ''] = 0.0
                self.detail[i] = self.detail[i].astype(float)
            # create would recommend percentage for boots
            self.detail['would_recommend_percentage'] = 0.0
            # delete it after adding first review data to boots detail
            # self.detail['first_review_date'] = ''
            self.detail['first_review_date'] = ''
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
        self.detail.drop_duplicates(subset='prod_id', inplace=True)
        self.detail.reset_index(drop=True, inplace=True)

        if save:
            if self.source == 'sph':
                self.detail.to_feather(
                    self.sph.detail_clean_path/f'{self.clean_file_name}')  # , index=None)
            elif self.source == 'bts':
                self.detail.to_feather(
                    self.bts.detail_clean_path/f'{self.clean_file_name}')  # , index=None)

        return self.detail

    def item_cleaner(self, data: pd.DataFrame, save: bool) -> pd.DataFrame:
        """item_cleaner [summary]

        [extended_summary]

        Args:
            data (pd.DataFrame): [description]
            save (bool): [description]

        Returns:
            pd.DataFrame: [description]
        """
        nlp = spacy.load('en_core_web_lg')

        if self.source == 'sph':
            meta_files = self.sph.metadata_clean_path.glob(
                'cat_cleaned_sph_product_metadata_all*')
        elif self.source == 'bts':
            meta_files = self.bts.metadata_clean_path.glob(
                'cat_cleaned_bts_product_metadata_all*')

        meta = pd.read_feather(max(meta_files, key=os.path.getctime))

        new_product_list = meta.prod_id[meta.new_flag == 'new'].unique()
        clean_product_list = meta.prod_id[meta.clean_flag == 'clean'].unique()
        vegan_product_list = meta.prod_id[meta.product_type.apply(
            lambda x: True if 'vegan' in x else False)].unique()

        self.item = data
        del data
        gc.collect()

        self.item[self.item.columns.difference(['product_name'])] \
            = self.item[self.item.columns.difference(['product_name'])]\
            .apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

        def get_item_price(x: list) -> float:
            """get_item_price [summary]
            Args:
                x (list): [description]

            Returns:
                float: [description]
            """
            x = [float(i) for i in x]
            if len(x) == 1:
                return x[0]
            elif len(x) == 0:
                return np.nan
            else:
                return min(x)

        self.item = self.item[~self.item.item_price.isna()]
        self.item.reset_index(inplace=True, drop=True)

        self.item.item_price = self.item.item_price.astype(str).apply(
            lambda x: Cleaner.clean_price(x)).str.replace('/', ' ').str.split().apply(
            get_item_price)
        # self.item.item_price = self.item.item_price.apply(
        #     get_item_price)
        if self.source == 'sph':
            def get_item_size_from_item_name(x: str) -> str:
                """get_item_size_from_item_name [summary]

                Args:
                    x (str): [description]

                Returns:
                    str: [description]
                """
                if x.item_size == '' and x.item_name != '':
                    if ' oz' in x.item_name or x.item_name.count(' ml') >= 1 or x.item_name.count(' g') >= 1:
                        return x.item_name
                    else:
                        return np.nan
                else:
                    return x.item_size

            def get_item_size(x: str) -> Tuple[str, str]:
                """get_item_size [summary]

                Args:
                    x (str): [description]

                Returns:
                    Tuple[str, str]: [description]
                """
                if x != '':
                    lst = str(x).split('/')
                    if len(lst) == 1:
                        size_oz, size_ml_gm = lst[0], ''
                    else:
                        size_oz, size_ml_gm = lst[0], lst[1]
                    return size_oz, size_ml_gm
                else:
                    return '', ''

            self.item.item_size = self.item.item_size.fillna('')
            self.item.item_size = self.item.item_size.apply(
                lambda x: x.split('item')[0] if 'item' in x else x)

            self.item.item_name = self.item.item_name.fillna('')
            self.item.item_name = self.item.item_name.str.replace(
                'selected', '').str.replace('-', ' ').str.strip()
            self.item.item_size = self.item.apply(
                get_item_size_from_item_name, axis=1)

            self.item.item_size = self.item.item_size.str.replace(
                'size', '').str.replace('•', '').str.strip()
            self.item['size_oz'], self.item['size_ml_gm'] = zip(
                *self.item.item_size.apply(get_item_size))

            self.item.drop('item_size', inplace=True, axis=1)

        elif self.source == 'bts':
            self.item['size_ml_gm'] = self.item.product_name.apply(
                lambda x: x.split()[-1] if 'ml' in x.lower() or 'gm' in x else '')
            self.item['size_ml_gm'] = self.item['size_ml_gm'].apply(
                lambda x: x if any(char.isdigit() for char in x) else '')
            self.item['size_oz'] = self.item.size_ml_gm.apply(
                lambda x: str(round(float(re.sub("[^0-9]", "", x))*0.033814, 2)) + ' oz' if x != '' else '')

        self.item.meta_date = pd.to_datetime(
            self.item.meta_date, infer_datetime_format=True)

        self.item['clean_flag'] = self.item.prod_id.apply(
            lambda x: 'clean' if x in clean_product_list else '')
        # self.item['new_flag'] = self.item.prod_id.apply(
        #     lambda x: 'New' if x in new_product_list else '')

        def clean_ing_sep(x: str) -> str:
            """clean_ing_sep [summary]

            Args:
                x (str): [description]

            Returns:
                str: [description]
            """
            if x.clean_flag == 'Clean' and x.item_ingredients is not np.nan:
                return x.item_ingredients.split('clean at sephora')[0]+'\n'
            else:
                return x.item_ingredients

        replace_strings_before = (('(and)', ', '), (';', ', '),
                                  ('may contain', 'xxcont'), ('(', '/'),
                                  (')', ' '), ('\n', ','),
                                  ('%', ' percent '), ('.', ' dott '),
                                  ('/', ' slash '), ('\n', ','))
        self.item.item_ingredients = self.item.apply(lambda x: clean_ing_sep(x), axis=1).apply(
            lambda x: reduce(lambda a, kv: a.replace(*kv),
                             replace_strings_before, x)
            if x is not np.nan else np.nan).apply(lambda x: re.sub(r"[^a-zA-Z0-9%\s,-.]+", '', x)
                                                  if x is not np.nan else np.nan)
        self.item['ingredient'] = self.item.item_ingredients.apply(
            lambda x: [text for text in nlp(x).text.split(',')]
            if x is not np.nan else np.nan)

        self.ing = self.item[['prod_id', 'ingredient']]
        self.ing = self.ing.explode('ingredient').drop_duplicates()
        self.ing.dropna(inplace=True)

        if self.source == 'sph':
            self.ing['vegan_flag'] = self.ing.prod_id.apply(
                lambda x: 'vegan' if x in vegan_product_list else '')
        elif self.source == 'bts':
            self.ing['vegan_flag'] = self.ing.ingredient.apply(
                lambda x: 'vegan' if 'vegan' in x else '')

        self.ing['clean_flag'] = self.ing.prod_id.apply(
            lambda x: 'clean' if x in clean_product_list else '')
        self.ing['new_flag'] = self.ing.prod_id.apply(
            lambda x: 'new' if x in new_product_list else '')

        self.ing = self.ing[~self.ing.ingredient.isin(
            ['synthetic fragrances synthetic fragrances 1 synthetic fragrances 1 12 2 \
            synthetic fragrances concentration 1 formula type acrylates ethyl acrylate', '1'])]

        replace_strings_after = (('percent', '% '), ('dott', '.'),
                                 ('xxcont', ':may contain '), ('slash', ' / '),
                                 ('er fruit oil', 'lavender fruit oil')
                                 )
        self.ing.ingredient = self.ing.ingredient.apply(
            lambda x: reduce(lambda a, kv: a.replace(*kv),
                             replace_strings_after, x) if x is not np.nan else np.nan)

        bannedwords = pd.read_excel(self.out.external_path/'banned_words.xlsx',
                                    sheet_name='banned_words')['words'].str.strip().str.lower().tolist()
        banned_phrases = pd.read_excel(self.out.external_path/'banned_phrases.xlsx',
                                       sheet_name='banned_phrases')['phrases'].str.strip().str.lower().tolist()

        strip_strings = ('/', '.', '-', '', ' ')
        i = 0
        while i < 5:
            self.ing.ingredient = self.ing.ingredient.apply(lambda x: (' ').join(
                [w if w not in bannedwords else ' ' for w in x.split()]).strip())
            self.ing.ingredient = self.ing.ingredient.apply(
                lambda x: reduce(lambda a, v: a.strip(v), strip_strings, x))
            self.ing = self.ing[~self.ing.ingredient.isin(banned_phrases)]
            self.ing = self.ing[self.ing.ingredient != '']
            self.ing.ingredient = self.ing.ingredient.apply(
                lambda x: reduce(lambda a, v: a.strip(v), strip_strings, x))
            self.ing = self.ing[~self.ing.ingredient.str.isnumeric()]
            self.ing = self.ing[self.ing.ingredient != '']
            i += 1

        del banned_phrases, bannedwords
        gc.collect()

        self.ing.drop_duplicates(inplace=True)
        self.ing.reset_index(inplace=True, drop=True)
        self.ing['meta_date'] = self.item.meta_date.max()

        self.item.drop(columns=['item_ingredients', 'ingredient', 'clean_flag'],
                       inplace=True, axis=1)
        self.item.drop_duplicates(inplace=True)
        self.item.reset_index(inplace=True, drop=True)
        columns = ['prod_id',
                   'product_name',
                   'item_name',
                   'item_price',
                   'meta_date',
                   'size_oz',
                   'size_ml_gm']
        self.item = self.item[columns]

        if save:
            if self.source == 'sph':
                self.item.to_feather(
                    self.sph.detail_clean_path/f'{self.clean_file_name}')
                self.ing.to_feather(
                    self.sph.detail_clean_path/f'{self.clean_file_name.replace("item", "ingredient")}')
            if self.source == 'bts':
                self.item.to_feather(
                    self.bts.detail_clean_path/f'{self.clean_file_name}')
                self.ing.to_feather(
                    self.bts.detail_clean_path/f'{self.clean_file_name.replace("item", "ingredient")}')

        # Push Item File to S3. No more processing required for Item file.
        self.item.fillna('', inplace=True)
        self.item = self.item.replace('\n', ' ', regex=True)
        self.item = self.item.replace('~', ' ', regex=True)

        self.clean_file_name = self.clean_file_name + '.csv'
        self.item.to_csv(
            self.out.output_path/f'{self.clean_file_name}', index=None, sep='~')
        file_manager.push_file_s3(
            file_path=self.out.output_path/f'{self.clean_file_name}', job_name='item')
        Path(self.out.output_path/f'{self.clean_file_name}').unlink()

        return self.item, self.ing

    def review_cleaner(self, data: pd.DataFrame, save: bool) -> pd.DataFrame:
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
        self.review[self.review.columns.difference(['product_name'])] \
            = self.review[self.review.columns.difference(['product_name'])]\
            .apply(lambda x: x.astype(str).str.lower() if(x.dtype == 'object') else x)

        self.review = self.review[~self.review.review_text.isna()]
        self.review = self.review[self.review.review_text != '']
        self.review = self.review[self.review.review_rating != 'n']
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
            self.review['helpful_n'], self.review['helpful_y'] = zip(
                *self.review.helpful.str.replace(' ', '').str.split('helpful', expand=True).loc[:, 1:2].values)

            hlp_regex = re.compile('[a-zA-Z()]')
            self.review.helpful_y = self.review.helpful_y.apply(
                lambda x: hlp_regex.sub('', str(x)))  # .astype(float)
            self.review.helpful_n = self.review.helpful_n.apply(
                lambda x: hlp_regex.sub('', str(x)))  # .astype(float)

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

        if self.source == 'bts':
            self.review.helpful_n = self.review.helpful_n.replace(
                '', 0).astype(float)
            self.review.helpful_y = self.review.helpful_y.replace(
                '', 0).astype(float)
            self.review['age'], self.review['eye_color'], self.review['hair_color'],\
                self.review['skin_tone'], self.review['skin_type'] = '', '', '', '', ''
        # convert ratings to numbers
        rating_regex = re.compile('stars|star|no|nan')
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
        self.review.recommend[(self.review.recommend.isin(['recommends this product'])) | (
            self.review.review_rating == 5)] = 'yes'
        self.review.recommend[(self.review.recommend != 'yes') & (
            self.review.review_rating.isin([1, 2]))] = 'no'
        self.review.recommend[(self.review.recommend != 'yes') & (
            self.review.review_rating.isin([3, 4]))] = 'not_avlbl'

        self.review.review_text = self.review.review_text.str.replace(
            '...read more', '')
        self.review.review_text = self.review.review_text.str.replace(
            '…read more', '')
        self.review = self.review.replace('\n', ' ', regex=True)
        self.review.drop_duplicates(inplace=True)
        self.review.reset_index(drop=True, inplace=True)

        if save:
            if self.source == 'sph':
                self.review.to_feather(
                    self.sph.review_clean_path/f'{self.clean_file_name}')
            elif self.source == 'bts':
                self.review.to_feather(
                    self.bts.review_clean_path/f'{self.clean_file_name}')

        return self.review
