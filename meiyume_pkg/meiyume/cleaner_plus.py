"""The module to clean and structure unstructured webscraped natural language data."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import gc
import os
import re
import warnings
from ast import literal_eval
# from datetime import datetime, timedelta
from functools import reduce
from pathlib import Path
from typing import *
from typing import Union

import numpy as np

import pandas as pd
import spacy
from tqdm import tqdm
from tqdm.notebook import tqdm

from unidecode import unidecode

from meiyume.utils import (Boots, Logger, MeiyumeException, ModelsAlgorithms,
                           S3FileManager, Sephora)

file_manager = S3FileManager()
tqdm.pandas()
warnings.simplefilter(action='ignore')
np.random.seed(1337)


class Cleaner():
    """Cleaner class uses high performance python following functional programming to clean and structure data at scale."""

    def __init__(self, path: Union[str, Path] = Path.cwd()):
        """__init__ Cleaner class instace initializer.

        Args:
            path (Union[str, Path], optional): Folder path where the cleaned output data structure will be saved
                                               and uncleaned data will be read. Defaults to current directory(Path.cwd()).

        """
        self.path = Path(path)
        self.sph = Sephora(path=self.path)
        self.bts = Boots(path=self.path)
        self.out = ModelsAlgorithms(path=self.path)

    def clean(self, data: Union[str, Path, pd.DataFrame], save: bool = True,
              logs: bool = False, source: Optional[str] = None, definition: Optional[str] = None) -> pd.DataFrame:
        """Clean method takes an uncleaned data file or path, determines the source and applies relevant function to clean webdata.

        Clean method is dependent on four methods to clean specific types of e-commerce webdata:
        1. Metadata cleaner
        2. Detail cleaner
        3. Item cleaner
        4. Review cleaner

        Once the data is cleaned and tranformed to relational structure the data is pushed to S3 storage for further processing
        and insights generation by Algorithms module.

        Args:
            data (Union[str, Path, pd.DataFrame]): Uncleaned data file path or dataframe.
            save (bool, optional): Whether to save the cleaned data to disk. Defaults to True.
            logs (bool, optional): Whether to generate logs during cleaning action. Defaults to False.
            source (Optional[str], optional): The website code from which the data is extracted. Defaults to None.(Current accepted values: [sph, bts])
            definition (Optional[str], optional): The type of data. Defaults to None.(Accepted values: [Metadata, detail, item, review])

        Raises:
            MeiyumeException: Raises exception if source or data files are incorrect.

        Returns:
            pd.DataFrame: Cleaned and structured metadata, detail, item, ingredient and review data.

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

        Raises:
            MeiyumeException: Raises exception if incorrect data definition is passed to the utility function.

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
        """make_price separates cleaned product price data into individual pricing components.

        Args:
            price (str): input price.

        Returns:
            Tuple[str, str, str]: Cleaned and separated small product price, larger product price and mrp.

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
        """clean_price removes unwanted characters from product price data.

        Args:
            price (str): input price
        Returns:
            str: Cleaned price.

        """
        replace_strings = (('$', ''), ('(', '/ '),
                           (')', ''), ('value', ''),
                           ('£', ''), ('nan', ''))

        return reduce(lambda a, kv: a.replace(*kv), replace_strings, price)

    def metadata_cleaner(self, data: pd.DataFrame, save: bool) -> pd.DataFrame:
        """metadata_cleaner cleans e-commerce product metdata.

        Args:
            data (pd.DataFrame): Metadata to clean.
            save (bool): Whether to save cleaned data to disk.

        Returns:
            pd.DataFrame: Cleaned metadata.

        """
        self.meta = data
        del data
        gc.collect()

        self.meta[self.meta.columns.difference(['product_name', 'product_page', 'brand'])] \
            = self.meta[self.meta.columns.difference(['product_name', 'product_page', 'brand'])]\
            .apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
        self.meta.drop_duplicates(inplace=True)

        # self.meta.source = self.meta.source.str.lower()
        if self.source == 'bts':

            self.meta.category[(self.meta.category.isin(['beauty', 'toiletries', ])) &
                               (self.meta.product_type.isin(['conditioner',
                                                             'luxury-beauty-hair',
                                                             'shampoo',
                                                             'hair-value-packs-and-bundles',
                                                             'thinning-hair'
                                                             ]))] = 'hair'
            self.meta.category[(
                self.meta.product_type.isin(['make-up-remover-']))] = 'makeup'
            self.meta.category[(
                self.meta.product_type.isin(['bath-accessories', 'bath-body-gifts-',
                                             'bestsellers-luxury-bath-body',
                                             'body-scrub', 'bubble-bath-oil',
                                             ]))] = 'bath-body'

            cat_dict = {"men": ["mens"],
                        "skincare":	["skincare"],
                        "makeup-tools":	["beauty"],
                        "makeup-cosmetics": ["makeup"],
                        "fragrance": ["fragrance"],
                        "gifts": ["gifts for her", "gifts for him"],
                        "hair-products":	["hair"],
                        "bath-body":	["bathroom essentials", "luxury bath & body", "toiletries", "baby-child"],
                        # "toiletries": []
                        }
            df = pd.DataFrame.from_dict(cat_dict, orient='index').reset_index()
            df = df.melt(id_vars=["index"]).drop(columns='variable')
            df.columns = ['to_cat', 'from_cat']

            self.meta = self.meta[self.meta.category.isin(['baby-child',
                                                           'beauty',
                                                           'fragrance',
                                                           'hair',
                                                           'makeup',
                                                           'mens',
                                                           'skincare',
                                                           'toiletries',
                                                           ])]

            self.meta.reset_index(inplace=True, drop=True)

            self.meta.category = self.meta.category.apply(
                lambda x: df.to_cat[df.from_cat == x].values[0])

            brand_names = pd.read_csv(
                self.out.external_path/'brand_db.csv').brand_name.tolist()

            self.meta.brand = self.meta.product_name.apply(
                lambda x: [i for i in brand_names if unidecode(i.lower()) in unidecode(x.lower())])
            self.meta.brand = self.meta.brand.apply(
                lambda x: x[0] if len(x) > 0 else '')

            self.meta['price1'], self.meta['price2'] = zip(
                *self.meta.price.str.split('|', expand=True).values)

            self.meta.discount = self.meta.discount.map(Cleaner.clean_price)
            self.meta.discount[self.meta.discount == ''] = str(0)

            self.meta.price1 = self.meta.price1.apply(self.clean_price).apply(
                lambda x: x.split()[0]).astype('float')

            self.meta.price2 = self.meta.price2.fillna('')
            self.meta.price2 = self.meta.price2.apply(self.clean_price)
            self.meta.price2 = self.meta.apply(lambda x: str(x.price1) if x.price2 ==
                                               '' else x.price2, axis=1).apply(lambda x: x.split()[0]).astype('float')

            def get_low_high_price(x):
                if x.price1 > x.price2:
                    high_p = x.price1
                    low_p = x.price2
                elif x.price1 == x.price2:
                    high_p = x.price1
                    low_p = x.price1
                else:
                    high_p = x.price2
                    low_p = x.price1
                return low_p, high_p

            self.meta['low_p'], self. meta['high_p'] = zip(
                *self.meta.apply(get_low_high_price, axis=1))

            self.meta['mrp'] = self.meta.high_p.astype(
                float) + self.meta.discount.astype(float)

            self.meta.drop(columns=['discount', 'price',
                                    'price1', 'price2'], inplace=True)

        def fix_multi_low_price(x):
            """Choose correct low price."""
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

            if self.meta.low_p[self.meta.low_p.apply(len) > 7].count() != 0:
                self.meta.low_p[self.meta.low_p.apply(len) > 7], self.meta.mrp[self.meta.low_p.apply(len) > 7] =\
                    zip(*self.meta.low_p[self.meta.low_p.apply(len)
                                         > 7].apply(fix_multi_low_price))

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

        self.meta.drop_duplicates(inplace=True)
        self.meta.reset_index(drop=True, inplace=True)

        if save:
            if self.source == 'sph':
                metadata_filename = self.sph.metadata_clean_path / \
                    f'cat_{self.clean_file_name}'
                self.meta.to_feather(
                    self.sph.metadata_clean_path/f'cat_{self.clean_file_name}')
                self.meta_no_cat.to_feather(
                    self.sph.metadata_clean_path/f'no_cat_{self.clean_file_name}')
                self.meta_no_cat.to_feather(
                    self.sph.detail_crawler_trigger_path/f'no_cat_{self.clean_file_name}')
                self.meta_no_cat.to_feather(
                    self.sph.review_crawler_trigger_path/f'no_cat_{self.clean_file_name}')

            elif self.source == 'bts':
                metadata_filename = self.bts.metadata_clean_path / \
                    f'cat_{self.clean_file_name}'
                self.meta.to_feather(
                    self.bts.metadata_clean_path/f'cat_{self.clean_file_name}')
                self.meta_no_cat.to_feather(
                    self.bts.metadata_clean_path/f'no_cat_{self.clean_file_name}')
                self.meta_no_cat.to_feather(
                    self.bts.detail_crawler_trigger_path/f'no_cat_{self.clean_file_name}')
                self.meta_no_cat.to_feather(
                    self.bts.review_crawler_trigger_path/f'no_cat_{self.clean_file_name}')

            file_manager.push_file_s3(
                file_path=metadata_filename, job_name='cleaned_pre_algorithm')

        return self.meta

    def detail_cleaner(self, data: pd.DataFrame, save: bool) -> pd.DataFrame:
        """detail_cleaner cleans e-commerce product detail data.

        Args:
            data (pd.DataFrame): Detail data file to clean.
            save (bool): Whether to save cleaned data to disk.

        Returns:
            pd.DataFrame: Cleaned detail data.

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
            self.detail.votes = self.detail.votes.apply(lambda x: x.split()[0])
            self.detail.votes = self.detail.votes.apply(lambda x: float(x.replace('k', ''))*1000
                                                        if 'k' in x else float(x.replace('m', ''))*1000000)
            # self.detail.votes = ''

            # # split sephora rating distribution
            # def split_rating_dist(x):
            #     if x is not np.nan:
            #         ratings = literal_eval(x)
            #         return int(ratings[1]), int(ratings[3]), int(ratings[5]),\
            #             int(ratings[7]), int(ratings[9])
            #     else:
            #         return (0 for i in range(5))

            # self.detail['five_star'], self.detail['four_star'], self.detail['three_star'],\
            #     self.detail['two_star'],  self.detail['one_star'] = \
            #     zip(*self.detail.rating_dist.map(split_rating_dist))
            self.detail.drop('rating_dist', axis=1, inplace=True)
            self.detail['five_star'], self.detail['four_star'], self.detail['three_star'],\
                self.detail['two_star'],  self.detail['one_star'] = (
                    '', '', '', '', '')

            # clean sephora would recommend
            self.detail.would_recommend = self.detail.would_recommend.astype(str).str.replace(
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

        if self.source == 'sph':
            detail_filename = self.sph.detail_clean_path / \
                f'{self.clean_file_name}'
        elif self.source == 'bts':
            detail_filename = self.bts.detail_clean_path / \
                f'{self.clean_file_name}'

        if save:
            self.detail.to_feather(detail_filename)  # , index=None)
            file_manager.push_file_s3(
                file_path=detail_filename, job_name='cleaned_pre_algorithm')

        return self.detail

    def item_cleaner(self, data: pd.DataFrame, save: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """item_cleaner cleans e-commerce product item data.


        Item cleaner generates two files, 1. Cleaned Item file and 2. Cleaned Ingredient File.
        Once the cleaned item file is generated it does not require any algorithmic processing
        and is pushed to S3 storage directly for Redshit ingestion.

        Args:
            data (pd.DataFrame): Item data to clean.
            save (bool): Whether to save cleaned data to disk.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Cleaned Item and Ingredient data.

        """
        nlp = spacy.load('en_core_web_lg')

        # if metadata_file:
        #     if not isinstance(metadata_file, pd.core.frame.DataFrame):
        #         try:
        #             meta = pd.read_feather(metadata_file)
        #         except Exception as ex:
        #             meta = pd.read_csv(metadata_file)
        #     else:
        #         meta = metadata_file
        # else:
        #     if self.source == 'sph':
        #         metadata_file = max(self.sph.metadata_clean_path.glob(
        #             'cat_cleaned_sph_product_metadata_all*'), key=os.path.getctime)
        #     elif self.source == 'bts':
        #         metadata_file = max(self.bts.metadata_clean_path.glob(
        #             'cat_cleaned_bts_product_metadata_all*'), key=os.path.getctime)
        #     print(metadata_file)
        #     meta = pd.read_feather(metadata_file)

        # new_product_list = meta.prod_id[meta.new_flag == 'new'].unique()
        # clean_product_list = meta.prod_id[meta.clean_flag == 'clean'].unique()
        # vegan_product_list = meta.prod_id[meta.product_type.apply(
        #     lambda x: True if 'vegan' in x else False)].unique()

        self.item = data
        del data
        gc.collect()

        self.item[self.item.columns.difference(['product_name'])] \
            = self.item[self.item.columns.difference(['product_name'])]\
            .apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

        def get_item_price(x: list) -> float:
            """get_item_price chooses the correct item price if more than one price is mentioned in website.

            Args:
                x (list): List of prices of a product.

            Returns:
                float: The chosen correct price of a product.

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

        if self.source == 'sph':
            self.item.item_price = self.item.item_price.astype(str).apply(
                lambda x: Cleaner.clean_price(x)).str.replace('/', ' ').str.split().apply(
                get_item_price)
        elif self.source == 'bts':
            self.item.item_price = self.item.item_price.astype(str).apply(
                lambda x: self.clean_price(x)).str.replace('/', ' ').str.split().apply(lambda x: x[0]).astype(float)
        # self.item.item_price = self.item.item_price.apply(
        #     get_item_price)
        if self.source == 'sph':
            def get_item_size_from_item_name(x: str) -> str:
                """get_item_size_from_item_name extracts the size of an item if it is mentioned in item name.

                Args:
                    x (str): item name

                Returns:
                    str: Extracted item size.

                """
                if x.item_size == '' and x.item_name != '':
                    if ' oz' in x.item_name or x.item_name.count(' ml') >= 1 or x.item_name.count(' g') >= 1:
                        return x.item_name
                    else:
                        return np.nan
                else:
                    return x.item_size

            def get_item_size(x: str) -> Tuple[str, str]:
                """get_item_size breaks item size data into oz, ml and gm components.

                Args:
                    x (str): Item size.

                Returns:
                    Tuple[str, str]: Size in oz and ml_gm.

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

            self.item.size_oz = self.item.size_oz.str.replace('out of stock:', '',
                                                              regex=True).str.replace(
                'nan', '', regex=True).apply(
                lambda x: x.split('oz')[0]+'oz' if x is not '' and ' oz' in x else x).apply(
                    lambda x: x if ' oz' in x else '')
            self.item.size_ml_gm = self.item.size_ml_gm.apply(
                lambda x: x.split('ml')[0]+'ml' if x is not '' and ' ml' in x else x).apply(
                lambda x: x.split('g')[0]+'g' if x is not '' and ' g' in x else x)

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

        # self.item['clean_flag'] = self.item.prod_id.apply(
        #     lambda x: 'clean' if x in clean_product_list else '')
        # self.item['new_flag'] = self.item.prod_id.apply(
        #     lambda x: 'New' if x in new_product_list else '')

        def clean_ing_sep(x: str) -> str:
            """clean_ing_sep separates unwanted repetitive ingredient information from actual ingredient data.

            Args:
                x (str): Jumbled/messy ingredient data.

            Returns:
                str: Cleaned required ingredients.

            """
            if 'clean at sephora' in x.item_ingredients.lower() or 'formulated without' in x.item_ingredients.lower():
                return x.item_ingredients.lower().split('clean at sephora')[0]+'\n'
            else:
                return x.item_ingredients

        replace_strings_before = (('(and)', ', '), (';', ', '),
                                  ('may contain', 'xxcont'), ('(', '/'),
                                  (')', ' '), ('\n', ','),
                                  ('%', ' percent '), ('.', ' dott '),
                                  ('/', ' slash '), ('\n', ','))
        self.item_ing = self.item.dropna(axis=0, subset=['item_ingredients'])
        # self.item_ing.item_ingredients.dropna(inplace=True)
        self.item_ing.item_ingredients = self.item_ing.apply(lambda x: clean_ing_sep(x), axis=1).apply(
            lambda x: reduce(lambda a, kv: a.replace(*kv),
                             replace_strings_before, x)).apply(lambda x: re.sub(r"[^a-zA-Z0-9%\s,-.]+", '', x))
        self.item_ing['ingredient'] = self.item_ing.item_ingredients.apply(
            lambda x: [text for text in nlp(x).text.split(',')]
            if x is not np.nan else np.nan)

        self.ing = self.item_ing[['prod_id', 'ingredient']]
        self.ing = self.ing.explode('ingredient').drop_duplicates()
        self.ing.dropna(inplace=True)

        del self.item_ing
        gc.collect()

        # if self.source == 'sph':
        #     self.ing['vegan_flag'] = self.ing.prod_id.apply(
        #         lambda x: 'vegan' if x in vegan_product_list else '')
        # elif self.source == 'bts':
        #     self.ing['vegan_flag'] = self.ing.ingredient.apply(
        #         lambda x: 'vegan' if 'vegan' in x else '')

        # self.ing['clean_flag'] = self.ing.prod_id.apply(
        #     lambda x: 'clean' if x in clean_product_list else '')
        # self.ing['new_flag'] = self.ing.prod_id.apply(
        #     lambda x: 'new' if x in new_product_list else '')

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
        while i <= 4:
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

        ing_slash = self.ing[self.ing.ingredient.str.count('/') > 0]
        ing_non_slash = self.ing[self.ing.ingredient.str.count('/') == 0]

        ing_slash.ingredient = ing_slash.ingredient.str.split('/')
        ing_slash = ing_slash.explode('ingredient').drop_duplicates()
        ing_slash.reset_index(inplace=True, drop=True)
        ing_slash.ingredient = ing_slash.ingredient.str.strip()
        ing_slash = ing_slash[ing_slash.ingredient != '']

        self.ing = pd.concat([ing_non_slash, ing_slash],
                             axis=0, ignore_index=True)
        self.ing.ingredient[(self.ing.ingredient.str.contains('%')) &
                            (self.ing.ingredient.str.contains('\.'))] \
            = self.ing.ingredient[(self.ing.ingredient.str.contains('%')) &
                                  (self.ing.ingredient.str.contains('\.'))].str.replace(' \. ', '.').str.replace(' %', '%')

        del ing_slash, ing_non_slash
        gc.collect()

        ing_dot = self.ing[self.ing.ingredient.str.count(' \. ') > 0]
        ing_non_dot = self.ing[self.ing.ingredient.str.count(' \. ') == 0]

        ing_dot.ingredient = ing_dot.ingredient.str.split(' \. ')
        ing_dot = ing_dot.explode('ingredient').drop_duplicates()
        ing_dot.reset_index(inplace=True, drop=True)
        ing_dot.ingredient = ing_dot.ingredient.str.strip()
        ing_dot = ing_dot[ing_dot.ingredient != '']

        self.ing = pd.concat([ing_non_dot, ing_dot], axis=0, ignore_index=True)

        del ing_dot, ing_non_dot
        gc.collect()

        i = 0
        while i <= 3:
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

        self.ing = self.ing[self.ing.ingredient.str.len() > 2]
        self.ing = self.ing[~self.ing.ingredient.apply(lambda x: True if len(x) < 5 and
                                                       any(i in x for i in ['1', '2', '3', '4', '5', '6',
                                                                            '7', '8', '9', '0', '%']) else False)]
        self.ing = self.ing[~self.ing.ingredient.apply(
            lambda x: True if len(x) <= 9 and any(i in x for i in ['.', '%', 'mg', 'ml', 'cm', 'oz', 'gram'])
            and all(i not in x for i in ['aha', 'bha']) else False)]

        self.ing = self.ing[~self.ing.ingredient.str.startswith('000')]

        self.ing.drop_duplicates(inplace=True)
        self.ing.reset_index(inplace=True, drop=True)
        self.ing['meta_date'] = self.item.meta_date.max()

        self.item.drop(columns=['item_ingredients'],
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

        if self.source == 'sph':
            item_filename = self.sph.detail_clean_path / \
                f'{self.clean_file_name}'
            ingredient_filename = self.sph.detail_clean_path / \
                f'{self.clean_file_name.replace("item", "ingredient")}'
        elif self.source == 'bts':
            item_filename = self.bts.detail_clean_path / \
                f'{self.clean_file_name}'
            ingredient_filename = self.bts.detail_clean_path / \
                f'{self.clean_file_name.replace("item", "ingredient")}'

        if save:
            self.item.to_feather(item_filename)
            self.ing.to_feather(ingredient_filename)
            file_manager.push_file_s3(
                file_path=ingredient_filename, job_name='cleaned_pre_algorithm')

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
        """review_cleaner cleans e-commerce product review data.

        Review cleaner creates the user attributes for e-commerce data along with all the cleaning operations and
        data transformations.

        Args:
            data (pd.DataFrame): Metadata to clean.
            save (bool): Whether to save cleaned data to disk.

        Returns:
            pd.DataFrame: Cleaned review data.

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
            # self.review['helpful_n'], self.review['helpful_y'] = zip(
            #     *self.review.helpful.astype(str).str.replace(' ',
            #                                                  '').str.split('helpful',
            #                                                                expand=True).loc[:, 1:2].values)
            self.review['helpful_n'], self.review['helpful_y'] = zip(*self.review.helpful.str.replace(
                '[', '').str.replace(']', '').str.split(',', expand=True).values)
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

                if x.get('eyes') is not None:
                    eye_c = x.get('eyes')
                else:
                    eye_c = np.nan
                if x.get('hair') is not None:  # hair_color
                    hair_c = x.get('hair')
                else:
                    hair_c = np.nan

                if x.get('skin_tone') is not None:
                    skintn = x.get('skin_tone')
                else:
                    skintn = np.nan

                if x.get('skin') is not None:  # skin_type
                    skinty = x.get('skin')
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
        if self.source == 'sph':
            self.review.review_rating = self.review.review_rating.astype(str).apply(
                lambda x: rating_regex.sub('', x)).astype(float)
        # self.review.review_rating = self.review.review_rating.astype(int)
        # convert to pd datetime
        self.review = self.review[~self.review.review_date.isna()]
        self.review = self.review[self.review.review_date != 'none']
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

        if self.source == 'sph':
            review_filename = self.sph.review_clean_path / \
                f'{self.clean_file_name}'
        elif self.source == 'bts':
            review_filename = self.bts.review_clean_path / \
                f'{self.clean_file_name}'

        if save:
            self.review.to_feather(review_filename)
            file_manager.push_file_s3(
                file_path=review_filename, job_name='cleaned_pre_algorithm')

        return self.review
