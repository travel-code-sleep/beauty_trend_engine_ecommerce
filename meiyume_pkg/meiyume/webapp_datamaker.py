"""Datamaker module to read Redshift database, format data as per frontend visualization requirement and push data to webapp S3 storage for use."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import concurrent.futures
import gc
import os
import time
import re
import warnings
from ast import literal_eval
from datetime import datetime, timedelta
from functools import reduce
from pathlib import Path
from ast import literal_eval
from typing import *

import numpy as np
import pandas as pd

from meiyume.utils import (Boots, Logger, MeiyumeException, ModelsAlgorithms,
                           RedShiftReader, S3FileManager, Sephora)

db = RedShiftReader()
file_manager = S3FileManager()

warnings.simplefilter(action='ignore')
np.random.seed(1337)


class RefreshData():
    """RefreshData class contains functions to update datafiles required by webapp basend on schedule."""

    def __init__(self, path='.'):
        """__init__ initializes the class instance.

        The initialization function creates path variables for data storage and later deletion after successfully uploading data to S3 storage.

        Args:
            path (str, optional): Path where the webapp data will be locally dumped before pushing to S3 storage. Defaults to '.'.

        """
        self.path = Path(path)
        # self.sph = Sephora(path=self.path)
        # self.bts = Boots(path=self.path)
        # self.out = ModelsAlgorithms(path=self.path)
        self.dash_data_path = Path(r'D:\Amit\Meiyume\meiyume_data\dash_data')
        self.landing_page_data = {}

    def hasNumbers(self, inputString: str)->bool:
        """hasNumbers checks whether string contains numeric characters.

        Args:
            inputString (str): input text

        Returns:
            bool: True if inputstring contains numeric characters.

        """
        return bool(re.search(r'\d', inputString))

    def refresh_market_trend_data(self, push_file_to_S3: bool, job_name: str)->None:
        """refresh_market_trend_data connects to Redshift tables and updates all data files to reflect latest data on market trend page.

        Args:
            push_file_to_S3 (bool): Whether to transfer updated files to S3 storage.
            job_name (str): Name of the S3 transfer job to select correct data sotrage path on S3.

        """
        # Get Metadata
        metadata = db.query_database(
            "select prod_id, category, product_type, new_flag, meta_date from r_bte_meta_detail_f")
        metadata['source'] = metadata.prod_id.apply(
            lambda x: 'us' if 'sph' in x else 'uk')
        metadata.drop_duplicates(inplace=True)
        metadata.reset_index(inplace=True, drop=True)
        metadata.meta_date = metadata.meta_date.astype('datetime64[M]')

        meta_df = metadata[['prod_id', 'category',
                            'product_type']].drop_duplicates(subset='prod_id')
        # Get Review Data
        reviews = db.query_database(
            "select prod_id, review_date, is_influenced, review_text from r_bte_product_review_f")
        reviews.drop_duplicates(
            subset=['prod_id', 'review_text', 'review_date'], inplace=True)
        reviews.drop(columns='review_text', inplace=True)

        meta_df.set_index('prod_id', inplace=True)
        reviews.set_index('prod_id', inplace=True)

        reviews = reviews.join(meta_df, how='left')
        reviews = reviews[~reviews.category.isna()]
        reviews = reviews[~reviews.review_date.isna()]

        del meta_df
        gc.collect()

        reviews.reset_index(inplace=True)
        reviews['month'] = reviews['review_date'].astype('datetime64[M]')
        reviews['source'] = reviews.prod_id.apply(
            lambda x: 'us' if 'sph' in x else 'uk')
        reviews[reviews.columns.difference(['product_name'])] \
            = reviews[reviews.columns.difference(['product_name'])]\
            .apply(lambda x: x.astype(str).str.lower() if(x.dtype == 'object') else x)
        reviews.drop(columns=['review_date'], inplace=True)

        # Influenced Review Trend
        rev_by_marketing_cate_month = reviews[reviews.is_influenced == 'yes'].groupby(
            by=['source', 'category', 'month']).prod_id.count().reset_index()
        # for col in ['source', 'category']:
        #     rev_by_marketing_cate_month[col] = rev_by_marketing_cate_month[col].astype(
        #         'category')
        rev_by_marketing_cate_month.rename(
            columns={'prod_id': 'review_text'}, inplace=True)
        rev_by_marketing_cate_month.to_feather(
            self.dash_data_path/'review_trend_by_marketing_category_month')

        del rev_by_marketing_cate_month
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'review_trend_by_marketing_category_month',
                job_name=job_name)

        rev_by_marketing_ptype_month = reviews[reviews.is_influenced == 'yes'].groupby(
            by=['source', 'category', 'product_type', 'month']).prod_id.count().reset_index()
        # for col in ['source', 'category', 'product_type']:
        #     rev_by_marketing_ptype_month[col] = rev_by_marketing_ptype_month[col].astype(
        #         'category')
        rev_by_marketing_ptype_month.rename(
            columns={'prod_id': 'review_text'}, inplace=True)
        rev_by_marketing_ptype_month.to_feather(
            self.dash_data_path/'review_trend_by_marketing_product_type_month')

        del rev_by_marketing_ptype_month
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'review_trend_by_marketing_product_type_month',
                job_name=job_name)

        # Review Trend
        rev_by_cate_month = reviews.groupby(
            by=['source', 'category', 'month']).prod_id.count().reset_index()
        # for col in ['source', 'category']:
        #     rev_by_cate_month[col] = rev_by_cate_month[col].astype(
        #         'category')
        rev_by_cate_month.rename(
            columns={'prod_id': 'review_text'}, inplace=True)
        rev_by_cate_month.to_feather(
            self.dash_data_path/'review_trend_category_month')

        del rev_by_cate_month
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'review_trend_category_month',
                job_name=job_name)

        rev_by_ptype_month = reviews.groupby(
            by=['source', 'category', 'product_type', 'month']).prod_id.count().reset_index()
        # for col in ['source', 'category', 'product_type']:
        #     rev_by_ptype_month[col] = rev_by_ptype_month[col].astype(
        #         'category')
        rev_by_ptype_month.rename(
            columns={'prod_id': 'review_text'}, inplace=True)
        rev_by_ptype_month.to_feather(
            self.dash_data_path/'review_trend_product_type_month')

        del rev_by_ptype_month
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'review_trend_product_type_month',
                job_name=job_name)

        # Product Launch Trend
        meta_product_launces_trend_category_month = metadata[metadata.new_flag == 'new'].groupby(
            by=['source', 'category', 'meta_date']).prod_id.count().reset_index()
        meta_product_launces_trend_category_month.rename(
            columns={'prod_id': 'new_product_count'}, inplace=True)
        # for col in ['source', 'category']:
        #     meta_product_launces_trend_category_month[col] = meta_product_launces_trend_category_month[col].astype(
        #         'category')
        meta_product_launces_trend_category_month.to_feather(
            self.dash_data_path/'meta_product_launch_trend_category_month')

        del meta_product_launces_trend_category_month
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'meta_product_launch_trend_category_month',
                job_name=job_name)

        meta_product_launch_trend_product_type_df = metadata[metadata.new_flag ==
                                                             'new'].groupby(by=['source', 'category',
                                                                                'product_type', 'meta_date']).prod_id.count().reset_index()
        meta_product_launch_trend_product_type_df.rename(
            columns={'prod_id': 'new_product_count'}, inplace=True)
        # for col in ['source', 'category', 'product_type']:
        #     meta_product_launch_trend_product_type_df[col] = meta_product_launch_trend_product_type_df[col].astype(
        #         'category')
        meta_product_launch_trend_product_type_df.to_feather(
            self.dash_data_path/'meta_product_launch_trend_product_type_month')

        del meta_product_launch_trend_product_type_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'meta_product_launch_trend_product_type_month',
                job_name=job_name)

        metadata.new_flag[metadata.new_flag == ''] = 'old'
        # this is awesome way to get in-class counts. reuse this everywhere
        product_launch_intensity_category_df = metadata.pivot_table(index=['source', 'category', 'meta_date'],
                                                                    columns='new_flag', aggfunc='size', fill_value=0).reset_index()
        product_launch_intensity_category_df['product_count'] = product_launch_intensity_category_df.new + \
            product_launch_intensity_category_df.old
        product_launch_intensity_category_df['launch_intensity'] = round(
            product_launch_intensity_category_df.new/product_launch_intensity_category_df.product_count, 3)
        # for col in ['source', 'category']:
        #     product_launch_intensity_category_df[col] = product_launch_intensity_category_df[col].astype(
        #         'category')
        product_launch_intensity_category_df.to_feather(
            self.dash_data_path/'meta_product_launch_intensity_category_month')

        del product_launch_intensity_category_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'meta_product_launch_intensity_category_month',
                job_name=job_name)

        # Ingredient Launch Trend
        ingredients = db.query_database(
            "select prod_id, new_flag, meta_date, category, product_type from r_bte_product_ingredients_f")
        ingredients['source'] = ingredients.prod_id.apply(
            lambda x: 'us' if 'sph' in x else 'uk')
        new_ingredient_trend_category_df = ingredients[ingredients.new_flag == 'new_ingredient'].groupby(
            by=['source', 'category', 'meta_date']).new_flag.count().reset_index()
        new_ingredient_trend_category_df.rename(
            columns={'new_flag': 'new_ingredient_count'}, inplace=True)
        # for col in ['source', 'category']:
        #     new_ingredient_trend_category_df[col] = new_ingredient_trend_category_df[col].astype(
        #         'category')
        new_ingredient_trend_category_df.to_feather(
            self.dash_data_path/'new_ingredient_trend_category_month')

        del new_ingredient_trend_category_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'new_ingredient_trend_category_month',
                job_name=job_name)

        new_ingredient_trend_product_type_df = ingredients[ingredients.new_flag == 'new_ingredient'].groupby(
            by=['source', 'category', 'product_type', 'meta_date']).new_flag.count().reset_index()
        new_ingredient_trend_product_type_df.rename(
            columns={'new_flag': 'new_ingredient_count'}, inplace=True)
        # for col in ['source', 'category', 'product_type']:
        #     new_ingredient_trend_product_type_df[col] = new_ingredient_trend_product_type_df[col].astype(
        #         'category')
        new_ingredient_trend_product_type_df.to_feather(
            self.dash_data_path/'new_ingredient_trend_product_type_month')

        del new_ingredient_trend_product_type_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'new_ingredient_trend_product_type_month',
                job_name=job_name)

        del metadata, reviews, ingredients
        gc.collect()

    def refresh_category_page_data(self, push_file_to_S3: bool, job_name: str)->None:
        """refresh_category_page_data connects to Redshift tables and updates all data files to reflect latest data on category insights page.

        Args:
            push_file_to_S3 (bool): Whether to transfer updated files to S3 storage.
            job_name (str): Name of the S3 transfer job to select correct data sotrage path on S3.

        """
        metadata = db.query_database(
            "select prod_id, product_name, brand, category, product_type, new_flag, meta_date, low_p, high_p,\
             mrp, reviews, bayesian_estimate, first_review_date from r_bte_meta_detail_f")

        numeric_cols = ['low_p', 'high_p', 'mrp',
                        'reviews', 'bayesian_estimate']
        metadata[numeric_cols] = metadata[numeric_cols].apply(
            pd.to_numeric, errors='coerce')
        metadata['source'] = metadata.prod_id.apply(
            lambda x: 'us' if 'sph' in x else 'uk')
        metadata['meta_date'] = metadata['meta_date'].astype('datetime64')

        dt = metadata.groupby('source').meta_date.max(
        ).reset_index().to_dict('records')

        meta = []
        for i in dt:
            meta.append(metadata[(metadata.source == i['source']) &
                                 (metadata.meta_date == i['meta_date'])])
        metadata = pd.concat(meta, axis=0)
        metadata.drop_duplicates(subset='prod_id', inplace=True)
        metadata.reset_index(inplace=True, drop=True)
        metadata.high_p = metadata.apply(
            lambda x: x.low_p if x.high_p == 0 else x.high_p, axis=1)
        metadata.mrp = metadata.apply(
            lambda x: x.high_p if x.mrp == 0 else x.mrp, axis=1)

        df_db = db.query_database('select t.*\
                from (select prod_id, review_date,\
                             row_number() over (partition by prod_id order by review_date) as seqnum\
                      from r_bte_product_review_f\
                     ) t\
                where seqnum = 1')
        df_db = db.query_database('select t.*\
                        from (select prod_id, review_date,\
                                    row_number() over (partition by prod_id order by review_date) as seqnum\
                            from r_bte_product_review_f\
                            ) t\
                        where seqnum = 1')
        df_db.rename(
            columns={'review_date': 'first_review_date'}, inplace=True)
        df_db.drop(columns='seqnum', inplace=True)
        df_db = df_db[df_db.prod_id.isin(metadata.prod_id.tolist())]
        metadata.set_index('prod_id', inplace=True)
        df_db.set_index('prod_id', inplace=True)
        metadata.drop(columns='first_review_date', inplace=True)
        metadata = metadata.join(df_db, how='left')
        metadata.reset_index(inplace=True)
        metadata.first_review_date.fillna('',  inplace=True)

        reviews = db.query_database(
            "select prod_id, review_date, is_influenced, review_text, sentiment from r_bte_product_review_f")
        reviews.drop_duplicates(
            subset=['prod_id', 'review_text', 'review_date'], inplace=True)
        reviews.drop(columns='review_text', inplace=True)
        reviews = reviews[reviews.prod_id.isin(metadata.prod_id.tolist())]

        prod_sentiment_count = reviews.groupby(
            by=['prod_id', 'sentiment']).size().unstack(fill_value=0).reset_index()

        metadata.set_index('prod_id', inplace=True)
        prod_sentiment_count.set_index('prod_id', inplace=True)

        metadata = metadata.join(prod_sentiment_count,
                                 on='prod_id', how='left')
        metadata.reset_index(inplace=True)
        metadata.reviews = metadata.positive + metadata.negative
        metadata.rename(columns={'positive': 'positive_reviews',
                                 'negative': 'negative_reviews'}, inplace=True)
        metadata.positive_reviews = round(
            metadata.positive_reviews/metadata.reviews, 2)
        metadata.negative_reviews = round(
            metadata.negative_reviews/metadata.reviews, 2)

        metadata.positive_reviews = metadata.positive_reviews.apply(
            lambda x: str(round(x*100, 2)) + " %")
        metadata.negative_reviews = metadata.negative_reviews.apply(
            lambda x: str(round(x*100, 2)) + " %")

        del reviews, meta, prod_sentiment_count
        gc.collect()

        grouped = metadata.groupby(['source', 'category', 'product_type'])
        # Pricing Analysis Data
        pricing_analytics_df = grouped.agg(min_price=pd.NamedAgg(column='low_p', aggfunc='min'),
                                           max_price=pd.NamedAgg(
                                               column='high_p', aggfunc='max'),
                                           avg_low_price=pd.NamedAgg(
                                               column='low_p', aggfunc='mean'),
                                           avg_high_price=pd.NamedAgg(
                                               column='high_p', aggfunc='mean')
                                           ).reset_index()
        pricing_analytics_df[['min_price', 'max_price',
                              'avg_low_price', 'avg_high_price']] = pricing_analytics_df[[
                                  'min_price', 'max_price',
                                  'avg_low_price', 'avg_high_price']].apply(lambda x: round(x, 2), axis=1)

        # for col in ['source', 'category', 'product_type']:
        #     pricing_analytics_df[col] = pricing_analytics_df[col].astype(
        #         'category')
        pricing_analytics_df.to_feather(
            self.dash_data_path/'category_page_pricing_data')

        del pricing_analytics_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'category_page_pricing_data',
                job_name=job_name)

        # Product Analysis Data
        cat_page_distinct_brands_products_df = grouped.agg(distinct_brands=pd.NamedAgg(column='brand',
                                                                                       aggfunc='nunique'),
                                                           distinct_products=pd.NamedAgg(column='prod_id',
                                                                                         aggfunc='nunique')
                                                           ).reset_index()
        # for col in ['source', 'category', 'product_type']:
        #     cat_page_distinct_brands_products_df[col] = cat_page_distinct_brands_products_df[col].astype(
        #         'category')
        cat_page_distinct_brands_products_df.to_feather(
            self.dash_data_path/'category_page_distinct_brands_products')

        del cat_page_distinct_brands_products_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'category_page_distinct_brands_products',
                job_name=job_name)

        # New Products Data
        cat_page_new_products_df = metadata[metadata.new_flag == 'new'].groupby(
            ['source', 'category', 'product_type']).new_flag.count().reset_index()
        cat_page_new_products_df = cat_page_new_products_df.rename(
            columns={'new_flag': 'new_product_count'})
        # for col in ['source', 'category', 'product_type']:
        #     cat_page_new_products_df[col] = cat_page_new_products_df[col].astype(
        #         'category')
        cat_page_new_products_df.to_feather(
            self.dash_data_path/'category_page_new_products_count')

        del cat_page_new_products_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'category_page_new_products_count',
                job_name=job_name)

        # Product Varieties and Price Data
        items = db.query_database(
            "select prod_id, item_price, size_oz, meta_date from r_bte_product_item_f")

        metadata.set_index('prod_id', inplace=True)
        items.set_index('prod_id', inplace=True)
        items = items.join(metadata[['category', 'product_type']], how='left')

        items.reset_index(inplace=True)
        metadata.reset_index(inplace=True)

        items['source'] = items.prod_id.apply(
            lambda x: 'us' if 'sph' in x else 'uk')

        item = []
        for i in dt:
            item.append(items[(items.source == i['source']) &
                              (items.meta_date == i['meta_date'])])
        items = pd.concat(item, axis=0)
        items.drop_duplicates(subset='prod_id', inplace=True)
        items.product_type.fillna('', inplace=True)
        items = items[items.product_type != '']
        items.reset_index(inplace=True, drop=True)

        df1 = items.groupby(['source', 'category', 'product_type']
                            ).agg(product_variations=pd.NamedAgg(column='prod_id',
                                                                 aggfunc='count')).reset_index()
        df2 = items.drop_duplicates(subset=['item_price', 'prod_id']
                                    ).groupby(['source', 'category', 'product_type']
                                              ).agg(avg_item_price=pd.NamedAgg(column='item_price',
                                                                               aggfunc='mean')).reset_index()
        cat_page_item_variations_price_df = pd.concat(
            [df1, df2[['avg_item_price']]], axis=1)
        cat_page_item_variations_price_df.avg_item_price = cat_page_item_variations_price_df.avg_item_price.apply(
            lambda x: round(x, 2))

        del df1, df2, item
        gc.collect()

        cat_page_item_variations_price_df.avg_item_price = cat_page_item_variations_price_df.avg_item_price.apply(
            lambda x: round(x, 2))
        # for col in ['source', 'category', 'product_type']:
        #     cat_page_item_variations_price_df[col] = cat_page_item_variations_price_df[col].astype(
        #         'category')
        cat_page_item_variations_price_df.to_feather(
            self.dash_data_path/'category_page_item_variations_price')

        del cat_page_item_variations_price_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'category_page_item_variations_price',
                job_name=job_name)

        # Packaging Analysis Data
        cat_page_item_package_oz_df = items.drop_duplicates(subset=['size_oz', 'prod_id']
                                                            ).groupby(
                                                                ['source', 'category',
                                                                    'product_type', 'size_oz']
        ).agg(avg_price=pd.NamedAgg(column='item_price', aggfunc='mean'),
              product_count=pd.NamedAgg(column='prod_id', aggfunc='count')
              ).reset_index()

        cat_page_item_package_oz_df = cat_page_item_package_oz_df[
            cat_page_item_package_oz_df.size_oz != '']

        cat_page_item_package_oz_df.rename(
            columns={'size_oz': 'item_size'}, inplace=True)

        cat_page_item_package_oz_df = cat_page_item_package_oz_df[cat_page_item_package_oz_df.item_size.apply(
            self.hasNumbers)]
        cat_page_item_package_oz_df = cat_page_item_package_oz_df[cat_page_item_package_oz_df.item_size.str.len(
        ) < 30]
        cat_page_item_package_oz_df.item_size = cat_page_item_package_oz_df.item_size.str.replace(
            'out of stock:', '')
        cat_page_item_package_oz_df.reset_index(inplace=True, drop=True)
        cat_page_item_package_oz_df.avg_price = cat_page_item_package_oz_df.avg_price.apply(
            lambda x: round(x, 2))
        # for col in ['source', 'category', 'product_type']:
        #     cat_page_item_package_oz_df[col] = cat_page_item_package_oz_df[col].astype(
        #         'category')
        cat_page_item_package_oz_df.to_feather(
            self.dash_data_path/'category_page_item_package_oz')

        del cat_page_item_package_oz_df, items
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'category_page_item_package_oz',
                job_name=job_name)

        # Top Product Data
        cat_page_top_products_df = metadata.groupby(['source', 'category', 'product_type']
                                                    )['prod_id', 'brand', 'product_name',
                                                      'bayesian_estimate', 'reviews', 'positive_reviews',
                                                      'negative_reviews', 'first_review_date'
                                                      ].apply(
            lambda x: x.nlargest(20, columns=['bayesian_estimate'])
        ).reset_index()

        cat_page_top_products_df.drop(columns=['level_3'], inplace=True)
        cat_page_top_products_df.rename(
            columns={'bayesian_estimate': 'adjusted_rating'}, inplace=True)
        cat_page_top_products_df.adjusted_rating = cat_page_top_products_df.adjusted_rating.apply(
            lambda x: round(x, 3))

        cat_page_top_products_df = cat_page_top_products_df[~cat_page_top_products_df.reviews.isna(
        )]
        cat_page_top_products_df = cat_page_top_products_df[
            cat_page_top_products_df.positive_reviews.str.replace('%', '').str.strip().astype(float) >= 80]
        cat_page_top_products_df = cat_page_top_products_df[cat_page_top_products_df.reviews >= 5]

        cat_page_top_products_df.reset_index(drop=True, inplace=True)
        # cat_page_top_products_df.small_size_price = cat_page_top_products_df.apply(
        #     lambda x: f'${x.small_size_price}'
        #     if x.source == 'us' else f'£{x.small_size_price}', axis=1)
        # cat_page_top_products_df.big_size_price = cat_page_top_products_df.apply(
        #     lambda x: f'${x.big_size_price}'
        #     if x.source == 'us' else f'£{x.big_size_price}', axis=1)
        # for col in ['source', 'category', 'product_type']:
        #     cat_page_top_products_df[col] = cat_page_top_products_df[col].astype(
        #         'category')
        cat_page_top_products_df.to_feather(
            self.dash_data_path/'category_page_top_products')

        del cat_page_top_products_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'category_page_top_products',
                job_name=job_name)

        # New Product Data
        cat_page_new_products_details_df = metadata[metadata.new_flag.str.lower() == 'new'
                                                    ][
            ['source', 'prod_id', 'product_name',
             'brand', 'category', 'product_type',
             'positive_reviews', 'negative_reviews', 'bayesian_estimate', 'reviews',
             'first_review_date'
             ]
        ]
        cat_page_new_products_details_df.rename(
            columns={'bayesian_estimate': 'adjusted_rating'}, inplace=True)
        cat_page_new_products_details_df.adjusted_rating = cat_page_new_products_details_df.adjusted_rating.apply(
            lambda x: round(x, 3))
        cat_page_new_products_details_df.sort_values(
            by='adjusted_rating', inplace=True, ascending=False)
        cat_page_new_products_details_df.reset_index(drop=True, inplace=True)

        # cat_page_new_products_details_df.small_size_price = cat_page_new_products_details_df.apply(
        #     lambda x: f'${x.small_size_price}'
        #     if x.source == 'us' else f'£{x.small_size_price}', axis=1)
        # cat_page_new_products_details_df.big_size_price = cat_page_new_products_details_df.apply(
        #     lambda x: f'${x.big_size_price}'
        #     if x.source == 'us' else f'£{x.big_size_price}', axis=1)

        # for col in ['source', 'category', 'product_type', 'brand']:
        #     cat_page_new_products_details_df[col] = cat_page_new_products_details_df[col].astype(
        #         'category')
        cat_page_new_products_details_df.to_feather(
            self.dash_data_path/'category_page_new_products_details')

        del cat_page_new_products_details_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'category_page_new_products_details',
                job_name=job_name)

        # New Ingredients Data
        ingredients = db.query_database("select prod_id, ingredient, new_flag, ingredient_type, \
            brand, meta_date, category,\
            product_name, product_type, ban_flag, \
            bayesian_estimate from r_bte_product_ingredients_f")
        ingredients['source'] = ingredients.prod_id.apply(
            lambda x: 'us' if 'sph' in x else 'uk')

        ing = []
        for i in dt:
            ing.append(ingredients[(ingredients.source == i['source']) &
                                   (ingredients.meta_date == i['meta_date'])])
        ingredients = pd.concat(ing, axis=0)
        ingredients.drop_duplicates(
            subset=['ingredient', 'prod_id'], inplace=True)
        ingredients.reset_index(inplace=True, drop=True)

        cat_page_new_ingredients_df = ingredients[ingredients.new_flag == 'new_ingredient'][
            ['prod_id', 'source', 'category', 'product_type', 'brand', 'product_name', 'ingredient',
             'ingredient_type', 'bayesian_estimate', 'ban_flag']
        ]
        cat_page_new_ingredients_df.rename(
            columns={'bayesian_estimate': 'adjusted_rating'}, inplace=True)
        cat_page_new_ingredients_df = cat_page_new_ingredients_df[cat_page_new_ingredients_df.ingredient.str.len(
        ) > 7]
        cat_page_new_ingredients_df.reset_index(inplace=True, drop=True)
        # for col in cat_page_new_ingredients_df.columns:
        #     if col not in ['ingredient', 'adjusted_rating', 'ingredient']:
        #         cat_page_new_ingredients_df[col] = cat_page_new_ingredients_df[col].astype(
        #             'category')
        cat_page_new_ingredients_df.to_feather(
            self.dash_data_path/'category_page_new_ingredients')

        del cat_page_new_ingredients_df, ingredients, ing
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'category_page_new_ingredients',
                job_name=job_name)

        # User Attribute Data
        reviews = db.query_database(
            "select prod_id, age, eye_color, hair_color, skin_tone, review_text, skin_type, \
                review_date from  r_bte_product_review_f")
        reviews.drop_duplicates(
            subset=['prod_id', 'review_text', 'review_date'], inplace=True)
        reviews.drop(columns='review_text', inplace=True)
        metadata.set_index('prod_id', inplace=True)
        reviews.set_index('prod_id', inplace=True)
        reviews = reviews.join(
            metadata[['category', 'product_type']], how='left')
        reviews.reset_index(inplace=True)
        metadata.reset_index(inplace=True)
        reviews = reviews[~reviews.category.isna()]
        reviews = reviews[~reviews.review_date.isna()]
        reviews.drop_duplicates(inplace=True)
        reviews['month'] = reviews['review_date'].astype('datetime64[M]')
        reviews['source'] = reviews.prod_id.apply(
            lambda x: 'us' if 'sph' in x else 'uk')
        reviews.replace('none', '', regex=True, inplace=True)
        cat_page_reviews_by_user_attributes = reviews[['prod_id', 'source', 'category',
                                                       'product_type', 'age', 'eye_color',
                                                       'hair_color', 'skin_tone', 'skin_type']
                                                      ][(reviews.age != '') |
                                                        (reviews.eye_color != '') |
                                                        (reviews.hair_color != '') |
                                                        (reviews.skin_tone != '') |
                                                        (reviews.skin_type != '')
                                                        ].reset_index(drop=True)
        cat_page_reviews_by_user_attributes.drop(
            columns='prod_id', inplace=True)
        for col in cat_page_reviews_by_user_attributes.columns:
            cat_page_reviews_by_user_attributes[col] = cat_page_reviews_by_user_attributes[col].astype(
                'category')
        cat_page_reviews_by_user_attributes.to_feather(
            self.dash_data_path/'category_page_reviews_by_user_attributes')

        del cat_page_reviews_by_user_attributes, reviews, metadata
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'category_page_reviews_by_user_attributes',
                job_name=job_name)

    def refresh_product_page_data(self, push_file_to_S3: bool, job_name: str)->None:
        """refresh_product_page_data connects to Redshift tables and updates all data files to reflect latest data on product insights page.

        Args:
            push_file_to_S3 (bool): Whether to transfer updated files to S3 storage.
            job_name (str): Name of the S3 transfer job to select correct data sotrage path on S3.

        """
        # get metadata
        metadata = db.query_database("select prod_id, product_name, brand, category, product_type, new_flag, \
                                     meta_date, low_p, high_p, \
                                     mrp, reviews, bayesian_estimate, first_review_date from r_bte_meta_detail_f")
        metadata['source'] = metadata.prod_id.apply(
            lambda x: 'us' if 'sph' in x else 'uk')
        metadata['meta_date'] = metadata['meta_date'].astype('datetime64')

        # initialize landing page data dict
        self.landing_page_data['products'] = metadata.prod_id.nunique()
        self.landing_page_data['brands'] = metadata.brand.nunique()

        dt = metadata.groupby('source').meta_date.max(
        ).reset_index().to_dict('records')
        meta = []
        for i in dt:
            meta.append(metadata[(metadata.source == i['source']) &
                                 (metadata.meta_date == i['meta_date'])])
        metadata = pd.concat(meta, axis=0)
        metadata.fillna('', inplace=True)
        metadata.drop_duplicates(subset='prod_id', inplace=True)

        self.landing_page_data['latest_scraped_date'] = metadata.meta_date.max(
        )

        metadata[metadata.columns.difference(['product_name', 'brand'])] \
            = metadata[metadata.columns.difference(['product_name', 'brand'])]\
            .apply(lambda x: x.astype(str).str.lower() if(x.dtype == 'object') else x)

        # get review data
        reviews = db.query_database(
            "select prod_id, review_date, sentiment, is_influenced, meta_date, age, eye_color,\
                 review_rating, hair_color, skin_tone, skin_type, review_text from r_bte_product_review_f")
        reviews.drop_duplicates(
            subset=['prod_id', 'review_text', 'review_date'], inplace=True)
        reviews.drop(columns='review_text', inplace=True)

        self.landing_page_data['reviews'] = reviews.shape[0]

        reviews.fillna('', inplace=True)

        first_review_date_df = reviews.groupby(
            'prod_id').review_date.min().reset_index()

        metadata.set_index('prod_id', inplace=True)
        reviews.set_index('prod_id', inplace=True)
        first_review_date_df.set_index('prod_id', inplace=True)

        metadata = metadata.join(first_review_date_df, how='left')
        metadata.first_review_date = metadata.review_date
        metadata.drop(columns='review_date', inplace=True)

        del first_review_date_df
        gc.collect()

        reviews = reviews.join(
            metadata[['source', 'category', 'product_type']], how='left')

        metadata.reset_index(inplace=True)
        reviews.reset_index(inplace=True)

        reviews = reviews[reviews.prod_id.isin(metadata.prod_id.tolist())]
        reviews['review_date'] = reviews['review_date'].astype('datetime64[M]')
        reviews[reviews.columns.difference(['product_name'])] \
            = reviews[reviews.columns.difference(['product_name'])]\
            .apply(lambda x: x.astype(str).str.lower() if(x.dtype == 'object') else x)
        reviews.reset_index(inplace=True, drop=True)

        # get review summary data
        review_sum = db.query_database(
            "select prod_id, pos_review_summary, neg_review_summary, \
                pos_talking_points, neg_talking_points \
                    from r_bte_product_review_summary_f")
        review_sum = review_sum[review_sum.prod_id.isin(
            metadata.prod_id.tolist())]
        review_sum.reset_index(inplace=True, drop=True)

        # get ingredient data
        ingredients = db.query_database("select prod_id, product_name, brand, category, product_type, \
            ingredient, ingredient_type, new_flag, ban_flag, meta_date \
                from r_bte_product_ingredients_f")
        ingredients = ingredients[ingredients.prod_id.isin(
            metadata.prod_id.tolist())]
        ingredients['source'] = ingredients.prod_id.apply(
            lambda x: 'us' if 'sph' in x else 'uk')

        ing = []
        for i in dt:
            ing.append(ingredients[(ingredients.source == i['source']) &
                                   (ingredients.meta_date == i['meta_date'])])
        ingredients = pd.concat(ing, axis=0)
        ingredients.fillna('', inplace=True)
        ingredients.drop_duplicates(
            subset=['prod_id', 'ingredient'], inplace=True)

        ingredients.reset_index(inplace=True, drop=True)
        ingredients[ingredients.columns.difference(['product_name', 'brand'])] \
            = ingredients[ingredients.columns.difference(['product_name', 'brand'])]\
            .apply(lambda x: x.astype(str).str.lower() if(x.dtype == 'object') else x)

        # get item data
        items = db.query_database("select * from r_bte_product_item_f")
        items = items[items.prod_id.isin(metadata.prod_id.tolist())]
        items['meta_date'] = items['meta_date'].astype('datetime64[M]')
        items.fillna('', inplace=True)
        items.reset_index(inplace=True, drop=True)

        # Dropdown and Metadata Data
        prod_page_metadetail_data_df = metadata
        del metadata
        gc.collect()

        prod_page_metadetail_data_df.rename(columns={'low_p': 'small_size_price', 'high_p': 'big_size_price',
                                                     'bayesian_estimate': 'adjusted_rating'}, inplace=True)
        prod_page_metadetail_data_df.adjusted_rating = prod_page_metadetail_data_df.adjusted_rating.apply(
            lambda x: round(float(x), 2) if x != '' else '')
        prod_page_metadetail_data_df = prod_page_metadetail_data_df[
            prod_page_metadetail_data_df.small_size_price != '']
        prod_page_metadetail_data_df.big_size_price = prod_page_metadetail_data_df.apply(
            lambda x: x.small_size_price if x.big_size_price == '' else x.big_size_price, axis=1)
        prod_page_metadetail_data_df.mrp = prod_page_metadetail_data_df.apply(
            lambda x: x.big_size_price if x.mrp == '' else x.mrp, axis=1)
        prod_page_metadetail_data_df.reset_index(inplace=True, drop=True)
        prod_page_metadetail_data_df.small_size_price = prod_page_metadetail_data_df.small_size_price.astype(
            float)
        prod_page_metadetail_data_df.big_size_price = prod_page_metadetail_data_df.big_size_price.astype(
            float)
        prod_page_metadetail_data_df.mrp = prod_page_metadetail_data_df.mrp.astype(
            str)
        prod_page_metadetail_data_df.adjusted_rating = prod_page_metadetail_data_df.adjusted_rating.astype(
            str)
        for col in ['brand', 'category', 'product_type', 'new_flag', 'source']:
            prod_page_metadetail_data_df[col] = prod_page_metadetail_data_df[col].astype(
                'category')
        prod_page_metadetail_data_df.drop_duplicates(inplace=True)
        prod_page_metadetail_data_df.reset_index(drop=True, inplace=True)
        prod_page_metadetail_data_df.to_feather(
            self.dash_data_path/'product_page_metadetail_data')

        del prod_page_metadetail_data_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'product_page_metadetail_data',
                job_name=job_name)

        # Review Tab Data
        review_sum.fillna("{}", inplace=True)
        review_sum.pos_talking_points[review_sum.pos_talking_points == ""] = "{}"
        review_sum.neg_talking_points[review_sum.neg_talking_points == ""] = "{}"
        indices = []
        for i in review_sum.iterrows():
            try:
                d = literal_eval(i[1].pos_talking_points)
                d = literal_eval(i[1].neg_talking_points)
            except Exception as ex:
                indices.append(i[0])
        review_sum = review_sum[~review_sum.index.isin(
            indices)].reset_index(drop=True)
        review_sum.pos_talking_points = review_sum.pos_talking_points.apply(
            lambda x: literal_eval(x) if x != "{}" else {})
        review_sum.neg_talking_points = review_sum.neg_talking_points.apply(
            lambda x: literal_eval(x) if x != "{}" else {})
        # Review Summary Data
        prod_page_review_sum_df = review_sum[[
            'prod_id', 'pos_review_summary', 'neg_review_summary']]

        prod_page_review_sum_df.drop_duplicates(inplace=True)
        prod_page_review_sum_df.reset_index(drop=True, inplace=True)
        prod_page_review_sum_df.to_feather(
            self.dash_data_path/'prod_page_product_review_summary')

        del prod_page_review_sum_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'prod_page_product_review_summary',
                job_name=job_name)

        # Talking Points Data
        prod_page_review_talking_points_df = review_sum[[
            'prod_id', 'pos_talking_points', 'neg_talking_points']]

        prod_page_review_talking_points_df.drop_duplicates(
            inplace=True, subset='prod_id')
        prod_page_review_talking_points_df.reset_index(drop=True, inplace=True)
        prod_page_review_talking_points_df.to_pickle(
            self.dash_data_path/'prod_page_review_talking_points')

        del prod_page_review_talking_points_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'prod_page_review_talking_points',
                job_name=job_name)

        # Review Sentiment and Time Series Data
        reviews.is_influenced = reviews.is_influenced.str.lower()
        prod_page_review_sentiment_influence_df = reviews[[
            'prod_id', 'sentiment', 'is_influenced', 'review_date', 'review_rating']]
        for col in ['prod_id', 'sentiment', 'is_influenced', 'review_rating']:
            prod_page_review_sentiment_influence_df[col] = prod_page_review_sentiment_influence_df[col].astype(
                'category')
        prod_page_review_sentiment_influence_df.review_date = prod_page_review_sentiment_influence_df.review_date.astype(
            'datetime64[M]')

        prod_page_review_sentiment_influence_df.reset_index(
            drop=True, inplace=True)
        prod_page_review_sentiment_influence_df.to_feather(
            self.dash_data_path/'prod_page_review_sentiment_influence')

        del prod_page_review_sentiment_influence_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'prod_page_review_sentiment_influence',
                job_name=job_name)

        # Review User Attribute Data
        prod_page_reviews_attribute_df = reviews[['prod_id', 'age', 'eye_color', 'hair_color', 'skin_tone', 'skin_type']][
            (reviews.age != '') | (reviews.eye_color != '') | (reviews.hair_color != '') | (reviews.skin_tone != '') | (reviews.skin_type != '')]
        for col in prod_page_reviews_attribute_df.columns:
            prod_page_reviews_attribute_df[col] = prod_page_reviews_attribute_df[col].astype(
                'category')
        prod_page_reviews_attribute_df.reset_index(inplace=True, drop=True)
        prod_page_reviews_attribute_df.to_feather(
            self.dash_data_path/'prod_page_reviews_attribute')

        del prod_page_reviews_attribute_df, reviews
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'prod_page_reviews_attribute',
                job_name=job_name)

        # Pricing and Ingredients Tab Data
        prod_page_item_df = items[['prod_id', 'product_name',
                                   'meta_date', 'item_name', 'item_price', 'size_oz']]
        prod_page_item_df.rename(
            columns={'size_oz': 'item_size'}, inplace=True)

        prod_page_item_df.item_size = prod_page_item_df.item_size.apply(
            lambda x: x if self.hasNumbers(x) else '').str.replace('out of stock:', '').str.replace('nan', '')
        prod_page_item_df.item_size = prod_page_item_df.item_size.apply(
            lambda x: x if len(x) < 30 else '')
        prod_page_item_df.reset_index(inplace=True, drop=True)
        for col in ['item_price', 'item_size']:
            prod_page_item_df[col] = prod_page_item_df[col].astype('category')
        prod_page_item_df.drop_duplicates(
            inplace=True, subset=['meta_date', 'prod_id', 'item_size', 'item_name'])
        prod_page_item_df.reset_index(inplace=True, drop=True)
        prod_page_item_df.to_feather(self.dash_data_path/'prod_page_item_data')

        del prod_page_item_df, items
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'prod_page_item_data',
                job_name=job_name)

        # Ingredient Data
        prod_page_ing_df = ingredients[['prod_id', 'product_name', 'ingredient', 'new_flag',
                                        'ingredient_type', 'ban_flag',
                                        'category', 'product_type', 'source']]
        for col in ['prod_id', 'product_name', 'ingredient', 'new_flag',
                    'ingredient_type', 'ban_flag',
                    'category', 'product_type', 'source']:
            prod_page_ing_df[col] = prod_page_ing_df[col].astype('category')
        prod_page_ing_df.to_feather(self.dash_data_path/'prod_page_ing_data')

        del prod_page_ing_df, ingredients
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'prod_page_ing_data',
                job_name=job_name)

    def refresh_ingredient_page_data(self, push_file_to_S3: bool, job_name: str)->None:
        """refresh_ingredient_page_data connects to Redshift tables and updates all data files to reflect latest data on ingredient insights page.

        Args:
            push_file_to_S3 (bool): Whether to transfer updated files to S3 storage.
            job_name (str): Name of the S3 transfer job to select correct data sotrage path on S3.

        """
        # get ingredients
        ing_page_ing_df = db.query_database(
            "select prod_id, ingredient, new_flag, ingredient_type, \
                category, product_name, product_type, vegan_flag, ban_flag \
                from r_bte_product_ingredients_f")
        ing_page_ing_df.fillna('', inplace=True)

        vegan_prods = ing_page_ing_df.prod_id[ing_page_ing_df.vegan_flag == 'vegan'].unique(
        ).tolist()
        ing_page_ing_df.ingredient_type = ing_page_ing_df.apply(
            lambda x: 'vegan' if x.ingredient_type == '' and x.prod_id in vegan_prods else x.ingredient_type, axis=1)
        ing_page_ing_df.drop(columns=['vegan_flag'], inplace=True)

        ing_page_ing_df.drop_duplicates(
            subset=['ingredient', 'prod_id'], inplace=True)
        ing_page_ing_df['source'] = ing_page_ing_df.prod_id.apply(
            lambda x: 'us' if 'sph' in x else 'uk')
        ing_page_ing_df[ing_page_ing_df.columns.difference(['product_name', ])] \
            = ing_page_ing_df[ing_page_ing_df.columns.difference(['product_name', ])]\
            .apply(lambda x: x.astype(str).str.lower() if(x.dtype == 'object') else x)

        self.landing_page_data['ingredients'] = ing_page_ing_df.ingredient.nunique(
        )

        ing_page_ing_df = ing_page_ing_df[ing_page_ing_df.category != '']
        ing_page_ing_df = ing_page_ing_df[ing_page_ing_df.product_type != 'new']

        for col in ['prod_id', 'ingredient', 'new_flag', 'ingredient_type',
                    'category', 'product_name', 'product_type', 'source',
                    'ban_flag']:
            ing_page_ing_df[col] = ing_page_ing_df[col].astype('category')

        ing_page_ing_df.reset_index(inplace=True, drop=True)
        ing_page_ing_df.to_feather(self.dash_data_path/'ing_page_ing_data')

        del ing_page_ing_df
        gc.collect()

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'ing_page_ing_data',
                job_name=job_name)

    def refresh_landing_page_data(self, push_file_to_S3: bool, job_name: str)->None:
        """refresh_landing_page_data connects to Redshift tables and updates all data files to reflect latest data on trend engine landing page.

        Args:
            push_file_to_S3 (bool): Whether to transfer updated files to S3 storage.
            job_name (str): Name of the S3 transfer job to select correct data sotrage path on S3.

        """
        image_keys = [i['Key'] for i in file_manager.get_matching_s3_keys(prefix='Feeds/BeautyTrendEngine/Image/Staging/', suffix='jpg')
                      if any(job in i['Key'].lower() for job in ['image'])]

        self.landing_page_data['images'] = len(image_keys)

        pd.DataFrame(self.landing_page_data, index=range(0, 1)).to_feather(
            self.dash_data_path/'landing_page_data')

        if push_file_to_S3:
            file_manager.push_file_s3(
                file_path=self.dash_data_path/'landing_page_data',
                job_name=job_name)

    def make(self, push_file_to_S3: bool = True, job_name: str = 'webapp'):
        """make runs the data updater functions to pull latest data from Redshift DB and create updated webapp data files.

        The updated datafiles overwrites the existing data in the S3 storage path and always reflects latest data on webpages.

        Args:
            push_file_to_S3 (bool, optional): Whether to transfer updated files to S3 storage. Defaults to True.
            job_name (str, optional): Name of the S3 transfer job to select correct data sotrage path on S3. Defaults to 'webapp'.

        """
        try:
            self.refresh_market_trend_data(push_file_to_S3, job_name)
            self.refresh_category_page_data(push_file_to_S3, job_name)
            self.refresh_product_page_data(push_file_to_S3, job_name)
            self.refresh_ingredient_page_data(push_file_to_S3, job_name)
            self.refresh_landing_page_data(push_file_to_S3, job_name)
        except Exception as ex:
            print('refresh job failed.')
        else:
            print('refresh job completed.')
