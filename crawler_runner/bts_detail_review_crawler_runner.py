""" This script runs spider to grab sephora review data."""
import gc
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from meiyume.bts.crawler import DetailReview
from meiyume.utils import RedShiftReader, chunks, ranges
from meiyume.algorithms import SexyMetaDetail, SexyIngredient

warnings.simplefilter(action='ignore')
db = RedShiftReader()

open_with_proxy_server = True


def get_metadata_with_last_scraped_review_date(meta_df: pd.DataFrame) -> pd.DataFrame:
    """get_metadata_with_last_scraped_review_date [summary]

    [extended_summary]

    Args:
        meta_df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    df = db.query_database("with cte as (select row_number() over (partition by prod_id\
                                 order by review_date desc) as rn,\
                                    prod_id,\
                                    review_date\
                               from r_bte_product_review_f\
                               where prod_id like 'bts%')\
                                select prod_id, review_date\
                                  from cte\
                                 where rn=1")
    df.columns = ['prod_id', 'last_scraped_review_date']
    df.set_index('prod_id', inplace=True)

    meta_df.set_index('prod_id', inplace=True)
    meta_df = meta_df.join(df, how='left')
    meta_df.reset_index(inplace=True)
    meta_df = meta_df.sample(frac=1).reset_index(drop=True)
    return meta_df


def exclude_scraped_products_from_tracker(bts_crawler: DetailReview, reset_na: bool = False) -> pd.DataFrame:
    """exclude_scraped_products_from_tracker [summary]

    [extended_summary]

    Args:
        bts_crawler (DetailReview): [description]
        reset_na (bool, optional): [description]. Defaults to False.

    Returns:
        pd.DataFrame: [description]
    """
    progress_tracker = pd.read_csv(
        bts_crawler.path/'boots/bts_detail_review_image_progress_tracker.csv')

    if reset_na:
        progress_tracker.scraped[progress_tracker.scraped == 'NA'] = 'N'

    progress_tracker = progress_tracker[~progress_tracker.scraped.isna(
    )]
    progress_tracker = progress_tracker[progress_tracker.scraped != 'Y']
    progress_tracker = progress_tracker.sample(frac=1).reset_index(drop=True)
    progress_tracker.to_csv(
        bts_crawler.path/'boots/bts_detail_review_image_progress_tracker.csv', index=None)
    return progress_tracker


def run_bts_crawler(meta_df: pd.DataFrame, bts_crawler: DetailReview):
    """run_bts_crawler [summary]

    [extended_summary]

    Args:
        meta_df (pd.DataFrame): [description]
        bts_crawler (DetailReview): [description]
    """
    for i in ranges(meta_df.shape[0], 30):
        print(i[0], i[-1])
        if i[0] == 0:
            fresh_start = True
            auto_fresh_start = True
        else:
            fresh_start = False
            auto_fresh_start = False
        bts_crawler.extract(metadata=meta_df, download=True, incremental=True, n_workers=4,
                            fresh_start=fresh_start, auto_fresh_start=auto_fresh_start,
                            start_idx=i[0], end_idx=i[-1],  # list_of_index=i,
                            open_headless=False, open_with_proxy_server=open_with_proxy_server, randomize_proxy_usage=True,
                            compile_progress_files=False, clean=False, delete_progress=False)

        bts_crawler.terminate_logging()
        del bts_crawler
        gc.collect()

        time.sleep(5)
        bts_crawler = DetailReview(
            path="D:/Amit/Meiyume/meiyume_data/spider_runner")

    progress_tracker = exclude_scraped_products_from_tracker(
        bts_crawler, reset_na=True)
    n_workers = 4
    trials = 4
    while progress_tracker.scraped[progress_tracker.scraped == 'N'].count() != 0:
        bts_crawler.extract(metadata=meta_df, download=True, incremental=True, n_workers=n_workers,
                            fresh_start=False, auto_fresh_start=False,
                            open_headless=False, open_with_proxy_server=open_with_proxy_server, randomize_proxy_usage=True,
                            compile_progress_files=False, clean=False, delete_progress=False)

        if trials <= 4:
            reset_na = True
        else:
            reset_na = False
        progress_tracker = exclude_scraped_products_from_tracker(
            bts_crawler, reset_na=reset_na)

        trials -= 1
        if trials == 0:
            break

    bts_crawler.extract(metadata=None, download=False, fresh_start=False, auto_fresh_start=False,
                        compile_progress_files=True,  clean=True, delete_progress=True)
    # Path(review_crawler.review_path/'sph_review_progress_tracker.csv').unlink()
    bts_crawler.terminate_logging()
    del bts_crawler
    gc.collect()


if __name__ == "__main__":
    bts_crawler = DetailReview(
        path=Path('D:\\Amit\\Meiyume\\meiyume_data\\spider_runner'))

    gecko_log_path = bts_crawler.review_path/'service/geckodriver.log'
    if gecko_log_path.exists():
        gecko_log_path.unlink()

    files = list(bts_crawler.review_crawler_trigger_path.glob(
        'no_cat_cleaned_bts_product_metadata_all*'))

    if len(files) > 0:
        meta_df = pd.read_feather(files[-1])[
            ['prod_id', 'product_name', 'product_page', 'meta_date']]

        meta_df = get_metadata_with_last_scraped_review_date(meta_df)

        run_bts_crawler(meta_df=meta_df, bts_crawler=bts_crawler)

        meta_ranker = SexyMetaDetail(
            path='D:/Amit/Meiyume/meiyume_data/spider_runner')
        meta_detail = meta_ranker.make(source='bts')

        del meta_detail
        gc.collect()

        sexy_ing = SexyIngredient(
            path='D:/Amit/Meiyume/meiyume_data/spider_runner')
        ing = sexy_ing.make(source='bts')

        del ing
        gc.collect()

        Path(files[0]).unlink()

        if gecko_log_path.exists():
            gecko_log_path.unlink()
    else:
        print('*Metadata Trigger File Not Found.*')
        sys.exit(1)
