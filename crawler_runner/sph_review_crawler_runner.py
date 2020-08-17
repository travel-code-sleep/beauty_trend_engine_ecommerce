""" This script runs spider to grab sephora review data."""
import gc
import os
import time
import warnings
from pathlib import Path

import pandas as pd
from meiyume.sph.crawler import Review
from meiyume.utils import chunks, ranges

warnings.simplefilter(action='ignore')


def get_metadata_with_last_scraped_review_date(meta_df: pd.DataFrame, review_crawler: Review) -> pd.DataFrame:
    """get_metadata_with_last_scraped_review_date [summary]

    [extended_summary]

    Args:
        review_crawler (Review): [description]

    Returns:
        pd.DataFrame: [description]
    """
    review_files = review_crawler.old_review_clean_files_path.glob(
        'cleaned_sph_product_review_all*')
    rvf = []
    for file in review_files:
        df = pd.read_feather(file)
        rvf.append(df)
    rev_df = pd.concat(rvf, axis=0, ignore_index=True)
    rev_df.drop_duplicates(inplace=True)
    rev_df.reset_index(inplace=True, drop=True)

    df = rev_df.groupby('prod_id').review_date.max().reset_index()
    df.columns = ['prod_id', 'last_scraped_review_date']
    df.set_index('prod_id', inplace=True)
    del rev_df
    gc.collect()

    meta_df.set_index('prod_id', inplace=True)
    meta_df = meta_df.join(df, how='left')
    meta_df.reset_index(inplace=True)
    meta_df = meta_df.sample(frac=1).reset_index(drop=True)
    return meta_df


def exclude_scraped_products_from_tracker(review_crawler: Review, reset_na: bool = False) -> pd.DataFrame:
    """exclude_scraped_products_from_tracker [summary]

    [extended_summary]

    Args:
        review_crawler (Review): [description]
        reset_na (bool, optional): [description]. Defaults to False.

    Returns:
        pd.DataFrame: [description]
    """
    progress_tracker = pd.read_csv(
        review_crawler.review_path/'sph_review_progress_tracker.csv')

    if reset_na:
        progress_tracker.review_scraped[progress_tracker.review_scraped == 'NA'] = 'N'

    progress_tracker = progress_tracker[~progress_tracker.review_scraped.isna(
    )]
    progress_tracker = progress_tracker[progress_tracker.review_scraped != 'Y']
    progress_tracker = progress_tracker.sample(frac=1).reset_index(drop=True)
    progress_tracker.to_csv(
        review_crawler.review_path/'sph_review_progress_tracker.csv', index=None)
    return progress_tracker


def run_review_crawler(meta_df: pd.DataFrame, review_crawler: Review):
    """run_review_crawler [summary]

    [extended_summary]

    Args:
        meta_df (pd.DataFrame): [description]
        review_crawler (Review): [description]
    """
    for i in ranges(meta_df.shape[0], 30):
        print(i[0], i[-1])
        if i[0] == 0:
            fresh_start = True
            auto_fresh_start = True
        else:
            fresh_start = False
            auto_fresh_start = False
        review_crawler.extract(metadata=meta_df, download=True, incremental=True, n_workers=8,
                               fresh_start=fresh_start, auto_fresh_start=auto_fresh_start,
                               start_idx=i[0], end_idx=i[-1],
                               open_headless=False, open_with_proxy_server=True, randomize_proxy_usage=True,
                               complie_progress_files=False, clean=False, delete_progress=False)

        review_crawler.terminate_logging()
        del review_crawler
        gc.collect()
        time.sleep(5)
        review_crawler = Review(
            path="D:/Amit/Meiyume/meiyume_data/spider_runner")

    progress_tracker = exclude_scraped_products_from_tracker(
        review_crawler, reset_na=True)
    n_workers = 6
    trials = 10
    while progress_tracker.review_scraped[progress_tracker.review_scraped == 'N'].count() != 0:
        review_crawler.extract(metadata=meta_df, download=True, incremental=True, n_workers=n_workers,
                               fresh_start=False, auto_fresh_start=False,
                               open_headless=False, open_with_proxy_server=True, randomize_proxy_usage=True,
                               complie_progress_files=False, clean=False, delete_progress=False)

        if trials <= 4:
            reset_na = True
        else:
            reset_na = False
        progress_tracker = exclude_scraped_products_from_tracker(
            review_crawler, reset_na=reset_na)

        trials -= 1
        if trials == 0:
            break

    review_crawler.extract(metadata=None, download=False, fresh_start=False, auto_fresh_start=False,
                           complie_progress_files=True,  clean=True, delete_progress=False)
    # Path(review_crawler.review_path/'sph_review_progress_tracker.csv').unlink()
    review_crawler.terminate_logging()
    del review_crawler
    gc.collect()


if __name__ == "__main__":
    review_crawler = Review(
        path="D:/Amit/Meiyume/meiyume_data/spider_runner")

    files = list(review_crawler.review_crawler_trigger_path.glob(
        'no_cat_cleaned_sph_product_metadata_all*'))

    if len(files) > 0:
        meta_df = pd.read_feather(files[0])[
            ['prod_id', 'product_name', 'product_page', 'meta_date']]

        meta_df = get_metadata_with_last_scraped_review_date(
            meta_df, review_crawler)

        run_review_crawler(meta_df=meta_df, review_crawler=review_crawler)

        Path(files[0]).unlink()
