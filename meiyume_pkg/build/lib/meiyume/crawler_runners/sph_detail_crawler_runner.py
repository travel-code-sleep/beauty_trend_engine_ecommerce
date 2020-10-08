"""This script runs spider to grab sephora metadata."""
import gc
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from meiyume.sph.crawler import Detail
from meiyume.algorithms import SexyMetaDetail, SexyIngredient
from meiyume.utils import chunks, ranges

warnings.simplefilter(action='ignore')

open_with_proxy_server = True


def exclude_scraped_products_from_tracker(detail_crawler: Detail, reset_na: bool = False) -> pd.DataFrame:
    """exclude_scraped_products_from_tracker [summary]
    [extended_summary]
    Args:
        detail_crawler (Detail): [description]
        reset_na (bool, optional): [description]. Defaults to False.
    Returns:
        pd.DataFrame: [description]
    """
    progress_tracker = pd.read_feather(
        detail_crawler.detail_path/'sph_detail_progress_tracker')

    if reset_na:
        progress_tracker.detail_scraped[progress_tracker.detail_scraped == 'NA'] = 'N'

    progress_tracker = progress_tracker[~progress_tracker.detail_scraped.isna(
    )]
    progress_tracker = progress_tracker[progress_tracker.detail_scraped != 'Y']
    progress_tracker = progress_tracker.sample(frac=1).reset_index(drop=True)
    progress_tracker.to_feather(
        detail_crawler.detail_path/'sph_detail_progress_tracker')
    return progress_tracker


def run_detail_crawler(meta_df: pd.DataFrame, detail_crawler: Detail):
    """run_detail_crawler [summary]
    [extended_summary]
    Args:
        meta_df (pd.DataFrame): [description]
        detail_crawler (Detail): [description]
    """
    for i in ranges(meta_df.shape[0], 30):
        print(i[0], i[-1])
        if i[0] == 0:
            fresh_start = False
            auto_fresh_start = False
        else:
            fresh_start = False
            auto_fresh_start = False
        detail_crawler.extract(metadata=meta_df, download=True, n_workers=8,
                               fresh_start=fresh_start, auto_fresh_start=auto_fresh_start,
                               start_idx=i[0], end_idx=i[-1],
                               open_headless=False, open_with_proxy_server=open_with_proxy_server,
                               randomize_proxy_usage=True,
                               compile_progress_files=False, clean=False, delete_progress=False)

        detail_crawler.terminate_logging()
        del detail_crawler
        gc.collect()
        time.sleep(5)
        detail_crawler = Detail(
            path="D:/Amit/Meiyume/meiyume_data/spider_runner")

    progress_tracker = exclude_scraped_products_from_tracker(
        detail_crawler, reset_na=True)
    n_workers = 4
    trials = 10
    while progress_tracker.detail_scraped[progress_tracker.detail_scraped == 'N'].count() != 0:
        detail_crawler.extract(metadata=meta_df, download=True, n_workers=n_workers,
                               fresh_start=False, auto_fresh_start=False,
                               open_headless=False, open_with_proxy_server=open_with_proxy_server,
                               randomize_proxy_usage=True,
                               compile_progress_files=False, clean=False, delete_progress=False)
        if trials <= 4:
            reset_na = True
        else:
            reset_na = False
        progress_tracker = exclude_scraped_products_from_tracker(
            detail_crawler, reset_na=reset_na)

        trials -= 1
        if trials == 0:
            break

    detail_crawler.extract(metadata=None, download=False, fresh_start=False, auto_fresh_start=False,
                           compile_progress_files=True,  clean=True, delete_progress=True)
    # Path(detail_crawler.review_path/'sph_review_progress_tracker.csv').unlink()
    detail_crawler.terminate_logging()
    del detail_crawler
    gc.collect()


if __name__ == "__main__":
    detail_crawler = Detail(
        path="D:/Amit/Meiyume/meiyume_data/spider_runner")

    gecko_log_path = detail_crawler.detail_path/'service/geckodriver.log'
    if gecko_log_path.exists():
        gecko_log_path.unlink()

    files = list(detail_crawler.detail_crawler_trigger_path.glob(
        'no_cat_cleaned_sph_product_metadata_all*'))

    if len(files) > 0:
        meta_df = pd.read_feather(files[0])

        run_detail_crawler(meta_df=meta_df, detail_crawler=detail_crawler)

        # Path(files[0]).unlink()

        meta_ranker = SexyMetaDetail(
            path='D:/Amit/Meiyume/meiyume_data/spider_runner')
        meta_detail = meta_ranker.make(source='sph')

        del meta_detail
        gc.collect()

        sexy_ing = SexyIngredient(
            path='D:/Amit/Meiyume/meiyume_data/spider_runner')
        ing = sexy_ing.make(source='sph')

        del ing
        gc.collect()

        if gecko_log_path.exists():
            gecko_log_path.unlink()
    else:
        print('*Metadata Trigger File Not Found.*')
        sys.exit(1)
