"""This script runs spider to grab sephora product images."""
import gc
import os
import time
import warnings
from pathlib import Path
import shutil

import pandas as pd
from meiyume.sph.crawler import Image
from meiyume.utils import RedShiftReader, S3FileManager, chunks, ranges


def get_meta_df() -> pd.DataFrame:
    """get_meta_df [summary]

    [extended_summary]

    Returns:
        pd.DataFrame: [description]
    """
    df = db.query_database(
        "select distinct product_page, prod_id, source from r_bte_meta_detail_f where source='sephora.com'")
    image_keys = [i['Key'] for i in file_manager.get_matching_s3_keys(prefix='Feeds/BeautyTrendEngine/Image/Staging/', suffix='jpg')
                  if any(job in i['Key'].lower() for job in ['image'])]
    prod_ids = set([key.split('/')[4] for key in image_keys])
    df = df[~df.prod_id.isin(prod_ids)]
    return df


def exclude_scraped_products_from_tracker(image_crawler: Image, reset_na: bool = False) -> pd.DataFrame:
    """exclude_scraped_products_from_tracker [summary]

    [extended_summary]

    Args:
        image_crawler (Image): [description]
        reset_na (bool, optional): [description]. Defaults to False.

    Returns:
        pd.DataFrame: [description]
    """
    progress_tracker = pd.read_csv(
        image_crawler.image_path/'sph_image_progress_tracker.csv')

    if reset_na:
        progress_tracker.image_scraped[progress_tracker.image_scraped == 'NA'] = 'N'

    progress_tracker = progress_tracker[~progress_tracker.image_scraped.isna(
    )]
    progress_tracker = progress_tracker[progress_tracker.image_scraped != 'Y']
    progress_tracker = progress_tracker.sample(frac=1).reset_index(drop=True)
    progress_tracker.to_csv(
        image_crawler.image_path/'sph_image_progress_tracker.csv', index=None)
    return progress_tracker


def run_image_crawler(meta_df: pd.DataFrame, image_crawler: Image):
    """run_image_crawler [summary]

    [extended_summary]

    Args:
        meta_df (pd.DataFrame): [description]
        image_crawler (Image): [description]
    """
    for i in ranges(meta_df.shape[0], 30):
        print(i[0], i[-1])
        if i[0] == 0:
            fresh_start = True
            auto_fresh_start = True
        else:
            fresh_start = False
            auto_fresh_start = False
        image_crawler.extract(metadata=meta_df, download=True, n_workers=6,
                              fresh_start=fresh_start, auto_fresh_start=auto_fresh_start,
                              list_of_index=i,
                              open_headless=False, open_with_proxy_server=True, randomize_proxy_usage=True)
    # Path(review_crawler.review_path/'sph_review_progress_tracker.csv').unlink()
    image_crawler.terminate_logging()
    del image_crawler
    gc.collect()


if __name__ == '__main__':

    db = RedShiftReader()
    file_manager = S3FileManager()

    image_crawler = Image(
        path="D:/Amit/Meiyume/meiyume_data/spider_runner")

    gecko_driver_path = image_crawler.image_path/'drivers'
    if gecko_driver_path.exists():
        shutil.rmtree(gecko_driver_path)

    gecko_log_path = image_crawler.image_path/'service/geckodriver.log'
    if gecko_log_path.exists():
        gecko_log_path.unlink()

    meta_df = get_meta_df()

    run_image_crawler(meta_df, image_crawler)

    if gecko_log_path.exists():
        gecko_log_path.unlink()
