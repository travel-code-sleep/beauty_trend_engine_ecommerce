"""This script runs spider to grab sephora metadata."""
import gc
import os
import time
import warnings
from pathlib import Path

import pandas as pd
from meiyume.bts.crawler import Metadata
from meiyume.utils import chunks, ranges

# from meiyume.utils import Sephora
warnings.simplefilter(action='ignore')

open_with_proxy_server = False
randomize_proxy_usage = False


def exclude_scraped_pages_from_tracker(metadata_crawler: Metadata, reset_na: bool = False) -> pd.DataFrame:
    """exclude_scraped_pages_from_tracker.

    [extended_summary]

    Args:
        metadata_crawler (Metadata): [description]

    Returns:
        pd.DataFrame: [description]
    """
    progress_tracker = pd.read_feather(
        metadata_crawler.metadata_path/'bts_metadata_progress_tracker')

    if reset_na:
        progress_tracker.scraped[progress_tracker.scraped == 'NA'] = 'N'

    progress_tracker = progress_tracker[~progress_tracker.scraped.isna(
    )]
    progress_tracker = progress_tracker[progress_tracker.scraped != 'Y']
    progress_tracker = progress_tracker.sample(frac=1).reset_index(drop=True)
    progress_tracker.to_feather(
        metadata_crawler.metadata_path/'bts_metadata_progress_tracker')
    return progress_tracker


def run_metdata_crawler(metadata_crawler: Metadata) -> None:
    """run_metdata_crawler

    [extended_summary]

    Args:
        metadata_crawler (Metadata): [description]
    """
    metadata_crawler.extract(download=True, fresh_start=True, auto_fresh_start=True,
                             n_workers=8, open_headless=False,
                             open_with_proxy_server=open_with_proxy_server,
                             randomize_proxy_usage=randomize_proxy_usage,
                             compile_progress_files=False, clean=False, delete_progress=False)

    progess_tracker = exclude_scraped_pages_from_tracker(
        metadata_crawler, reset_na=True)
    n_workers = 3
    trials = 5
    while progess_tracker.scraped[progess_tracker.scraped == 'N'].count() != 0:
        metadata_crawler.extract(download=True, fresh_start=False, auto_fresh_start=False,
                                 n_workers=n_workers, open_headless=False,
                                 open_with_proxy_server=open_with_proxy_server,
                                 randomize_proxy_usage=randomize_proxy_usage,
                                 compile_progress_files=False, clean=False, delete_progress=False)

        if trials <= 2:
            reset_na = True
        else:
            reset_na = False
        progress_tracker = exclude_scraped_pages_from_tracker(
            metadata_crawler, reset_na=reset_na)

        trials -= 1
        if trials == 0:
            break
    metadata_crawler.extract(download=False, fresh_start=False, auto_fresh_start=False,
                             compile_progress_files=True, clean=True, delete_progress=False)
    metadata_crawler.terminate_logging()
    del metadata_crawler
    gc.collect()


if __name__ == "__main__":
    metadata_crawler = Metadata(
        path="D:/Amit/Meiyume/meiyume_data/spider_runner")

    gecko_log_path = metadata_crawler.metadata_path/'service/geckodriver.log'
    if gecko_log_path.exists():
        gecko_log_path.unlink()

    run_metdata_crawler(metadata_crawler)

    if gecko_log_path.exists():
        gecko_log_path.unlink()
