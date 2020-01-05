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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import swifter
from meiyume.utils import Logger, Sephora, nan_equal, show_missing_value, MeiyumeException
from tqdm import tqdm


class Ingredients(Sephora):
    def __init__(self, path='.'):
        super().__init__(path=path)

    def prepare_data(self):
        meta_files = self.metadata_clean_path.glob(
            'cat_cleaned_sph_product_metadata_all*')
        meta = pd.read_feather(max(meta_files, key=os.path.getctime))
        ingredient_files = self.detail_clean_path.glob(
            'cleaned_sph_product_ingredient_all*')
        self.ingredient = pd.read_feather(
            max(ingredient_files, key=os.path.getctime))

        clean_prod_type = meta.product_type[meta.product_type.swifter.apply(
            lambda x: True if x.split('-')[0] == 'clean' else False)].unique()
        new_product_list = meta.prod_id[meta.new_flag == 'NEW'].unique()
        clean_product_list = meta.prod_id[meta.product_type.isin(
            clean_prod_type)].unique()

        self.ingredient['new_flag'] = self.ingredient.prod_id.swifter.apply(
            lambda x: 'New' if x in new_product_list else 'Old')
        self.ingredient['clean_flag'] = self.ingredient.prod_id.swifter.apply(
            lambda x: 'Clean' if x in clean_product_list else 'No')
