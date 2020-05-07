from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
## python utilities imports
import os
import time
import warnings
from pathlib import Path
from ast import literal_eval
from datetime import datetime, timedelta
from functools import reduce

## scipy imports
import numpy as np
import pandas as pd
import swifter
from tqdm import tqdm
from tqdm.notebook import tqdm

tqdm.pandas()
warnings.simplefilter(action='ignore')
np.random.seed(1337)

## text lib imports
import re
import string
import unidecode
import spacy

nlp = spacy.load('en_core_web_lg')

## custom utilities imports
from meiyume.utils import (Logger, Sephora, Boots, nan_equal, show_missing_value,\
                           MeiyumeException, S3FileManager)

file_manager = S3FileManager()


class Cleaner():
    pass
