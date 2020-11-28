import os
import glob
import argparse

import numpy as np
import pandas as pd

from azureml.core import Run
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
# parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
args = parser.parse_args()

path = args.data_folder
all_files = glob.glob(os.path.join(path, "*"))

df_from_each_file = (pd.read_csv(file) for file in all_files)
concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

print(concatenated_df.head(5))
