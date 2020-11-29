import os
import glob
import time
import joblib
import argparse

import numpy as np
import pandas as pd

from azureml.core import Run
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--min-df', type=float, dest='min_df', default=1, help='min document frequency')
parser.add_argument('--max-df', type=float, dest='max_df', default=1.0, help='max document frenquency')


args = parser.parse_args()

path = args.data_folder
min_df = args.min_df
max_df = args.max_df

all_files = glob.glob(os.path.join(path, "*"))
print(all_files)

df_from_each_file = (pd.read_csv(file) for file in all_files)
brand_descriptions_df = pd.concat(df_from_each_file, ignore_index=True)

print(brand_descriptions_df.head(5))

run = Run.get_context()

print("Training model...")
start = time.time()

model = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=min_df, max_df=max_df, stop_words='english', )
model.fit(brand_descriptions_df['description']) # Transform missing

print("Training completed")
end = time.time()
run.log('time', end - start)

print("Exporting model")
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/related_brands_tfidf_model.pkl')

print("Training job finished")
