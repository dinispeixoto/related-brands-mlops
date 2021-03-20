import os
import glob
import time
import joblib
import argparse

import numpy as np
import pandas as pd

from azureml.core import Run, Dataset
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


def main(args):
    min_df = args.min_df
    max_df = args.max_df
    dataset_name = args.dataset

    run = Run.get_context()
    workspace = run.experiment.workspace

    # Getting dataset
    brand_descriptions_dataset = Dataset.get_by_name(workspace, name=dataset_name)
    brand_descriptions_df = brand_descriptions_dataset.to_pandas_dataframe()

    print(brand_descriptions_df.head(5))

    print("Training model...")
    start = time.time()

    model = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=min_df, max_df=max_df, stop_words='english', )
    tfidf_matrix = model.fit_transform(brand_descriptions_df['description'])

    print("Training completed")
    end = time.time()
    run.log('time', end - start)

    # Build Recommendations
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    recommendations = {} # dictionary created to store the result in a dictionary format (ID : (Score,item_id))#
    for idx, row in brand_descriptions_df.iterrows(): #iterates through all the rows
        similar_indices = cosine_similarities[idx].argsort()[:-15:-1] 
        similar_items = [(cosine_similarities[idx][i], brand_descriptions_df['brand_id'][i]) for i in similar_indices]
        recommendations[row['brand_id']] = {'name': row['brand_name'], 'recommendations': similar_items[1:]}

    print("Exporting model")
    os.makedirs('outputs', exist_ok=True)

    joblib.dump(value=recommendations, filename='outputs/related_brands_recommendations.pkl')
    print("Training job finished")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, dest='dataset', help='dataset name')
    parser.add_argument('--min-df', type=float, dest='min_df', default=1, help='min document frequency')
    parser.add_argument('--max-df', type=float, dest='max_df', default=1.0, help='max document frenquency')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args=args)
