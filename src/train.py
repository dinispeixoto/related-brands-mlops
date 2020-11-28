import argparse
import os
import numpy as np
import glob

from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
# parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
args = parser.parse_args()


data_folder = args.data_folder
print('Data folder:', data_folder)