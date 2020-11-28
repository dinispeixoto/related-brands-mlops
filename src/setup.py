import os
import numpy as np
import matplotlib.pyplot as plt

from azureml.core import Workspace, Experiment, Dataset
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget


COMPUTE_NAME = 'githubcluster'
DATASET_NAME = 'brand_descriptions'
EXPERIMENT_NAME = 'related-brands'

# Load workspace configuration from the config.json file in the current folder
# TO-DO: documentation how on to download the configuration file
workspace = Workspace.from_config()

# Create an experiment, our runs will all share the same experiment
experiment = Experiment(workspace=workspace, name=EXPERIMENT_NAME)

# This will take advantage of cluster that is already available
# There's two options: a) create it through the UI b) run the sample pipeline
compute_target = workspace.compute_targets[COMPUTE_NAME]
if compute_target and type(compute_target) is AmlCompute:
    print("Found compute target: " + COMPUTE_NAME)


# Create a folder data/ where our dataset will be placed
data_folder = os.path.join(os.getcwd(), '../data')
os.makedirs(data_folder, exist_ok=True)


# Download the brand descriptions dataset 
brand_descriptions_dataset = workspace.datasets[DATASET_NAME]
brand_descriptions_dataset.to_csv_files(separator=',').download(target_path=data_folder, overwrite=True)


# Training script
# script_folder = os.path.join(os.getcwd(), "train.py")

