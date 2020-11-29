import os
import numpy as np
import matplotlib.pyplot as plt

from azureml.core import Workspace, Experiment, Dataset, ScriptRunConfig
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies


COMPUTE_NAME = 'githubcluster'
DATASET_NAME = 'brand_descriptions'
EXPERIMENT_NAME = 'related-brands'
DATA_FOLDER = '../data/'

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
data_folder = os.path.join(os.getcwd(), DATA_FOLDER)
os.makedirs(data_folder, exist_ok=True)


# Download the brand descriptions dataset 
brand_descriptions_tabular_dataset = workspace.datasets[DATASET_NAME]
brand_descriptions_file_dataset = brand_descriptions_tabular_dataset.to_csv_files(separator=',')

brand_descriptions_file_dataset.download(target_path=data_folder, overwrite=True)

# Setup the conda environment (environment.yml)
environment = Environment('training_env')
conda_dependencies = CondaDependencies.create(pip_packages=['azureml-dataset-runtime[pandas,fuse]', 'azureml-defaults'], conda_packages = ['scikit-learn==0.22.1'])

environment.python.conda_dependencies = conda_dependencies
environment.register(workspace=workspace)

# Submit training job with train.py script
script_folder = os.path.join(os.getcwd())

src = ScriptRunConfig(source_directory=script_folder,
                      script='train.py', 
                      arguments=['--data-folder', brand_descriptions_file_dataset.as_mount()],
                      compute_target=compute_target,
                      environment=environment)

run = experiment.submit(config=src)
run.wait_for_completion(show_output=True) 

model = run.register_model(model_name='related_brands', model_path='outputs/related_brands_tfidf_model.pkl')
print(model.name, model.id, model.version, sep='\t')
