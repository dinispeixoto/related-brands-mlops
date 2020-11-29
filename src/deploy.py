import uuid

from azureml.core import Workspace
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import Webservice, AciWebservice


# Create the configuration for the service on the ACI
aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                            #    tags={"data": "MNIST",  "method" : "sklearn"}, 
                                               description='Predict related brands')

# Load workspace
workspace = Workspace.from_config()

# Retrieve model from model registry
model = Model(workspace, 'related_brands')

# Use the same environment to run the application on the web service
environment = Environment.get(workspace=workspace, name="training_env", version="1")
inference_config = InferenceConfig(entry_script="score.py", environment=environment)

service_name = 'related-brands-svc-' + str(uuid.uuid4())[:4]
service = Model.deploy(workspace=workspace, 
                       name=service_name, 
                       models=[model], 
                       inference_config=inference_config, 
                       deployment_config=aciconfig)

service.wait_for_deployment(show_output=True)