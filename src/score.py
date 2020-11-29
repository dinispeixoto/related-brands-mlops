import json
import numpy as np
import os
import pickle
import joblib


def init():
    global recommendations
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    recommendations_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'related_brands_recommendations.pkl')
    recommendations = joblib.load(recommendations_path)


def run(raw_data):

    def recommend(id, num=5):
        response = []
        recs = recommendations[id]['recommendations'][:num]
        for rec in recs:
            brand_id = rec[1]
            score = rec[0]
            # brand_name = brand_descriptions_df.loc[brand_descriptions_df['brand_id'] == brand_id, 'brand_name'].iloc[0]
            brand_name = recommendations[brand_id]['name']
            response.append(f"Brand {brand_name} ({brand_id}) with score {str(score)}\n")
        
        return response

    brand_id = json.loads(raw_data)['data'][0]

    return recommend(brand_id)