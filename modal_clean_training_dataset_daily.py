import os
import modal
import joblib

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("training_dataset_cleaner_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "seaborn","scikit-learn==1.1.1","dataframe-image","Pillow"]) 

   @stub.function(cpu=1.0, image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("hopsworks_iris_api"))
   def f():
       g()

def g():
    '''
    Clears the training dataset for each feature view periodically
    '''
    import hopsworks
    import os

    # Connect to hopsworks and get the feature_group metadata
    hopsworks_api_key= os.environ["HOPS_LAB1_IRIS_KEY"]
    project = hopsworks.login(api_key_value = hopsworks_api_key)
    fs = project.get_feature_store()

    fw_wine_samples = fs.get_feature_view(name="wine_samples", version=1)
    fw_wine_quality = fs.get_feature_view(name="wine_quality", version=1)
    fw_iris         = fs.get_feature_view(name="iris",         version=1)

    fw_wine_samples.delete_all_training_datasets()
    fw_wine_quality.delete_all_training_datasets()
    fw_iris.delete_all_training_datasets()