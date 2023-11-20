import os
import modal
    
LOCAL=False

if LOCAL == False:
   stub = modal.Stub("iris_batch")
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.1.1","dataframe-image","Pillow"])
   
   @stub.function(image=hopsworks_image, schedule=modal.Period(hours=1), secret=modal.Secret.from_name("hopsworks_iris_api"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests
    import io

    # Login to hopsworks and get access to the file system
    hopsworks_api_key= os.environ["HOPS_LAB1_IRIS_KEY"]
    #hopsworks_api_key = os.environ["HOPSWORKS_IRIS_APY_KEY"]
    project = hopsworks.login(api_key_value = hopsworks_api_key)
    fs = project.get_feature_store()
    
    # Download the pre-trained model and load it
    mr = project.get_model_registry()
    model = mr.get_model("iris_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/iris_model.pkl")
    
    # Get the features from Hopsworks view
    feature_view = fs.get_feature_view(name="iris", version=1)
    batch_data = feature_view.get_batch_data()
    

    # Get the predictions of the model
    y_pred = model.predict(batch_data)
    #print(y_pred)

    # Get the prediction of the last flower inserted
    offset = 1
    flower = y_pred[y_pred.size-offset]

    # Get the image of the flower online, from an online GitHub repo, through HTTP method
    flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + flower + ".png"
    print("Flower predicted: " + flower)
    img = Image.open(requests.get(flower_url, stream=True).raw)            
    img.save("./latest_iris.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_iris.png", "Resources/images", overwrite=True)


    # Get from the feature store the full table (feature group) and gets the actual value of the last prediction
    iris_fg = fs.get_feature_group(name="iris", version=1)
    df = iris_fg.read() 
    #print(df)
    label = df.iloc[-offset]["variety"]
    label_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + label + ".png"
    print("Flower actual: " + label)
    img = Image.open(requests.get(label_url, stream=True).raw)        
    img.save("./actual_iris.png")
    dataset_api.upload("./actual_iris.png", "Resources/images", overwrite=True)
    
    # Create a new feature group where to put the couple "prediction/real label"
    monitor_fg = fs.get_or_create_feature_group(name="iris_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Iris flower Prediction/Outcome Monitoring"
                                                )
    
    # Get the new time and create a new row to store. Then insert (upload it on Hopsworks) it on the fg created.
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [flower],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    # If there problem persist, add the read option: .read("use_hive": True)
    history_df = monitor_fg.read()
   
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    # Take the recent history and create from it a graph, that than is saved on Hopsworks
    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    # Take the column of predictions and labels
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    print("Number of predictions to date: " + str(len(predictions)))
    print("Number of different flower type predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 3:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Setosa', 'True Versicolor', 'True Virginica'],
                             ['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 3 different flower type predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 3 different iris flower type predictions") 