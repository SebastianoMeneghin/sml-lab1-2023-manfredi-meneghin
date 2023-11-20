import os
import modal
import joblib

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("batch_wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "seaborn","scikit-learn==1.1.1","dataframe-image","Pillow", "numpy"]) 

   @stub.function(cpu=1.0, image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("hopsworks_iris_api"))
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
    import numpy as np

    # Login to hopsworks and get access to the file system
    hopsworks_api_key= os.environ["HOPS_LAB1_IRIS_KEY"]
    project = hopsworks.login(api_key_value = hopsworks_api_key)
    fs = project.get_feature_store()
    
    # Download the pre-trained model and load it
    mr = project.get_model_registry()
    model = mr.get_model("wine_model_pred", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model_pred.pkl")
    
    # Get the features from Hopsworks view and get the prediction on the new feature
    feature_view = fs.get_feature_view(name="wine_quality", version=1)
    batch_data = feature_view.get_batch_data()
    cleaned_batch = batch_data.drop(columns='key')
    y_pred = model.predict(cleaned_batch)

    # Get the prediction of the last wine inserted
    offset = 1
    wine = y_pred[y_pred.size - offset]
    print('This is your wine prediction:', wine)

    # Get the image of the wine quality online, from an online GitHub repo, through HTTP method
    wine_url = "https://raw.githubusercontent.com/SebastianoMeneghin/fotografie_iris/main/" + str(wine) + ".png"
    print("Wine quality predicted: " + str(wine))

    img = Image.open(requests.get(wine_url, stream=True).raw)            
    img.save("./latest_wine_quality.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_wine_quality.png", "Resources/images", overwrite=True)


    # Get from the feature store the full table (feature group) and gets the actual value of the last prediction
    wine_fg = fs.get_feature_group(name="wine_quality", version=1)
    df = wine_fg.read()

    quality_label = df.iloc[-offset]["quality"]
    quality_label_url = "https://raw.githubusercontent.com/SebastianoMeneghin/fotografie_iris/main/" + str(quality_label) + ".png"
    print("Actual wine quality: " + str(quality_label))
    img = Image.open(requests.get(quality_label_url, stream=True).raw)        
    img.save("./actual_wine_quality.png")
    dataset_api.upload("./actual_wine_quality.png", "Resources/images", overwrite=True)
    
    # Create a new feature group where to put the couple "prediction/real label"
    monitor_fg = fs.get_or_create_feature_group(name="wine_quality_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine quality Prediction/Outcome Monitoring"
                                                )
    
    # Get the new time and create a new row to store. Then insert (upload it on Hopsworks) it on the fg created.
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine],
        'label': [quality_label],
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
    dfi.export(df_recent, './df_recent_wine.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent_wine.png", "Resources/images", overwrite=True)
    
    # Take the column of predictions and labels
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    diff_qualities = predictions.value_counts().count()
    # Only create the confusion matrix when our wine_quality_predictions feature group has examples of at least 3 wine qualities
    print("Number of different wine qualities predictions to date: " + str(diff_qualities))
    if diff_qualities >= 2:
        results = confusion_matrix(labels, predictions)

        qualities = predictions.value_counts().index.sort_values()
        qualities_list = []
        for i in range(len(qualities)):
            qualities_array = np.asarray(qualities[i])
            qualities_list.append(qualities_array[0])

        true_qualities = []
        pred_qualities = []
        for quality in qualities_list:
            true_qualities.append('True Quality: ' + str(quality))
            pred_qualities.append('Pred Quality: ' + str(quality))

        df_cm = pd.DataFrame(results, true_qualities, pred_qualities)
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix_wine.png")
        dataset_api.upload("./confusion_matrix_wine.png", "Resources/images", overwrite=True)

    else:
        print("You need at least two wine quality type predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times!") 