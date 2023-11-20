import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os

hopsworks_iris_api_key = os.environ["HOPSWORKS_API_LAB1"]
project = hopsworks.login(api_key_value = hopsworks_iris_api_key)
fs = project.get_feature_store()

# Download the pre-trained model and load it
mr = project.get_model_registry()
model = mr.get_model("wine_model_feature_creator", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model_feature_creator.pkl")
print("Model downloaded")

def iris(type, fixed_acid, volatile_acid, citric_acid, residual_sugar, chlorides, free_sd, total_sd, density, ph, sulphates, alcohol):
    print("Calling function")

    colour = 0
    if fixed_acid == 'white':
        colour = 1
    else:
        colour = 2
     
    df = pd.DataFrame([[colour, fixed_acid, volatile_acid, citric_acid, residual_sugar, chlorides, free_sd, density, ph, sulphates, alcohol]], 
                      columns=['type', 'fixed_acid', 'volatile_acid', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sd', 'density', 'ph', 'sulphates', 'alcohol'])
    print("Predicting")
    print(df)

    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    print(res[0])

    flower_url = "https://raw.githubusercontent.com/SebastianoMeneghin/fotografie_iris/main/" + str(res[0]) + ".png"
    response = requests.get(flower_url, stream=True)
    img = Image.open(response.raw)
    img_array = np.array(img)
    return img_array
        
demo = gr.Interface(
    fn=iris,
    title="Test Wine Quality",
    description="Experiment with wine characteristics to predict which its quality is!",
    allow_flagging="never",
    inputs=[
        gr.Dropdown(label="Type", choices=["white", "red"]),
        gr.Number(label="Fixed Acidity"),
        gr.Number(label="Volatice Acidity"),
        gr.Number(label="Citric Acid"),
        gr.Number(label="Residual Sugar"),
        gr.Number(label="Chlorides"),
        gr.Number(label="Free Sulfur Dioxide"),
        gr.Number(label="Total Sulfur Dioxide"),
        gr.Number(label="Density"),
        gr.Number(label="pH"),
        gr.Number(label="sulphates"),
        gr.Number(label="alcohol"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch(debug=True)