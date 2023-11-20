import gradio as gr
from PIL import Image
import hopsworks
import os

hopsworks_iris_api_key = os.environ["HOPSWORKS_API_LAB1"]
project = hopsworks.login(api_key_value = hopsworks_iris_api_key)
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/latest_wine_quality.png")
dataset_api.download("Resources/images/actual_wine_quality.png")
dataset_api.download("Resources/images/df_recent_wine.png")
dataset_api.download("Resources/images/confusion_matrix_wine.png")

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Today's Predicted Wine Quality")
          input_img = gr.Image("latest_wine_quality.png", elem_id="predicted-img")
      with gr.Column():          
          gr.Label("Today's Actual Wine Image")
          input_img = gr.Image("actual_wine_quality.png", elem_id="actual-img")        
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction History")
          input_img = gr.Image("df_recent_wine.png", elem_id="recent-predictions")
      with gr.Column():          
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")
          input_img = gr.Image("confusion_matrix_wine.png", elem_id="confusion-matrix")        

demo.launch()