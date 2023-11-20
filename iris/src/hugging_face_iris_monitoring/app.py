import gradio as gr
from PIL import Image
import hopsworks
import os

hopsworks_iris_api_key = os.environ['HOPSWORKS_IRIS_APY_KEY']
project = hopsworks.login(api_key_value = hopsworks_iris_api_key)
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/latest_iris.png")
dataset_api.download("Resources/images/actual_iris.png")
dataset_api.download("Resources/images/df_recent_iris.png")
dataset_api.download("Resources/images/confusion_matrix_iris.png")

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Today's Predicted Image")
          input_img = gr.Image("latest_iris.png", elem_id="predicted-img")
      with gr.Column():          
          gr.Label("Today's Actual Image")
          input_img = gr.Image("actual_iris.png", elem_id="actual-img")        
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction History")
          input_img = gr.Image("df_recent_iris.png", elem_id="recent-predictions")
      with gr.Column():          
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")
          input_img = gr.Image("confusion_matrix_iris.png", elem_id="confusion-matrix")        

demo.launch()