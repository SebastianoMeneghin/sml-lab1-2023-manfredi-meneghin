# Task 2: Wine Quality Prediction


## About
This is the **Task 2 of the Lab1** of the course [ID2223 Scalable Machine Learning and Deep Learning](https://www.kth.se/student/kurser/kurs/ID2223?l=en).

The task consists of an **machine learning pipeline** created to predict the quality of a wine, given some specific characteristics of the wine. 


### The Team

* [Giovanni Manfredi](https://github.com/Silemo)
* [Sebastiano Meneghin](https://github.com/SebastianoMeneghin)


## Table of Contents
* [Introduction](#Introduction)
* [Modules description](#Modules-description)
* [Software used](#Software-used)


## Introduction
The **main steps** of the project are:
- 🍷 **EDA and backfill:** the features are taken from an online source. They are inspected and visualized, then modified according to what the analyses discovered. Data are then uploaded on Hopsworks feature store.
- 🏋🏻‍♀️ **Model evaluation and training:** the feature are downloaded from the feature store and different models are trained on them. The most accurate models are then saved on Hopsworks model registry.
- 🧪 **Feature creation:** once every day, a new feature is created thanks to a remote script execution which is running on Modal, depending on the statistics of the original data. Then it is uploaded in the main feature store.
- 🔎 **Performance monitoring:** new data created are tested daily and some performance metrics, as prediction accuracy and confusion matrix are created each time.
- 🕹️ **Data interaction:** it is possible to predict your own wine quality and look at the performances thanks to two User Interfaces created with Gradio tools and then deployed on HuggingFace;
    - 👀 [*Wine quality online prediction*](https://huggingface.co/spaces/SebastianoMeneghin/wine_quality)
    - 📺 [*Wine prediction monitor*](https://huggingface.co/spaces/SebastianoMeneghin/Wine_quality_monitoring)

**Once set the correct environments**, environment variables and secrets among the platforms, are described in *requirements.txt*, *environment.yml* and in the [*task description*](https://github.com/SebastianoMeneghin/sml-lab1-2023-manfredi-meneghin/blob/main/id2223_kth_lab1_2023.pdf), the project is completely runnable. If you are gonna have problems with too many training set, you can remotely run on Modal a [*dataset cleaner script*](https://github.com/SebastianoMeneghin/sml-lab1-2023-manfredi-meneghin/blob/main/modal_clean_training_dataset_daily.py).


## Modules description
### 🍷 wq_eda_and_backfill_feature_group.py/ipynb
Here's were the **data are taken from the online repository** of Wine Quality Dataset. The **data are visualized** using a mix of *graphical and analytical tools*, such as boxplot, violinplot, correlation matrix, but also quantiles, mean, max and min. (Graphical tools can be easily run on the [provided notebook](https://github.com/SebastianoMeneghin/sml-lab1-2023-manfredi-meneghin/blob/main/wine_quality/src/wq_eda_and_backfill_feature_group.ipynb), that contains the same code of the python file script).

The **analysis show** an high correlation between free sulfur dioxide and total free dioxide, so only one of the two is saved, to further train the ML models.

Since the original data are full of outliners and duplicates, a specific **section of the code clean the data** and add a variable "key" to the feature, that is used as primary key on the feature store.

Once normalized, **data are binned**, string attributes such as 'type' are converted to integer, and float attributes are converted to integer as well.

Two **different feature groups are stored** on Hopsworks, representing the data in two different stages of the pre-processing step. The second feature group is indeed created to keep track of useful attributes' statistics as mean, standard deviation, min and max, that will be used to create the data.

### 🏋🏻‍♀️ wq_training_pipeline.py
At the beginning of the code, some boolean variables (RF_TEST, GB_TEST, KNN_TEST) are configured, to **determine whether to evaluate specific machine learning models**. By setting them true, the code will perform also the model evualation. Following this, a feature view for the "wine_quality" dataset is then established, to access the data for training and testing

The core f this part is about training and testing machine learning models. **KNearestNeighbors (KNN), Gradient Boosting (GB) and RandomForest (RF) models are trained and tested**, but due to the low performance, KNN is not selected as prediction model. GB and RF are selected models and then trained with their best parameters on the dataset, and their performance metrics are shown. **GB is used as the prediction model** for the new features, **while RF is used** to determine the value of the 'quality' **for the new feature daily created**.

Essential components, including the confusion matrix image and the trained GB and RF models, are saved in designated directories. Furthermore, input and output schemas are defined for the models and they are **registered in the Hopsworks Model Registry**.

### 🧪 wq_feature_pipeline_daily.py
At the beginning, the LOCAL variable is set, determining **whether the code is executed locally or stubbed** on a modal environment. When not run locally it utilizes the modal library to set up the "wine_daily_creation" function, configuring it with a periodic daily schedule for execution, and the necessary, as well as the hopsworks secret key.

The **stub function generate, normalize and predict new wine features.** The *generate_wine* function constructs a single wine feature, incorporating a random wine type based on a the attributions' distributions. The statistics are obtained from the *wine_samples* feature group using the *get_wine_stats* function, facilitating the creation of random qualities via *create_random_qualities* function.

The *normalize_wine* function prepares the wine features for machine learning processing by **binning numerical values** and converting them into integers. The wine type is translated in numerical labels. The *predict_new_label function* is used to **predict labels for the new features** with the pre-trained RF model.

The most important function is *get_random_wine*: it **integrates these functionalities**, generating a random wine feature, normalizing it, predicting labels, and displaying the resulting row in form of a DataFrame. This is then **saved into the main feature group.**

### 🔎 wq_batch_inference_pipeline.py

### 🧹(extra) model_clean_training_dataset_daily.py
------------------------------------------------
### Software used

**Visual Studio Code** - main IDE

**GitKraken** - github

**Gradio & HuggingFace** - GUI

**Modal** - run daily remote script

**Hopsworks** - MLOps platform