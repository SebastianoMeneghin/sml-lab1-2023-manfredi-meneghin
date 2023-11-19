import hopsworks
import joblib
import math
import os
import numpy   as np
import pandas  as pd
import seaborn as sns
from   sklearn.neighbors       import KNeighborsClassifier
from   sklearn.ensemble        import GradientBoostingClassifier
from   sklearn.ensemble        import RandomForestClassifier
from   sklearn.model_selection import GridSearchCV
from   sklearn.model_selection import train_test_split
from   sklearn.metrics         import classification_report
from   sklearn.metrics         import confusion_matrix
from   sklearn                 import metrics
from   hsml.schema             import Schema
from   hsml.model_schema       import ModelSchema
from   matplotlib              import pyplot as plt


# Variables that can be turned to "True" to show the models' parameters' sets evaluation
RF_TEST  = False
GB_TEST  = False
KNN_TEST = False


# Here the environment variable is set
hopsworks_iris_api_key = os.environ['HOPSWORKS_IRIS_APY_KEY']
project = hopsworks.login(api_key_value = hopsworks_iris_api_key)
fs = project.get_feature_store()

# Create the feature view to access the dataset on hopsworks
wine_fg = fs.get_feature_group(name="wine_quality", version=1)
query = wine_fg.select_all()
feature_view = fs.get_or_create_feature_view(name="wine_quality",
                                  version=1,
                                  description="Read from Wine Quality Dataset",
                                  labels=["quality"],
                                  query=query)


############################### MODEL TRAINING ###############################
# The model evaluated for this classification problem are:
# - KNN: K Nearest Neighbours
# - GB:  Gradient Boosting
# - RF:  Random Forest
# You can read training data, randomly split into train/test sets of features (X) and labels (y)        
X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)

# Various RandomForest models are tested, in order to select the best parameters. In our project:
# Best parameters:  max_depth = 13, n_estimators: 139
if (RF_TEST):
    clf_RF = RandomForestClassifier(random_state=38)
    params_RF = {'n_estimators': range(25,150,20), 'max_depth': range(3,55,10)}
    gbc_RF = GridSearchCV(clf_RF, param_grid = params_RF, cv = 3, n_jobs=-1, verbose=1)
    gbc_RF.fit(X_train, y_train.values.ravel())
    cv_RF = pd.DataFrame(gbc_RF.cv_results_)
    print(cv_RF.sort_values(by = 'rank_test_score').T)
    print(gbc_RF.best_params_)


# Various GradientBoosting models are tested, in order to select the best parameters. In our project:
# Best parameters: max_depth: 10, 'n_estimators': 85
if (GB_TEST):
    clf_GB = GradientBoostingClassifier(random_state=25)
    params_GB = {'n_estimators': range(25,150,20), 'max_depth': range(3,55,10)}
    gbc_GB = GridSearchCV(clf_GB, param_grid = params_GB, cv = 3, n_jobs=-1, verbose=1)
    gbc_GB.fit(X_train, y_train.values.ravel())
    cv_GB = pd.DataFrame(gbc_GB.cv_results_)
    print(cv_GB.sort_values(by = 'rank_test_score').T)
    print(gbc_GB.best_params_)


# Various KNearestNeighbors models are tested, in order to select the best parameter. In our project:
# Best parameter: n_neighbors = 4
if (KNN_TEST):
    k_neighbors = range(2 , 13 , 2)
    training_acc = []
    testing_acc = []

    for i in k_neighbors:
        model = KNeighborsClassifier(n_neighbors= i)
        model.fit(X_train , y_train.values.ravel())
        training_acc.append(model.score(X_train , y_train.values.ravel()))
        testing_acc.append(model.score(X_test , y_test.values.ravel()))
        
    print(f'Training accuracy: f{training_acc}')
    print(f'Testing accuracy: f{testing_acc}')

    sum_acc = []
    for idx in range(len(training_acc)):
        sum_acc.append(training_acc[idx] + testing_acc[idx])
        
    plt.plot(k_neighbors, training_acc, label= 'Training')
    plt.plot(k_neighbors, testing_acc,  label= 'Testing')
    plt.plot(k_neighbors, sum_acc,      label = 'Sum')
    plt.xlabel('K-neighbors')
    plt.ylabel('Accuracy_score')
    plt.legend();
    plt.show()

# Since the performances of KNN (~46%) are far below the performances of RF (~56%) and GB (~58%),
# the two models used will be GB for classifying the values, while RF to create new features.
model_RF  = RandomForestClassifier(    random_state=38, max_depth=13, n_estimators=139)
model_GB  = GradientBoostingClassifier(random_state=25, max_depth=10, n_estimators=85)

# GB is trained and tested. Its metrics are shown and the results saved in a confusion_matrix
model_GB.fit(X_train, y_train.values.ravel())
y_pred_GB = model_GB.predict(X_test)
metrics_GB = classification_report(y_test, y_pred_GB, output_dict=True)
results_GB = confusion_matrix(y_test, y_pred_GB)
print(f'GB Metrics: {metrics_GB}')

# RF is trained and tested. Its metrics are shown to the user
model_RF.fit(X_train, y_train.values.ravel())
y_pred_RF = model_RF.predict(X_test)
metrics_RF = classification_report(y_test, y_pred_RF, output_dict=True)
results_RF = confusion_matrix(y_test, y_pred_RF)
print(f'RF Metrics: {metrics_RF}')


############################### CONFUSION MATRIX ###############################
# Create the confusion matrix as a figure, we will later store it as a PNG image file.
# Due to the scarcity and low accuracy of Quality=9, the confusion matrix might have different dimensions.
if results_GB.shape[0] == 6:
    df_cm = pd.DataFrame(results_GB, ['True 3','True 4','True 5','True 6','True 7','True 8'], ['Pred 3','Pred 4','Pred 5','Pred 6','Pred 7','Pred 8'])
else: 
    df_cm = pd.DataFrame(results_GB, ['True 3','True 4','True 5','True 6','True 7','True 8','True 9'], ['Pred 3','Pred 4','Pred 5','Pred 6','Pred 7','Pred 8','Pred 9'])
cm = sns.heatmap(df_cm, annot=True, fmt=".2f")
plt.show()
fig = cm.get_figure()


############################### SAVING MODEL ON HOPSWORKS ###############################
# We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
mr = project.get_model_registry()

# Create two directories for the prediction and the creation model.
pred_model_dir="wine_model_pred"
if os.path.isdir(pred_model_dir) == False:
    os.mkdir(pred_model_dir)

create_model_dir="wine_model_feature_creator"
if os.path.isdir(create_model_dir) == False:
    os.mkdir(create_model_dir)

# Save both our models and the confusion matrix of the prediction model
fig.savefig(pred_model_dir + "/confusion_matrix.png")
joblib.dump(model_GB, pred_model_dir + "/wine_model_pred.pkl")
joblib.dump(model_RF, create_model_dir + "/wine_model_feature_creator.pkl")

# Specify the schema of the models' input/output using the features (X_train) and labels (y_train)
input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)

# Create an entry in the model registry for each of the two models
wine_model_pred = mr.python.create_model(
    name="wine_model_pred", 
    metrics={"accuracy" : metrics_GB['accuracy']},
    model_schema=model_schema,
    description="Wine Quality Predictor"
)

wine_model_feature_creator = mr.python.create_model(
    name="wine_model_feature_creator", 
    metrics={"accuracy" : metrics_RF['accuracy']},
    model_schema=model_schema,
    description="Wine Quality feature creator"
)

# Upload the models to the model registry, including all files in 'pred_model_dir'
wine_model_pred.save(pred_model_dir)
wine_model_feature_creator.save(create_model_dir)

