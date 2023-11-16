import hopsworks
import joblib
import os
import pandas  as pd
import seaborn as sns
from   sklearn.neighbors import KNeighborsClassifier
from   sklearn.metrics   import confusion_matrix
from   sklearn.metrics   import classification_report
from   hsml.schema       import Schema
from   hsml.model_schema import ModelSchema
from   matplotlib        import pyplot as plt


# Here the environment variable is set
hopsworks_iris_api_key = os.environ['HOPSWORKS_IRIS_APY_KEY']
project = hopsworks.login(api_key_value = hopsworks_iris_api_key)
fs = project.get_feature_store()

# The feature view is the input set of features for your model. The features can come from different feature groups.    
# You can select features from different feature groups and join them together to create a feature view
iris_fg = fs.get_feature_group(name="iris", version=1)
query = iris_fg.select_all()
feature_view = fs.get_or_create_feature_view(name="iris",
                                  version=1,
                                  description="Read from Iris flower dataset",
                                  labels=["variety"],
                                  query=query)



############################### MODEL TRAINING ###############################
# You can read training data, randomly split into train/test sets of features (X) and labels (y)        
X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)

# Train our model with the Scikit-learn K-nearest-neighbors algorithm using our features (X_train) and labels (y_train)
# "ravel" is used to flatten a dataframe into a one-dimension numpy array
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train.values.ravel())

print(X_train)
print(y_train)

# Evaluate model performance using the features from the test set (X_test)
y_pred = model.predict(X_test)

# Compare predictions (y_pred) with the labels in the test set (y_test)
metrics = classification_report(y_test, y_pred, output_dict=True)
results = confusion_matrix(y_test, y_pred)

# Show then the metrics
print(metrics)


############################### CONFUSION MATRIX ###############################
# Create the confusion matrix as a figure, we will later store it as a PNG image file
df_cm = pd.DataFrame(results, ['True Setosa', 'True Versicolor', 'True Virginica'], ['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])
cm = sns.heatmap(df_cm, annot=True, fmt=".2f")
plt.show()
fig = cm.get_figure()



############################### SAVING MODEL ON HOPSWORKS ###############################
# We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
mr = project.get_model_registry()

# The contents of the 'iris_model' directory will be saved to the model registry. Create the dir, first.
model_dir="iris_model"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)

# Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
joblib.dump(model, model_dir + "/iris_model.pkl")
fig.savefig(model_dir + "/confusion_matrix.png")    

# Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)

# Create an entry in the model registry that includes the model's name, desc, metrics
iris_model = mr.python.create_model(
    name="iris_model", 
    metrics={"accuracy" : metrics['accuracy']},
    model_schema=model_schema,
    description="Iris Flower Predictor"
)

# Upload the model to the model registry, including all files in 'model_dir'
iris_model.save(model_dir)

