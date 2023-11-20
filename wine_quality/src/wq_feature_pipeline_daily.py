import os
import modal
import joblib

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_10mins")
   image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "seaborn","scikit-learn==1.1.1","dataframe-image","Pillow"]) 

   @stub.function(cpu=1.0, image=image, schedule=modal.Period(minutes=10), secret=modal.Secret.from_name("hopsworks_iris_api"))
   def f():
       g()


def generate_wine(values):
    """
    Returns a single wine feature as a single row in a DataFrame
    """
    import hopsworks
    import pandas as pd
    import random
    import os
    import json
    import joblib

    # Picks one casual type of wine, according to the type-distribution
    pick_random = random.uniform(0,3)
    random_type = ''
    if pick_random >= 2:
        random_type = 'red'
    else:
        random_type = 'white'

    new_val = {}
    new_val['type'] = random_type
    new_val.update(values)

    row_df = pd.DataFrame([new_val])
    
    columns_order = ['type', 'fixed_acid', 'volatile_acid', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sd', 'total_sd', 'density', 'ph', 'sulphates', 'alcohol']
    ordered_df = row_df[columns_order]

    return ordered_df


def get_wine_stats(wine_fg):
    '''
    Returns the statistics of the data taken from the Wine Samples feature group
    '''
    import hopsworks
    import pandas as pd
    import random
    import os
    import json
    import joblib

    # Get the statistics of the wine samples feature group
    wine_stat = wine_fg.get_statistics(commit_time=None)
    wine_stat_dict = wine_stat.to_dict()

    # Extract the content from the object and parse it a JSON
    content_str = wine_stat_dict['content']
    content_json = json.loads(content_str)

    # Initialize vectors    
    mean_vector = []
    min_vector = []
    max_vector = []
    std_vector = []
    column_vector = []

    # Extract information for each column
    for column_info in content_json['columns']:
        mean_vector.append(column_info['mean'])
        min_vector.append(column_info['minimum'])
        max_vector.append(column_info['maximum'])
        std_vector.append(column_info['stdDev'])
        column_vector.append(column_info['column'])

    stats_dict = {}

    # Iterate through the vectors and populate the dictionary
    for i in range(len(column_vector)):
        stats_dict[column_vector[i]] = {
            'mean': mean_vector[i],
            'min': min_vector[i],
            'max': max_vector[i],
            'std': std_vector[i]
        }

    return stats_dict


def create_random_qualities(stats):
    '''
    Computes random qualities provided the statistics on them
    '''
    import hopsworks
    import pandas as pd
    import random
    import os
    import json
    import joblib

    # Remove the key, since it's not needed in the new feature
    stats.pop("key")

    values = {}
    for var in stats:
        a = stats[var]['min']
        b = stats[var]['max']
        c = stats[var]['mean']
        d = stats[var]['std'] 

        value = abs(c + d * random.uniform(-2.1,2.1) + ((a-b)/c) * random.uniform(-0.3,0.3))
        values[var] = value

    return values

def normalize_wine(new_wine_df, project, row_df):
    '''
    Normalizes the wine to make it readable and processable by the ML system
    '''
    import hopsworks
    import pandas as pd
    import random
    import os
    import json
    import joblib

    stat_fs = project.get_feature_store()
    stat_fg = stat_fs.get_feature_group(name="wine_samples", version=1)
    query = stat_fg.select_all()
    feature_view = stat_fs.get_or_create_feature_view(name="wine_samples",
                                    version=1,
                                    description="Samples of wines",
                                    labels=["key"],
                                    query=query)

    stats_df, placeholder1, placeholder2, placeholder3 = feature_view.train_test_split(0.0001)

    new_key = stats_df.shape[0]
    quantiles = [0.2, 0.4, 0.6, 0.8]
    quant_div = len(quantiles)
    column_div = ['fixed_acid','volatile_acid', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sd', 'density', 'ph', 'sulphates', 'alcohol']

    # Numbers are converted to be processed by the ML models trained in the training pipeline
    for col in column_div:
        quant_val = []
        for div in range(quant_div):
            quant_val.append(stats_df[col].quantile(quantiles[div]))
            
        cell = row_df.at[0,col]

        for div in range(quant_div):
            low_quant = quant_val[div]
            if (div != quant_div - 1):
                up_quant = quant_val[div + 1]
            
            if (div == 0 and cell <= low_quant) or (div == quant_div - 1 and cell > up_quant) or ((cell > low_quant) and (cell <= up_quant)):
                    row_df.at[0,col] = div + 1


    # Assing number to label Red -> 1 and White -> 2
    wine_type = row_df.at[0, 'type']
    if wine_type == 'red':
        row_df.at[0, 'type'] = 1
    else:
        row_df.at[0, 'type'] = 2

    # Add the successive key to the new feature 
    row_df['key'] = new_key

    # Convert columns to integers instead of floats or string
    convert_column = ['key','type','fixed_acid','volatile_acid', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sd', 'density', 'ph', 'sulphates', 'alcohol']
    to_upload_df = row_df.drop(columns = ['total_sd'])
    for col in convert_column:
        to_upload_df = to_upload_df.astype({col: 'int64'})

    return to_upload_df


def predict_new_label(normalized_df, project):
    '''
    Predicts the label for the new feature
    '''
    import hopsworks
    import pandas as pd
    import random
    import os
    import json
    import joblib

    feature_df = normalized_df.drop(columns = 'key')

    # Download the pre-trained model and load it
    mr = project.get_model_registry()
    model = mr.get_model("wine_model_feature_creator", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model_feature_creator.pkl")

    # Get the predictions of the model
    y_pred = model.predict(feature_df)
    label = y_pred.astype(int)
    return label



def get_random_wine(wine_fg, project):
    """
    Returns a DataFrame containing one random iris flower
    """
    import hopsworks
    import pandas as pd
    import random
    import os
    import json
    import joblib

    stats  = get_wine_stats(wine_fg)
    values = create_random_qualities(stats)
    new_wine_df = generate_wine(values)
    normalized_df = normalize_wine(new_wine_df, project, new_wine_df)
    new_label = predict_new_label(normalized_df, project)
    
    normalized_df['quality'] = new_label

    print("Your new feature is:\n", normalized_df.head())

    return normalized_df


def g():
    import hopsworks
    import pandas as pd
    import random
    import os
    import json
    import joblib

    # Connect to hopsworks and get the feature_group metadata
    hopsworks_api_key= os.environ["HOPS_LAB1_IRIS_KEY"]
    project = hopsworks.login(api_key_value = hopsworks_api_key)
    fs = project.get_feature_store()

    wine_fg = fs.get_feature_group(name="wine_samples", version=1)

    # Create a new feature and save it as a dataframe with a single row
    wine_df = get_random_wine(wine_fg, project)

    # Then, insert the new feature just created
    wine_quality_fg = fs.get_feature_group(name="wine_quality", version=1)
    wine_quality_fg.insert(wine_df)