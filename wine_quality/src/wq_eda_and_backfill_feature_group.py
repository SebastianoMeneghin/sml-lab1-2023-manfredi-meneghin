from turtle import clear
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import hopsworks
import os
import math


# Part of the code used to disable some parts:
HOPS_WORKLOAD = True
DATA_VISUAL   = False
VALIDATION    = False

if(HOPS_WORKLOAD):
    # Get hopsworks
    hopsworks_api_key = os.environ['HOPSWORKS_IRIS_APY_KEY']
    project = hopsworks.login(api_key_value = hopsworks_api_key)
    fs = project.get_feature_store()

# Retrieve the "original" wine dataset
wine_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/wine.csv")

# Rename the columns in order to make there a consistent key, remove duplicates
wine_df = wine_df.rename(columns={"fixed acidity": "fixed_acid",
                        "volatile acidity": "volatile_acid",
                        "citric acid": "citric_acid",
                        "residual sugar": "residual_sugar",
                        "free sulfur dioxide": "free_sd",
                        "total sulfur dioxide": "total_sd",
                        "pH":"ph"})

# Check the number and type of the values, as well as count of NaN values
print(f'\n{wine_df.info()}')
print(f'\n{wine_df.describe(include = "all")}')
print(f'\n{wine_df.isnull().sum()}')


########################### DATATYPE #############################
# We have 6497 features, described by 13 values.
#
# Categorical features are: type (ordinal), quality (nominal)
# Numerical   features are: (int) quality; (float) fixed acid, residual_sugar, 
#                           chlorides, free_sd, total_sd, density, ph,
#                           sulphates, alcohol, quality
# 
# There are some missing values for: fixed_acid, volatile_acid, citric_acid, residual_sugar,
#                                    chlorides, ph, sulphates


########################### DATA VISUALISATION #############################

if (DATA_VISUAL):
    # Distribution are printed, with quartiles
    fig, axes = plt.subplots(6, 2, figsize=(15, 10))
    quantiles = ['25%','50%', '75%']
    colors = ['green', 'red', 'blue']

    print_wine = wine_df.drop(columns='type')
    for col, ax in zip(print_wine, axes.flat):

        sns.histplot(ax=ax, data=print_wine, x=print_wine[col], multiple='stack')
        desc = print_wine[col].describe()

        for i in range(len(quantiles)):
            ax.axvline(desc[quantiles[i]], color=colors[i], ls='--', label = desc[quantiles[i]])
            ax.annotate(desc[quantiles[i]], (desc[quantiles[i]], -1), color=colors[i])


    # Then plotbars containing the relation between different values of
    # 'quality' and the other variables are shown
    plot_bar_df = wine_df.drop(columns=['quality','type'])
    columns = list(plot_bar_df.columns)

    for col in range(len(columns)):
        plot = plt.figure(figsize= (5,5))
        sns.barplot(x='quality' , y= columns[col] , data= wine_df)
        plt.xlabel('quality')
        plt.ylabel(columns[col])

        plt.show()


    # Violins plot are printed to show the present of outlines and quartiles division
    for col in range(len(columns)):
        plot = plt.figure(figsize= (5,5))
        g = sns.violinplot(x='quality' , y= columns[col] , data= wine_df, inner='quartile')
        #plt.ylabel('quality')
        #plt.xlabel(columns[col])
        plt.show()


    # Same things can be done with boxplots
    for name in columns:
        sns.boxplot(wine_df[name])
        plt.xlabel(name)
        plt.show()


    # Pairsplot are printed to understand linear dependencies among variables
    plt.figure(figsize=(12,12))
    g = sns.pairplot(wine_df, hue='quality', markers='+')
    plt.show()


    # The correlation matrix is create
    numeric_columns = wine_df.select_dtypes(include='number').columns
    numeric_df = wine_df.drop(columns=['type'])

    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

    plt.title('Correlation Matrix Heatmap')
    plt.show()


############################# DISTRIBUTION INFO #############################
# - Type can be only: white (75%), red (25%)
# - Density have very low standard deviation (0.3%), possible f-test
# - Chlorides, residal sugar and volatile acid have long tails, with few elements on them
# - Total sulfur dioxide and Free sulfur dioxide are highly correlated and have similar distribution among quality data
# - It's plenty of outliers that could be removed to make the model more stable


############################# DATA CLEANING #############################
# - Remove the outliners
# - Remove data where there's a missing value and duplicates
# - Remove the variable Total Sulfur Dioxide (will be done before to train them)
# - Make the datatype consistent (will be done before to train them)
# - Variables binning (will be done before to train them)

# Delete the missing values and drop duplicates
clean1_df = wine_df.dropna(axis=0, how="any").drop_duplicates().reset_index(drop=True)
clean1_df = clean1_df.rename_axis("key").reset_index()

# Remove outliners
clean_list = clean1_df.columns
to_remove = set()

for name in clean_list:
  if name != 'type' and name != 'quality' and name != 'key':
    Q1 = clean1_df[name].quantile(0.25)
    Q3 = clean1_df[name].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    # Create arrays of boolean values indicating the outlier rows
    upper_array = np.where(clean1_df[name]>=upper)[0]
    lower_array = np.where(clean1_df[name]<=lower)[0]

    to_remove.update(upper_array)
    to_remove.update(lower_array)

to_remove_list = list(to_remove)

# Removing the outliers
clean2_df = clean1_df.drop(labels=list(to_remove), axis = 0).drop(columns='key').reset_index(drop=True)
clean2_df = clean2_df.rename_axis("key").reset_index()

print(clean2_df.info())
print(clean2_df.describe())

samples_df = clean2_df.drop(columns = ['type', 'quality'])

################################ DATA BINNING AND REMOVAL ###############################
quantiles = [0.2, 0.4, 0.6, 0.8]
quant_div = len(quantiles)
column_div = ['fixed_acid','volatile_acid', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sd', 'density', 'ph', 'sulphates', 'alcohol']
print(column_div)

for col in column_div:
    quant_val = []
    for div in range(quant_div):
        quant_val.append(clean2_df[col].quantile(quantiles[div]))
        
    for row in range(clean2_df.shape[0]):
        cell = clean2_df.at[row,col]

        for div in range(quant_div):
            #print(row, col, div)

            low_quant = quant_val[div]
            if (div != quant_div - 1):
                up_quant = quant_val[div + 1]
            
            if (div == 0 and cell <= low_quant) or (div == quant_div - 1 and cell > up_quant) or ((cell > low_quant) and (cell <= up_quant)):
                    clean2_df.at[row,col] = div + 1

################################ DATA CONVERSION ###############################
# Assing number to label Red -> 1 and White -> 2
for row in range(clean2_df.shape[0]):
    wine_type = clean2_df.at[row, 'type']
    if wine_type == 'red':
        clean2_df.at[row, 'type'] = 1
    else:
        clean2_df.at[row, 'type'] = 2

# Convert columns to integers instead of floats or string
convert_column = ['type','fixed_acid','volatile_acid', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sd', 'density', 'ph', 'sulphates', 'alcohol']
to_upload_df = clean2_df.drop(columns = ['total_sd'])
for col in convert_column:
    to_upload_df = to_upload_df.astype({col: 'int64'})

################################ DATA INSERTION ################################
# Insert our WineQuality DataFrame into a FeatureGroup. 
if(HOPS_WORKLOAD):
    wine_fg_pred = fs.get_or_create_feature_group(
        name="wine_quality",
        version=1,
        primary_key=["key"], 
        description="Wine quality dataset")
    wine_fg_pred.insert(to_upload_df)

# Insert WineQuality samples before the data binning and removel, that will be used for feature creation
    wine_fg_pred = fs.get_or_create_feature_group(
        name="wine_samples",
        version=1,
        primary_key=["key"], 
        description="Wine samples dataset")
    wine_fg_pred.insert(samples_df)

print(to_upload_df.head())
print(to_upload_df.tail())


################################ DATA VALIDATION ################################
# Data validation rules can be defined and validated. Indeed, the following code
# prevents to write data into the iris dataset when values exceed expected ranges.

# This is not done now, due to some problem with the 
if (VALIDATION):
    from great_expectations.core import ExpectationSuite, ExpectationConfiguration

    def expect(suite, column, min_val, max_val):
        suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column":column, 
                "min_value":min_val,
                "max_value":max_val,
            }
        )
    )

    suite = ExpectationSuite(expectation_suite_name="wine_qualities")
    expect(suite, "fixed_acid", to_upload_df['fixed_acid'].min, to_upload_df['fixed_acid'].max)
    expect(suite, "volatile_acid", to_upload_df['volatile_acid'].min, to_upload_df['volatile_acid'].max)
    expect(suite, "citric_acid ", to_upload_df['citric_acid'].min, to_upload_df['citric_acid'].max)
    expect(suite, "residual_sugar", to_upload_df['residual_sugar'].min, to_upload_df['residual_sugar'].max)
    expect(suite, "chlorides", to_upload_df['chlorides'].min, to_upload_df['chlorides'].max)
    expect(suite, "free_sd", to_upload_df['free_sd'].min, to_upload_df['free_sd'].max)
    expect(suite, "density", to_upload_df['density'].min, to_upload_df['density'].max)
    expect(suite, "ph", to_upload_df['ph'].min, to_upload_df['ph'].max)
    expect(suite, "sulphates", to_upload_df['sulphates'].min, to_upload_df['sulphates'].max)
    expect(suite, "alcohol", to_upload_df['alcohol'].min, to_upload_df['alcohol'].max)
    wine_fg.save_expectation_suite(expectation_suite=suite, validation_ingestion_policy="STRICT")