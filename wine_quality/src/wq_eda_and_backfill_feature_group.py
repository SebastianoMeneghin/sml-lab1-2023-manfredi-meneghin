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
DATA_VISUAL = False

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
print(to_remove_list)

# Removing the outliers
clean2_df = clean1_df.drop(labels=list(to_remove), axis = 0).drop(columns='key').reset_index(drop=True)
clean2_df = clean2_df.rename_axis("key").reset_index()

clean2_df.info()
clean2_df.describe()


################################ DATA INSERTION ################################
# Insert our WineQuality DataFrame into a FeatureGroup. 

if(HOPS_WORKLOAD):
    wine_fg = fs.get_or_create_feature_group(
        name="wine",
        version=1,
        primary_key=["key"], 
        description="Wine quality dataset")
    wine_fg.insert(clean2_df)


################################ DATA VALIDATION ################################
# Data validation rules can be defined and validated. Indeed, the following code
# prevents to write data into the iris dataset when values exceed expected ranges.

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

# Due to some problems with the conversion, the data are not uploaded
#suite = ExpectationSuite(expectation_suite_name="wine_qualities")
#expect(suite, "fixed_acid", clean2_df['fixed_acid'].min, clean2_df['fixed_acid'].max)
#expect(suite, "volatile_acid", clean2_df['volatile_acid'].min, clean2_df['volatile_acid'].max)
#expect(suite, "citric_acid ", clean2_df['citric_acid'].min, clean2_df['citric_acid'].max)
#expect(suite, "residual_sugar", clean2_df['residual_sugar'].min, clean2_df['residual_sugar'].max)
#expect(suite, "chlorides", clean2_df['chlorides'].min, clean2_df['chlorides'].max)
#expect(suite, "free_sd", clean2_df['free_sd'].min, clean2_df['free_sd'].max)
#expect(suite, "total_sd", clean2_df['total_sd'].min, clean2_df['total_sd'].max)
#expect(suite, "density", clean2_df['density'].min, clean2_df['density'].max)
#expect(suite, "ph", clean2_df['ph'].min, clean2_df['ph'].max)
#expect(suite, "sulphates", clean2_df['sulphates'].min, clean2_df['sulphates'].max)
#expect(suite, "alcohol", clean2_df['alcohol'].min, clean2_df['alcohol'].max)
#wine_fg.save_expectation_suite(expectation_suite=suite, validation_ingestion_policy="STRICT")