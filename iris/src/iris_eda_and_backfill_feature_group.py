import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
import hopsworks
import pandas as pd
import os

hopsworks_iris_api_key = os.environ['HOPSWORKS_IRIS_APY_KEY']
#hopsworks_iris_api_key = 'bP6PFOAzbXllM89C.l9gzwKTyxct786c3V1gwIQEvbQfZnSELxp7UM4RdBhw0eTaqkdl1Ld2a4A32UmR9'
project = hopsworks.login(api_key_value = hopsworks_iris_api_key)
fs = project.get_feature_store()

# Retrieve the online dataset of Iris
iris_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv")

# Print the characteristic of the values of the dataset (name, count, type)
print('\n')
iris_df.info()

# Display summary statistics of the dataframe, including count, mean, standard deviation, minimum, 25th percentile, 
# median, 75th percentile, and maximum values for each numeric column.
print(f'\n{iris_df.describe()}')

# Display the variety information
print(f'\n{iris_df["variety"].value_counts()}')



########################## EXPLORATORY DATA ANALYSIS ##########################
# Data of the dataset are visually displayed and statistically analyzed to 
# summarize its main characteristics, identify patterns, and uncover insights.

# We look at the distribution and range of values for the 4 different features, comparing them
g = sns.pairplot(iris_df, hue='variety', markers='+')
plt.show()

# We can now visualize the range of values for the length and width of the sepal and petal for each of the 3 flowers,
# with their quartiles indicated
g = sns.violinplot(y='variety', x='sepal_length', data=iris_df, inner='quartile')
plt.show()
g = sns.violinplot(y='variety', x='sepal_width', data=iris_df, inner='quartile')
plt.show()
g = sns.violinplot(y='variety', x='petal_length', data=iris_df, inner='quartile')
plt.show()
g = sns.violinplot(y='variety', x='petal_width', data=iris_df, inner='quartile')
plt.show()


################################ DATA INSERTION ################################
# Insert our Iris DataFrame into a FeatureGroup. 
# Let's write our historical iris feature values and labels to a feature group. 
# Since they are historical data, this process is called backfilling.

iris_fg = fs.get_or_create_feature_group(
    name="iris",
    version=1,
    primary_key=["sepal_length","sepal_width","petal_length","petal_width"], 
    description="Iris flower dataset")
iris_fg.insert(iris_df)


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
    
suite = ExpectationSuite(expectation_suite_name="iris_dimensions")

expect(suite, "sepal_length", 4.5, 8.0)
expect(suite, "sepal_width", 2.1, 4.5)
expect(suite, "petal_length", 1.2, 7)
expect(suite, "petal_width", 0.2, 2.5)
iris_fg.save_expectation_suite(expectation_suite=suite, validation_ingestion_policy="STRICT")  