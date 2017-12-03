import pandas as pd
from random import randint
import os
import math


def preprocess_data(read_directory, write_directory, number_test_samples, sort_label="TripType", group_label="VisitNumber", convert_categorical=True, categorical_columns=["Weekday", "DepartmentDescription"]):

    # Make write directory if it does not exist
    if not os.path.isdir(write_directory):
        try:
            os.makedirs(write_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # Clean all data
    data = read_data(read_directory + "/train.csv")
    data = clean_data(data, sort_label)
    write_data(write_directory + '/data.csv', data)

    # Convert data to categorical data
    if convert_categorical:
        numerical_data = convert_categorical_columns(data, categorical_columns)
        write_data(write_directory + '/data_numerical.csv', numerical_data)


    # Get Grouped Test Data
    data, groups = group_test_data(data, group_label, number_test_samples)
    return data, groups


def select_data_sets(data, write_directory, validation_percentage, convert_categorical=True, categorical_columns=["Weekday", "DepartmentDescription"], sort_label="TripType"):

    # Split data into training and validation data
    validation_data, training_data = split_data(data, validation_percentage, sort_label)

    # Write test, training, and validation data to CSV files
    write_data(write_directory + '/train.csv', training_data)
    write_data(write_directory + '/validation.csv', validation_data)

    # Called if converting categories to numerical values
    if convert_categorical:
        numerical_data = read_data(write_directory + "/data_numerical.csv")
        # Split data into training and validation data
        training_data_numerical = numerical_data.loc[numerical_data.index.isin(training_data.index)]
        validation_data_numerical = numerical_data.loc[numerical_data.index.isin(validation_data.index)]
        # Write training and validation data to CSV files
        write_data(write_directory + '/train_numerical.csv', training_data_numerical)
        write_data(write_directory + '/validation_numerical.csv', validation_data_numerical)


# Read CSV file into Pandas DataFrame
def read_data(file_name):
    return pd.read_csv(file_name)


# Write data into CSV file
def write_data(file_name, data):
    data.to_csv(file_name, index=False)


# Clean Data:
#   Order the data
#   Convert categorical columns to numerical ones
#   Remove null values
#   Write the data to a new CSV file
def clean_data(data, order_label):
    data = remove_null_values(data)
    data = order_data_by_label(data, order_label)
    return data


# Group Test Data:
#   Get test data that is grouped by a column value
#   i.e. select random number of groups with shared column value VisitNumber
def group_test_data(data, label, number_samples):
    if (number_samples >= data.shape[0]):
        number_samples = data.shape[0]

    visits = data[label].unique().tolist()

    groups = []
    group_indices = []
    group_values = []

    while len(groups) < number_samples:
        visit_index = randint(0, len(visits))

        if (visit_index not in group_indices):
            visit = visits[visit_index]

            group_indices.append(visit_index)
            group_values.append(visit)

            group = data.loc[data[label]==visit]
            groups.append(group.values)

    data = data[~data[label].isin(group_values)]
    return data, groups


# Convert categorical columns to numerical columns
def convert_categorical_columns(data, category_labels):
    numerical_data = convert_categorical_to_numerical(data.copy(), category_labels)
    return numerical_data


# Split Data:
#   Split the data by the input training percentage
#   Label determines how to group the data: ie TripType
def split_data(data, train_percentage, label):
    training_data = pd.DataFrame([], columns=list(data))
    test_data = pd.DataFrame([], columns=list(data))

    trip_types = data[label].unique().tolist()

    for trip in trip_types:
        group = data.loc[data[label]==trip]

        number_samples = math.floor((train_percentage * group.shape[0]) / 100)

        # Choose training data
        training_sample = group.sample(n=number_samples)
        training_data = training_data.append(training_sample, ignore_index=True)
        # Choose test data
        test_sample = group.loc[~group.index.isin(training_sample.index)]
        test_data = test_data.append(test_sample, ignore_index=True)

    return training_data, test_data


# Private Functions

# Convert categorical columns to numerical columns
def convert_categorical_to_numerical(data, features):
    categories = []
    for feature in features:
        category = {"count":0, "name":feature, "categories":{}}
        categories.append(category)

    for index, row in data.iterrows():
        for category in categories:
            if row[category["name"]] not in category["categories"]:
                category["count"] = category["count"] + 1
                category["categories"][row[category["name"]]] = category["count"]

            row[category["name"]] = category["categories"][row[category["name"]]]
            data.set_value(index, category["name"], row[category["name"]])

    return data


# Order data by column
def order_data_by_label(data, label):
    return data.sort_values(by=label)


# Remove null values from data
def remove_null_values(data):
    data = data.dropna(how='any')
    data = data.reset_index(drop=True)
    return data
