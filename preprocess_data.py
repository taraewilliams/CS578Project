import pandas as pd
from random import randint
import os


def preprocess_data(read_directory, write_directory, test_percentage, validation_percentage, convert_categorical=False, categorical_columns=["Weekday", "DepartmentDescription"], sort_label="TripType"):

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

    # Split data into test, training, and validation data
    test_data, remaining_data = split_data(data, test_percentage, sort_label)
    validation_data, training_data = split_data(remaining_data, validation_percentage, sort_label)
    # Write test, training, and validation data to CSV files
    write_data(write_directory + '/train.csv', training_data)
    write_data(write_directory + '/test.csv', test_data)
    write_data(write_directory + '/validation.csv', validation_data)

    # Called if converting categories to numerical values
    if convert_categorical:
        # Convert all data to numerical data
        numerical_data = convert_categorical_columns(data, categorical_columns)
        # Split data into test, training, and validation data
        training_data_numerical = numerical_data.loc[numerical_data.index.isin(training_data.index)]
        test_data_numerical = numerical_data.loc[numerical_data.index.isin(test_data.index)]
        validation_data_numerical = numerical_data.loc[numerical_data.index.isin(validation_data.index)]
        # Write all data, and test, training, and validation data to CSV files
        write_data(write_directory + '/data_numerical.csv', numerical_data)
        write_data(write_directory + '/train_numerical.csv', training_data_numerical)
        write_data(write_directory + '/test_numerical.csv', test_data_numerical)
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

        # Choose training data
        training_sample = group.sample(frac=train_percentage)
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
