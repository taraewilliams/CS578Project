import random
import csv
import pandas as pd
from random import randint


def preprocess_data():
    # data = read_data("Data/train.csv")
    # data = clean_data(data, "TripType", ["Weekday", "DepartmentDescription"])
    data = read_data("CleanData/data.csv")
    data = split_data(data, 0.9, 'TripType')


# Read CSV file into Pandas DataFrame
def read_data(file_name):
    return pd.read_csv(file_name)


# Clean Data:
#   Order the data
#   Convert categorical columns to numerical ones
#   Remove null values
#   Write the data to a new CSV file
def clean_data(data, order_label, category_labels):
    data = remove_null_values(data)
    data = order_data_by_label(data, order_label)
    data = convert_categorical_to_numerical(data, category_labels)
    data.to_csv('CleanData/data.csv', index=False)
    return data


# Split Data:
#   Split the data by the input training percentage
#   Label determines how to group the data: ie TripType
def split_data(data, train_percentage, label):
    training_data = pd.DataFrame([], columns=list(data))
    test_data = pd.DataFrame([], columns=list(data))

    trip_types = data['TripType'].unique().tolist()

    for trip in trip_types:
        group = data.loc[data.TripType==trip]

        # Choose training data
        training_sample = group.sample(frac=train_percentage)
        training_data = training_data.append(training_sample, ignore_index=True)
        # Choose test data
        test_sample = group.loc[~group.index.isin(training_sample.index)]
        test_data = test_data.append(test_sample, ignore_index=True)

    training_data.to_csv('CleanData/train.csv', index=False)
    test_data.to_csv('CleanData/test.csv', index=False)
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
