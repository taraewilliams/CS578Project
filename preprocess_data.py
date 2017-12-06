import pandas as pd
from random import randint
import os
import math


def preprocess_data_random(read_directory, write_directory, number_test_samples, validation_percentage, sort_label="TripType"):
    # Make write directory if it does not exist
    if not os.path.isdir(write_directory):
        try:
            os.makedirs(write_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # Clean all data
    data = read_data(read_directory + "/grouped_data.csv")
    data = clean_data(data, sort_label)
    write_data(write_directory + '/grouped_data.csv', data)

    # Get random test, training, and validation data
    test_random, data = get_random_samples(data, number_test_samples)
    training_random, data = get_random_samples(data, number_test_samples)
    validation_random, data = get_random_samples(data, number_test_samples/2)
    # Write test, training, and validation data to CSV files
    write_data(write_directory + '/train_random.csv', training_random)
    write_data(write_directory + '/test_random.csv', test_random)
    write_data(write_directory + '/validation_random.csv', validation_random)


    # Split data into training and validation data
    validation_data, training_data, unused1, unused2 = split_data(data, validation_percentage, sort_label)

    # Write test, training, and validation data to CSV files
    write_data(write_directory + '/train.csv', training_data)
    write_data(write_directory + '/remaining.csv', validation_data)


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
    data, numerical_data, test_groups, test_group_data = group_test_data(data, numerical_data, group_label, number_test_samples)
    data, numerical_data, validation_groups, validation_group_data = group_test_data(data, numerical_data, group_label, (number_test_samples/2) )
    data, numerical_data, training_groups, training_group_data = group_test_data(data, numerical_data, group_label, (number_test_samples) )

    # Write grouped data to CSV files
    write_data(write_directory + '/group_test.csv', test_group_data)
    write_data(write_directory + '/group_validation.csv', validation_group_data)
    write_data(write_directory + '/group_train.csv', training_group_data)

    return data, numerical_data, test_groups, validation_groups, training_groups


def select_data_sets(data, numerical_data, write_directory, validation_percentage, convert_categorical=True, categorical_columns=["Weekday", "DepartmentDescription"], sort_label="TripType"):

    # Split data into training and validation data
    validation_data, training_data, validation_data_numerical, training_data_numerical = split_data(data, validation_percentage, sort_label, True, numerical_data)

    # Write test, training, and validation data to CSV files
    write_data(write_directory + '/train.csv', training_data)
    write_data(write_directory + '/validation.csv', validation_data)

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
def group_test_data(data, numerical_data, label, number_samples):
    if (number_samples >= numerical_data.shape[0]):
        number_samples = numerical_data.shape[0]

    visits = numerical_data[label].unique().tolist()

    group_data = pd.DataFrame([], columns=list(data))
    groups = []
    group_indices = []
    group_values = []
    total_size = 0
    is_full = False

    while not is_full:
        visit_index = randint(0, len(visits)-1)

        if (visit_index not in group_indices):
            visit = visits[visit_index]
            group = numerical_data.loc[data[label]==visit]

            total_size += group.shape[0]
            if (total_size > number_samples):
                is_full = True
            else:
                groups.append(group.values)
                group_indices.append(visit_index)
                group_values.append(visit)
                group_data = group_data.append(group, ignore_index=True)

    data = data[~data[label].isin(group_values)]
    numerical_data = numerical_data[~numerical_data[label].isin(group_values)]
    return data, numerical_data, groups, group_data


def get_random_samples(data, sample_size):
    sample_data = data.sample(n=int(sample_size))
    data = data.loc[~data.index.isin(sample_data.index)]
    return sample_data, data


# Convert categorical columns to numerical columns
def convert_categorical_columns(data, category_labels):
    numerical_data = convert_categorical_to_numerical(data.copy(), category_labels)
    return numerical_data


# Split Data:
#   Split the data by the input training percentage
#   Label determines how to group the data: ie TripType
def split_data(data, train_percentage, label, convert_categorical=False, numerical_data = []):
    training_data = pd.DataFrame([], columns=list(data))
    test_data = pd.DataFrame([], columns=list(data))
    num_training_data = pd.DataFrame([], columns=list(data))
    num_test_data = pd.DataFrame([], columns=list(data))

    trip_types = data[label].unique().tolist()

    for trip in trip_types:
        group = data.loc[data[label]==trip]

        number_samples = math.floor((train_percentage * group.shape[0]) / 100)
        if (number_samples == group.shape[0]):
            number_samples -= 1

        # Choose training data
        training_sample = group.sample(n=number_samples)
        training_data = training_data.append(training_sample, ignore_index=True)

        # Choose test data
        test_sample = group.loc[~group.index.isin(training_sample.index)]
        test_data = test_data.append(test_sample, ignore_index=True)

        if convert_categorical:
            group_num = numerical_data.loc[data[label]==trip]
            training_sample_num = group_num.sample(n=number_samples)
            test_sample_num = group_num.loc[~group_num.index.isin(training_sample.index)]
            num_test_data = num_test_data.append(test_sample_num, ignore_index=True)
            num_training_data = num_training_data.append(training_sample_num, ignore_index=True)

    return training_data, test_data, num_training_data, num_test_data


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
