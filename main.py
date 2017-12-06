import preprocess_data as preprocessor

def main():
    # data, numerical_data, test_data, validation_data, training_data = preprocessor.preprocess_data("Data", "CleanData", 10000)
    # preprocessor.select_data_sets(data, numerical_data, "CleanData", 98)

    preprocessor.preprocess_data_random("Data", "CleanData", 10000, 80)

main()
