import preprocess_data as preprocessor
import group_data as group

def main():
    data, test_data = preprocessor.preprocess_data("Data", "CleanData", 100)
    preprocessor.select_data_sets(data, "CleanData", 99)

main()
