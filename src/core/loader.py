import os
import numpy as np

class Loader:
    def create_csv_for_california_housing(filepath):
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split

        housing = fetch_california_housing()
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data,
            housing.target.reshape(-1, 1),
            random_state=42
            )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full,
            y_train_full,
            random_state=42
            )

        train_data = np.c_[X_train, y_train]
        valid_data = np.c_[X_valid, y_valid]
        test_data = np.c_[X_test, y_test]
        header_cols = housing.feature_names + ["MedianHouseValue"]
        header = ",".join(header_cols)

        np.savetxt(os.path.join(filepath, 'train.csv'), train_data, delimiter=',', header=header)
        np.savetxt(os.path.join(filepath, 'valid.csv'), valid_data, delimiter=',', header=header)
        np.savetxt(os.path.join(filepath, 'test.csv'), test_data, delimiter=',', header=header)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())
    Loader.create_csv_for_california_housing('../../data')