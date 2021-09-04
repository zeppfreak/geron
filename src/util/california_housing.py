import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CaliforniaHousing:
    def __init__(self):
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

        # preprocess 
        pipeline = Pipeline(steps=[
            ("standard_scaler", StandardScaler())
        ], verbose=True)

        X_train = pipeline.fit_transform(X_train)
        X_valid = pipeline.fit_transform(X_valid)
        X_test = pipeline.fit_transform(X_test) 

        # merge X and y
        self.train_data = np.c_[X_train, y_train]
        self.valid_data = np.c_[X_valid, y_valid]
        self.test_data = np.c_[X_test, y_test]
        header_cols = housing.feature_names + ["MedianHouseValue"]
        self.header = ",".join(header_cols)
    
    def create_csv(self, filepath):
        os.makedirs(filepath, exist_ok=True) 
        np.savetxt(os.path.join(filepath, 'train.csv'), self.train_data, delimiter=',', header=self.header)
        np.savetxt(os.path.join(filepath, 'valid.csv'), self.valid_data, delimiter=',', header=self.header)
        np.savetxt(os.path.join(filepath, 'test.csv'), self.test_data, delimiter=',', header=self.header)
    
    def create_multiple_csv(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        train_filepaths = self._save_to_multiple_csv_files(filepath, self.train_data, "train", self.header, n_parts=20)
        valid_filepaths = self._save_to_multiple_csv_files(filepath, self.valid_data, "valid", self.header, n_parts=10)
        test_filepaths = self._save_to_multiple_csv_files(filepath, self.test_data, "test", self.header, n_parts=10)
    
    def _save_to_multiple_csv_files(self, filepath, data, name_prefix, header=None, n_parts=10):
        path_format = os.path.join(filepath, "{}_{:02d}.csv")

        filepaths = []
        m = len(data)
        for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
            part_csv = path_format.format(name_prefix, file_idx)
            filepaths.append(part_csv)
            with open(part_csv, "wt", encoding="utf-8") as f:
                if header is not None:
                    f.write(header)
                    f.write("\n")
                for row_idx in row_indices:
                    f.write(",".join([repr(col) for col in data[row_idx]]))
                    f.write("\n")
        return filepaths

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())
    ch = CaliforniaHousing()
    ch.create_csv('../../data')
    ch.create_multiple_csv('../../data/multiple')