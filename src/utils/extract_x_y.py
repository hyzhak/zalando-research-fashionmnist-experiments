import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

img_rows = 28
img_cols = 28


def extract_x_and_y(input_file):
    with input_file.open('r') as f:
        df = pd.read_csv(f, dtype=np.int16)

    pixel_features = df.columns[df.columns.str.contains('pixel')]
    return df[pixel_features].values, df['label'].values


def reshape_X_to_2d(X):
    return X.reshape(X.shape[0], img_rows, img_cols, 1)


def get_train_valid_test_subsets(train_size, valid_size, random_seed, train_file, test_file):
    if train_size is not None:
        train_size = float(train_size)
        if train_size >= 1.0:
            train_size = int(train_size)

        X_train, X_valid, y_train, y_valid = train_test_split(
            *extract_x_and_y(train_file),
            train_size=train_size,
            random_state=random_seed,
        )
    else:
        if valid_size >= 1.0:
            valid_size = int(valid_size)

        X_train, X_valid, y_train, y_valid = train_test_split(
            *extract_x_and_y(train_file),
            test_size=valid_size,
            random_state=random_seed,
        )

    X_test, y_test = extract_x_and_y(test_file)
    return X_train, X_valid, X_test, y_train, y_valid, y_test
