import numpy as np
import pandas as pd

img_rows = 28
img_cols = 28


def extract_x_and_y(input_file):
    with input_file.open('r') as f:
        df = pd.read_csv(f, dtype=np.int16)

    pixel_features = df.columns[df.columns.str.contains('pixel')]
    return df[pixel_features].values, df['label'].values


def reshape_X_to_2d(X):
    return X.reshape(X.shape[0], img_rows, img_cols, 1)
