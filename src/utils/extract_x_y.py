import numpy as np
import pandas as pd


def extract_x_and_y(input_file):
    with input_file.open('r') as f:
        df = pd.read_csv(f, dtype=np.int16)

    pixel_features = df.columns[df.columns.str.contains('pixel')]
    return df[pixel_features], df['label']
