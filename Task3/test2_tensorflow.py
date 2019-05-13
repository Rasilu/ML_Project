import numpy as np
import pandas as pd
import tensorflow as tf

tf.enable_eager_execution()

# read file with pandas
training_df = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")

features = ['feature1', 'feature2', 'feature3']
print(training_df)

training_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(training_df[features].values, tf.float32),
            tf.cast(training_df['target'].values, tf.int32)
        )
    )
)

for features_tensor, target_tensor in training_dataset:
    print(f'features:{features_tensor} target:{target_tensor}')