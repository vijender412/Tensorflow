import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ds_size = 1000
ds = tf.data.Dataset.from_tensor_slices(np.arange(ds_size))
ds = ds.shuffle(ds_size, seed=42, reshuffle_each_iteration=False).batch(15)

ds_np = list(ds.as_numpy_iterator())
print('Original Data')
for batch in ds_np[:5]:
    print(batch)

path = '../data'
tfrecord_filename = f'{ds_size}.tfrecord'
ds_out = ds.map(tf.io.serialize_tensor)

writer = tf.data.experimental.TFRecordWriter(f'{path}/{tfrecord_filename}')
writer.write(ds_out)

new_ds = tf.data.TFRecordDataset(f'{path}/{tfrecord_filename}')
new_ds = new_ds.map(lambda x: tf.io.parse_tensor(x, tf.int32))

new_ds_np = list(new_ds.as_numpy_iterator())
print('Loaded Data')
for batch in new_ds_np[:5]:
    print(batch)