import os
import tensorflow as tf

# This notebook show use of shuffle, batch, cache and prefetch operations of tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def printds(ds, quantity=5):
    for example in ds.take(quantity):
        print(example)

data = list(range(100))
ds = tf.data.Dataset.from_tensor_slices((data))
printds(ds)

shuffled = ds.shuffle(buffer_size=10)
printds(shuffled)

shuffled = ds.shuffle(buffer_size=50)
printds(shuffled)

# larger buffer is better for randomized

batched = ds.batch(12, drop_remainder=True)
printds(batched)

batched = ds.batch(10)
printds(batched)

shuffle_batch = shuffled.batch(5)
printds(shuffle_batch)

# final the order matters
shuffle_batch =  ds.shuffle(10).batch(10)
printds(shuffle_batch)


cached = shuffle_batch.cache()
printds(cached)

printds(cached,10) #now its actually cached and no randomized now

prefetch = ds.prefetch(tf.data.AUTOTUNE)
printds(prefetch)

