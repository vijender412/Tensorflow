from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == '__main__':
    # dataset github.com/zalandoresearch/fashion-mnist
    url = 'https://storage.googleapis.com/acg-datasets/fmnist_test.tgz'
    cache_dir = '..'
    cache_subdir = 'data'

    keras.utils.get_file('fmnist_test.tqz', url, extract=True,
                         cache_dir=cache_dir, cache_subdir=cache_subdir) # keras will insure to not download the data again

    extract_path = f'{cache_dir}/{cache_subdir}/fmnist_test'

    # To display sample image
    # class_zero_images = os.listdir(f'{extract_path}/0')
    # im = Image.open(f'{extract_path}/0/{class_zero_images[0]}')
    # plt.imshow(im, cmap='Greys')

    # image transformation
    test_preprocess = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255
    )

    test_generator = test_preprocess.flow_from_directory(
        extract_path,
        target_size=(28,28),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=2,
        shuffle=True,
        seed=42
    )

    test_batch = test_generator.next()
