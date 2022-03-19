import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
# from history import plot_history, save_history

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def retrieve_data():
    train_url = 'https://storage.googleapis.com/ibeans/train.zip'
    valid_url = 'https://storage.googleapis.com/ibeans/validation.zip'
    test_url = 'https://storage.googleapis.com/ibeans/test.zip'
    cache_dir = '../data'
    cache_subdir = 'beans'

    tf.keras.utils.get_file('train.zip', train_url, extract=True, cache_dir=cache_dir, cache_subdir=cache_subdir)
    tf.keras.utils.get_file('valid.zip', valid_url, extract=True, cache_dir=cache_dir, cache_subdir=cache_subdir)
    tf.keras.utils.get_file('test.zip', test_url, extract=True, cache_dir=cache_dir, cache_subdir=cache_subdir)

    return f'{cache_dir}/{cache_subdir}'

def wrangle_data(data_dir):
    scale_factor = 1 / 255.0
    train_datagen = ImageDataGenerator(rescale=scale_factor,
                                       horizontal_flip=True,
                                       zoom_range=0.2,
                                       rotation_range=30,
                                       fill_mode='nearest'
                                       )
    valid_datagen = ImageDataGenerator(rescale=scale_factor)
    test_datagen = ImageDataGenerator(rescale=scale_factor)

    image_size = (500, 500)
    encoding = 'categorical'

    training_data = train_datagen.flow_from_directory(f'{data_dir}/train',
                                                      target_size=image_size,
                                                      class_mode=encoding,
                                                      shuffle=True)
    validation_data = valid_datagen.flow_from_directory(f'{data_dir}/validation',
                                                        target_size=image_size,
                                                        class_mode=encoding,
                                                        shuffle=False)
    testing_data = valid_datagen.flow_from_directory(f'{data_dir}/test',
                                                     target_size=image_size,
                                                     class_mode=encoding,
                                                     shuffle=False)

    return training_data, validation_data, testing_data

def compile_model(new_model):
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())
    return new_model

def deep_cnn_model():
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((500, 500, 3)),
        tf.keras.layers.experimental.preprocessing.Resizing(125,125),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return compile_model(new_model)

def save_model(model, name, history, test_data):
    test_loss, test_acc = model.evaluate(test_data)

    # save model information
    save_name = f'../models/beans/{name}-{len(history.epoch):02d}-{test_acc:0.4f}'
    model.save(f'{save_name}.h5')

    # save history information
    save_history(history, save_name)

if __name__ == '__main__':
    data_dir = retrieve_data()

    train_data, valid_data, test_data = wrangle_data(data_dir)

    model_name = 'deep_cnn'
    model = deep_cnn_model()

    earlystop = EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=f'../model/ckpts/beans/{model_name}-'+'{epoch:02d}-{val_accuracy:.4f}')

    history = model.fit(train_data, validation_data=valid_data, epochs=25, callbacks=[earlystop,checkpoint])

    save_model(model, model_name, history, test_data)