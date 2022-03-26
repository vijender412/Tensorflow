import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from history import plot_history, save_history

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def wrangle_data(ds, shuffle=False, augment=False):
    batch_size = 32
    AUTOTUNE = tf.data.AUTOTUNE

    ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)

def compile_model(new_model):
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())
    return new_model

def deep_cnn_model():
    # model = tf.keras.Sequential([
    #     tf.keras.layers.InputLayer((150, 150, 3)),
    #     tf.keras.layers.experimental.preprocessing.Resizing(125,125),
    #     tf.keras.layers.Conv2D(64, 3, activation='relu'),
    #     tf.keras.layers.MaxPool2D(),
    #     tf.keras.layers.Conv2D(64, 3, activation='relu'),
    #     tf.keras.layers.MaxPool2D(),
    #     tf.keras.layers.Conv2D(128, 3, activation='relu'),
    #     tf.keras.layers.MaxPool2D(),
    #     tf.keras.layers.Conv2D(128, 3, activation='relu'),
    #     tf.keras.layers.MaxPool2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(256, activation='relu'),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(5, activation='softmax')
    # ])
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(150,150,3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    return compile_model(model)

def save_model(model, name, history, test_data):
    test_loss, test_acc = model.evaluate(test_data)

    # save model information
    save_name = f'../models/beans/{name}-{len(history.epoch):02d}-{test_acc:0.4f}'
    model.save(f'{save_name}.h5')

    # save history information
    save_history(history, save_name)

if __name__ == '__main__':

    (train_ds, val_ds, test_ds), info = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )

    print(info)
    print(info.features['label'].names)

    # print(tfds.show_examples(train_ds, info))  # for visualization

    IMG_SIZE = 150

    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
    ])

    train_data = wrangle_data(train_ds, shuffle=True, augment=True)
    valid_data = wrangle_data(val_ds)
    test_data = wrangle_data(test_ds)


    model_name = 'deep_cnn'
    model = deep_cnn_model()

    earlystop = EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=f'../models/ckpts/flowers/{model_name}-'+'{epoch:02d}-{val_accuracy:.4f}')
    TB = tf.keras.callbacks.TensorBoard(f'../models/tensorboard/flowers/{model_name}-'+'tensorboard')

    history = model.fit(train_data, validation_data=valid_data, epochs=5, callbacks=[earlystop,checkpoint,TB])

    save_model(model, model_name, history, test_data)

    # Run this on cmd
    # from tensorboard import program
    # tb = program.TensorBoard()
    # tb.configure(argv=[None, '--logdir', tracking_address])
    # url = tb.launch()