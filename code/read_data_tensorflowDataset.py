import os
import tensorflow as tf
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def wrangle_data(dataset, split):
    wrangled = dataset.map(lambda img,label: (tf.cast(img, tf.float32) / 255.0, label))
    wrangled = wrangled.cache()
    if split == 'train':
        wrangled = wrangled.shuffle(60000)

    return wrangled.batch(64).prefetch(tf.data.AUTOTUNE)

def compile_model(new_model):
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())
    return new_model

def create_model():
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((28,28,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return compile_model(new_model)

if __name__ == '__main__':
    mnist_train,info = tfds.load('mnist', split='train', as_supervised=True, with_info=True)
    mnist_test = tfds.load('mnist', split='test', as_supervised=True)

    print(info)
    print(info.features['label'].names)

    # print(tfds.show_examples(mnist_train, info))  # for visualization

    train_data = wrangle_data(mnist_train, 'train')
    test_data = wrangle_data(mnist_test, 'test')

    model = create_model()

    history = model.fit(train_data, epochs=5)

    print(model.evaluate(test_data))

    # save model
    model.save('../models/mnist.h5')