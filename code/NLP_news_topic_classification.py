import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, GlobalAvgPool1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from history import plot_history, save_history

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def retrieve_data():
    (ag_news_train, ag_news_test), info = tfds.load('ag_news_subset', split=['train','test'], with_info=True)
    return ag_news_train, ag_news_test, info

def split_feature_labels(dataset):
    features = []
    labels = []
    for ex in dataset:
        features.append(ex['title'].numpy())
        labels.append(ex['label'].numpy())
    features = np.array([x.decode('utf-8') for x in features])
    labels = np.array([float(x) for x in labels])
    return features, labels

def wrangle_data(tokenizer, features, seq_length):
    tokens = tokenizer.texts_to_sequences(features)
    features_padded = pad_sequences(tokens, maxlen=seq_length, padding='post', truncating='post')
    return np.array(features_padded), tokens

def dnn_model(word_dim, embedding_dim, seq_length):
    new_model = tf.keras.Sequential([
        Embedding(word_dim, embedding_dim, input_length=seq_length),
        GlobalAvgPool1D(),
        Dense(32, activation='relu'),
        Dense(4, activation='softmax')
    ])
    return compile_model(new_model)

def compile_model(new_model):
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())
    return new_model

def save_model(model, name, history, test_data, test_labels):
    test_loss, test_acc = model.evaluate(test_data, test_labels)

    # save model information
    save_name = f'../models/news/{name}-{len(history.epoch):02d}-{test_acc:0.4f}'
    model.save(f'{save_name}.h5')

    # Save history information
    save_history(history, save_name)

if __name__ == '__main__':

    train_ds, test_ds, ds_info = retrieve_data()

    train_titles, train_labels = split_feature_labels(train_ds)
    test_titles, test_labels = split_feature_labels(test_ds)

    word_dimension = 7000
    sequence_length =24

    tokenizer = Tokenizer(num_words=word_dimension, oov_token='~~~')
    tokenizer.fit_on_texts(train_titles)

    train_data, train_tokens = wrangle_data(tokenizer, train_titles, sequence_length)
    test_data, test_tokens = wrangle_data(tokenizer, test_titles, sequence_length)

    model_name = 'dnn'
    embedding_dimenstion = 9

    earlystop = EarlyStopping('val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=f'../models/ckpts/news/{model_name}/'+'{epoch:02d}-{val_accuracy:.4f}')

    model = dnn_model(word_dimension, embedding_dimenstion, sequence_length)
    history = model.fit(train_data,train_labels, validation_split=0.1, batch_size=64, epochs=25, callbacks=[earlystop, checkpoint])

    plot_history(history)

    save_model(model, model_name, history, test_data, test_labels)