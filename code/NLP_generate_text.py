import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def retrieve_data():
    url = 'https://storage.googleapis.com/acg-datasets/tiny_frankenstein.tgz'
    cache_dir = '..'
    cache_subdir = 'data'
    tf.keras.utils.get_file('tiny_frankenstain.tgz', url, extract=True, cache_dir=cache_dir, cache_subdir=cache_subdir)

    # load data
    frank_file = f'{cache_dir}/{cache_subdir}/tiny_frankenstein.txt'

    with open(frank_file, 'r') as f:
        frank_data = f.read().lower()

    return frank_data

def wrangle_data(sequences, examples, batch_size):
    examples = examples + 1
    seq_expand = tf.expand_dims(sequences, -1)
    dataset = tf.data.Dataset.from_tensor_slices(seq_expand)
    dataset = dataset.window(examples, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda b: b.batch(examples))
    dataset = dataset.map(lambda x: (x[:-1], x[-1]))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def bd_rnn(token_count, sequence_length):
    new_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(token_count, 32, input_length=sequence_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dense(token_count, activation='softmax')
    ])
    return compile_model(new_model)

def compile_model(new_model):
    new_model.compile(optimizer=tf.keras.optimizers.Adam(0.03), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())
    return new_model

if __name__ == '__main__':
    data = retrieve_data()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([data])
    known_words = len(tokenizer.word_index)
    total_tokens = known_words + 1  # padding token

    # convert text into tokens
    frank_tokens = tokenizer.texts_to_sequences([data])[0]

    seq_length = 72
    train_data = wrangle_data(frank_tokens, seq_length, 64)

    model = bd_rnn(total_tokens, seq_length)
    history = model.fit(train_data, epochs=1)  # epochs 10

    model.save('../models/frankenstein.h5')

    # Predict Text
    token_lookup = {v: k for k,v in tokenizer.word_index.items()}

    seed = frank_tokens[-seq_length:]
    seed_text = ''

    for t in seed:
        seed_text += token_lookup[t] + " "
    print(seed_text)

    gen_tokens = 50

    output = []

    for _ in range(gen_tokens):
        tokens = pad_sequences([seed], maxlen=seq_length, padding='pre', truncating='pre')
        prediction = model.predict(tokens)
        next_token = np.argmax(prediction)
        output.append(token_lookup[next_token + 1])
        seed.append(next_token)

    print(' '.join(output))