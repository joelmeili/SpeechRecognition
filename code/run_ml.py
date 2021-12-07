# importing packages
import os, glob
import numpy as np
import tensorflow as tf
from itertools import chain
from tensorflow import keras, lite

# defining path to data
path = "../data/"
subjects = os.listdir(path)
SAMPLING_RATE = 44100

# getting all relevant data files
silent_files = [glob.glob(path + subject + "/silent/*.wav") for subject in subjects]
silent_files = list(chain(*silent_files))

speaking_files = [glob.glob(path + subject + "/read/*.wav") for subject in subjects]
speaking_files = list(chain(*speaking_files))

singing_files = [glob.glob(path + subject + "/sing/*.wav") for subject in subjects]
singing_files = list(chain(*singing_files))

audio_files = silent_files + speaking_files + singing_files
class_names = ["silent", "speaking", "singing"]
labels = ["silent"] * len(silent_files) + ["speaking"] * len(speaking_files) + ["singing"] * len(singing_files)
labels = [0 if label == "silent" else 1 if label == "speaking" else 2 for label in labels]

# shuffling the data
SHUFFLE_SEED = 2021
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_files)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

# determining percentage of validation split
VALID_SPLIT = 0.3
BATCH_SIZE = 128
EPOCHS = 30

# generating data set
def paths_and_labels_to_dataset(audio_paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    return tf.data.Dataset.zip((audio_ds, label_ds))

def path_to_audio(path):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, 2 * SAMPLING_RATE)
    
    return audio

def audio_to_fft(audio):
    audio = tf.squeeze(audio, axis = -1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real = audio, imag = tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis = -1)
    
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

# splitting data into training and validation
idx = int(VALID_SPLIT * len(audio_files))
train_audio_files = audio_files[:-idx]
train_labels = labels[:-idx]

valid_audio_files = audio_files[-idx:]
valid_labels = labels[-idx:]

# creating the train and validation data set
train_ds = paths_and_labels_to_dataset(train_audio_files, train_labels)
train_ds = train_ds.shuffle(buffer_size = BATCH_SIZE * 8, seed = SHUFFLE_SEED).batch(
    BATCH_SIZE
)

valid_ds = paths_and_labels_to_dataset(valid_audio_files, valid_labels)
valid_ds = valid_ds.shuffle(buffer_size = 32 * 8, seed = SHUFFLE_SEED).batch(32)

# transforming audio wave to the frequency domain
train_ds = train_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls = tf.data.AUTOTUNE
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

valid_ds = valid_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls = tf.data.AUTOTUNE
)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

# building model
def residual_block(x, filters, conv_num = 3, activation = "relu"):
    # Shortcut
    s = keras.layers.Conv1D(filters, 1, padding = "same")(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding = "same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding = "same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size = 2, strides = 2)(x)


def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape = input_shape, name = "input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = keras.layers.AveragePooling1D(pool_size = 3, strides = 3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation = "relu")(x)
    x = keras.layers.Dense(128, activation = "relu")(x)

    outputs = keras.layers.Dense(num_classes, activation = "softmax", name = "output")(x)

    return keras.models.Model(inputs = inputs, outputs = outputs)

# run model
model = build_model((SAMPLING_RATE, 1), len(class_names))
model.summary()

# compiling the model using Adam's default learning rate
model.compile(
    optimizer = "Adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"]
)

# running the model
history = model.fit(
    train_ds,
    epochs = EPOCHS,
    validation_data = valid_ds
)

# saving the  model
model_path = "../SEALMP4/app/src/main/assets/"
model.save(model_path + "trained_model.h5")
converter = lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()

with open(model_path + "trained_model.tflite", "wb") as f:
    f.write(tfmodel)