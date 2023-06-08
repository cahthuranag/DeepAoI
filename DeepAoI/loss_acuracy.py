import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, GaussianNoise, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K


def train_and_plot_metrics(n, k, data, epochs=200, batch_size=32, validation_split=0.2):
    """
    Train the autoencoder and plot the training and validation loss and accuracy.
    Args:
        n: Block length of the code.
        k: Message length of the code.
        data: Training data.
        epochs: Number of epochs to train the autoencoder.
        batch_size: Batch size for training.
        validation_split: Fraction of the training data to be used as validation data.
        Returns:
        fig: Training and validation loss and accuracy plots.
    """
    np.random.seed(1)
    tf.random.set_seed(3)

    M = 2**k
    R = k / n  # code rate
    N = data.shape[0]

    label = np.random.randint(M, size=N)
    encoded_data = []
    for i in label:
        temp = np.zeros(M)
        temp[i] = 1
        encoded_data.append(temp)
    encoded_data = np.array(encoded_data)

    input_signal = Input(shape=(M,))
    encoded = Dense(M, activation='relu')(input_signal)
    encoded1 = Dense(n, activation='linear')(encoded)
    encoded2 = Lambda(lambda x: np.sqrt(n)*K.l2_normalize(x, axis=1))(encoded1)
    snr_train = 5.01187  # converted 7 dB of EbNo
    encoded3 = GaussianNoise(np.sqrt(1/(2*R*snr_train)))(encoded2)

    decoded = Dense(M, activation='relu')(encoded3)
    decoded1 = Dense(M, activation='softmax')(decoded)
    autoencoder = Model(input_signal, decoded1)
    adam = Adam(lr=0.01)
    autoencoder.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    history = autoencoder.fit(encoded_data, encoded_data, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    num_iterations = range(1, len(training_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(num_iterations, training_loss, '-', label='Training Loss')
    plt.plot(num_iterations, validation_loss, '-', label='Validation Loss')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs. Number of Iterations')
    plt.legend()
    plt.show()

    training_acc = history.history['acc']
    validation_acc = history.history['val_acc']

    plt.figure(figsize=(10, 5))
    plt.plot(num_iterations, training_acc, '-', label='Training Accuracy')
    plt.plot(num_iterations, validation_acc, '-', label='Validation Accuracy')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy vs. Number of Iterations')
    plt.legend()
    plt.show()



n = 16
k = 8
data = np.random.rand(1000, n)  # Replace with your actual data
train_and_plot_metrics(n, k, data)
