def deepencoder (n,k,snr):
 """Deep encoder for AWGN channel
 Args:
 n (int): Number of bits in a block.
 k (int): Number of bits in a message.
 snr (float): SNR in linear scale.
 Returns:
 simulation AAoI.
    """
 import numpy as np
 import tensorflow as tf
 import keras
 from keras.layers import Input, Dense, GaussianNoise,Lambda,Dropout
 from keras.models import Model
 from keras import regularizers
 from tensorflow.keras.layers import BatchNormalization
 from keras.optimizers import Adam,SGD
 from keras import backend as K
 import numpy as np
 import tensorflow as tf
 np.random.seed(1)
 tf.random.set_seed(3)
 M=2**k  
 R = k/n # code rate
 N = 1000 
 label = np.random.randint(M,size=N)
 data = []
 for i in label:
     temp = np.zeros(M)
     temp[i] = 1
     data.append(temp)
 data = np.array(data)
 input_signal = Input(shape=(M,))
 encoded = Dense(M, activation='relu')(input_signal)
 encoded1 = Dense(n, activation='linear')(encoded)
 encoded2 = Lambda(lambda x: np.sqrt(n)*K.l2_normalize(x,axis=1))(encoded1)
 snr_train = 5.01187 #  coverted 7 db of EbNo
 encoded3 = GaussianNoise(np.sqrt(1/(2*R*snr_train)))(encoded2)
 
 decoded = Dense(M, activation='relu')(encoded3)
 decoded1 = Dense(M, activation='softmax')(decoded)
 autoencoder = Model(input_signal, decoded1)
 adam = Adam(lr=0.01)
 autoencoder.compile(optimizer=adam, loss='categorical_crossentropy')
 autoencoder.fit(data, data,
                 epochs=300,
                 batch_size=32)
 from keras.models import load_model
 encoder = Model(input_signal, encoded2)
 encoded_input = Input(shape=(n,))
 deco = autoencoder.layers[-2](encoded_input)
 deco = autoencoder.layers[-1](deco)
 decoder = Model(encoded_input, deco)
 N = 10000
 test_label = np.random.randint(M,size=N)
 test_data = []
 for i in test_label:
     temp = np.zeros(M)
     temp[i] = 1
     test_data.append(temp)    
 test_data = np.array(test_data)
 # checking generated data
 temp_test = 6
 noise_std = np.sqrt(1/(2*R*snr))
 no_errors = 0
 noise = noise_std * np.random.randn(N,n)
 encoded_signal = encoder.predict(test_data) 
 final_signal = encoded_signal + noise
 pred_final_signal =  decoder.predict(final_signal)
 pred_output = np.argmax(pred_final_signal,axis=1)
 no_errors = (pred_output != test_label)
 no_errors =  no_errors.astype(int).sum()
 ber= no_errors / N
 return ber  

def ber_bpsk(snr, block_size):
    """ 
    Calculate the Block Error Rate (BLER) for uncoded BPSK.
    Args:
        snr (float): SNR in linear scale.
        block_size (int): Block size of the code.
    Returns:
        bler (float): Block Error Rate (BLER).
    """
    import numpy as np
    
    # Convert SNR from linear to dB
    snr_db = 10 * np.log10(snr)
    
    # Calculate the Bit Error Rate (BER) for uncoded BPSK
    ber = 0.5 * np.exp(-0.1 * snr_db)
    
    # Calculate the Block Error Rate (BLER) from BER
    bler = 1 - (1 - ber) ** block_size
    
    return bler


